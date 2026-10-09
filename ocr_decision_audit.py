"""Conservative, alignment-derived OCR decision diagnostics.

Independent pairwise alignments cannot establish causal edit provenance. A
four-way decision is emitted only for invariant one-to-one correspondences in
all three pairs, outside gap neighborhoods and multi-token edit blocks. Other
rows remain visible rather than being forced into TP/FP/FN categories.
"""
from dataclasses import dataclass

from ocr_metrics import align_tokens

DECISION_STATUSES = (
    'TRUE_NEGATIVE', 'TRUE_POSITIVE', 'FALSE_POSITIVE',
    'FALSE_NEGATIVE', 'SUBOPTIMAL_FLIP',
)


@dataclass(frozen=True)
class DecisionEvent:
    status: str
    gold_index: object
    raw_index: object
    corrected_index: object
    gold: str
    raw: str
    corrected: str
    context: str
    raw_options: str = ''
    corrected_options: str = ''


def _format_options(options):
    return ','.join(str(i) for i in sorted(x for x in options if x is not None)) + (
        ',<gap>' if None in options and len(options) > 1 else '<gap>' if None in options else '')


def _context(words, index):
    if index is None:
        return ''
    return '... ' + ' '.join(words[max(0, index - 2):index]) + ' > [' + words[index] + '] < ' + \
        ' '.join(words[index + 1:index + 3]) + ' ...'


def _pair_annotations(left, right, alignment):
    """Flag gaps in both coordinates, including start/end and all optimal paths.

    Consecutive nonmatches on the representative path are not reliable
    single-token decisions. Exact concatenation equality identifies a pure
    whitespace shift even when equal token counts produce no DP gap.
    """
    details = alignment.details
    boundary_left, boundary_right = set(), set()
    complex_left = set(details.complex_reference)
    complex_right = set(details.complex_hypothesis)
    for slot in details.insertion_slots:
        boundary_left.update(i for i in (slot - 1, slot) if 0 <= i < len(left))
    for slot in details.deletion_slots:
        boundary_right.update(j for j in (slot - 1, slot) if 0 <= j < len(right))
    for i, options in enumerate(details.reference_options):
        if None in options:
            boundary_left.update(k for k in (i - 1, i, i + 1) if 0 <= k < len(left))
    for j, options in enumerate(details.hypothesis_options):
        if None in options:
            boundary_right.update(k for k in (j - 1, j, j + 1) if 0 <= k < len(right))

    def mark_run(run):
        if not run:
            return
        indices_left = {i for i, _ in run if i is not None}
        indices_right = {j for _, j in run if j is not None}
        joined_left = ''.join(left[i] for i, _ in run if i is not None)
        joined_right = ''.join(right[j] for _, j in run if j is not None)
        if any(i is None or j is None for i, j in run) or joined_left == joined_right:
            boundary_left.update(indices_left)
            boundary_right.update(indices_right)
        elif len(run) > 1:
            complex_left.update(indices_left)
            complex_right.update(indices_right)

    run = []
    for i, j in details.steps:
        if i is not None and j is not None and left[i] == right[j]:
            mark_run(run)
            run = []
        else:
            run.append((i, j))
    mark_run(run)
    return boundary_left, boundary_right, complex_left, complex_right


def audit_tokens(gt_tokens, raw_tokens, corr_tokens):
    """Return one row per gold token plus separate raw/corrected gap rows.

    DP minimizes (word edits, -exact matches). Invariant means across all paths
    tied under BOTH objectives, not across every minimum-edit-only path.
    Insertions/deletions and uncertain correspondences are not counted as
    confident single-token repairs/damage. Indices and options are zero-based.
    """
    gr = align_tokens(gt_tokens, raw_tokens, include_details=True)
    gc = align_tokens(gt_tokens, corr_tokens, include_details=True)
    rc = align_tokens(raw_tokens, corr_tokens, include_details=True)
    gr_flags = _pair_annotations(gt_tokens, raw_tokens, gr)
    gc_flags = _pair_annotations(gt_tokens, corr_tokens, gc)
    rc_flags = _pair_annotations(raw_tokens, corr_tokens, rc)
    gr_map = {i: j for i, j in gr.details.steps if i is not None}
    gc_map = {i: j for i, j in gc.details.steps if i is not None}
    events = []
    for i, g in enumerate(gt_tokens):
        raw_options = gr.details.reference_options[i]
        corr_options = gc.details.reference_options[i]
        # Representative tokens are shown even for unresolved rows; options
        # make it explicit that these must not be read as certain mappings.
        r_index, c_index = gr_map[i], gc_map[i]
        r = raw_tokens[r_index] if r_index is not None else ''
        c = corr_tokens[c_index] if c_index is not None else ''
        ambiguous = len(raw_options) != 1 or len(corr_options) != 1
        boundary = i in gr_flags[0] or i in gc_flags[0]
        complex_edit = i in gr_flags[2] or i in gc_flags[2]
        consistent = r_index is not None and c_index is not None
        if consistent:
            inverse_raw = gr.details.hypothesis_options[r_index]
            inverse_corr = gc.details.hypothesis_options[c_index]
            forward = rc.details.reference_options[r_index]
            backward = rc.details.hypothesis_options[c_index]
            ambiguous |= any(len(s) != 1 for s in (inverse_raw, inverse_corr, forward, backward))
            consistent = (inverse_raw == {i} and inverse_corr == {i}
                          and forward == {c_index} and backward == {r_index})
            boundary |= r_index in rc_flags[0] or c_index in rc_flags[1]
            complex_edit |= r_index in rc_flags[2] or c_index in rc_flags[3]
        if ambiguous:
            status = 'AMBIGUOUS_ALIGNMENT'
        elif boundary:
            status = 'WORD_BOUNDARY'
        elif complex_edit:
            status = 'COMPLEX_EDIT_BLOCK'
        elif not consistent:
            status = 'UNRESOLVED_CORRESPONDENCE'
        elif r == g and c == g:
            status = 'TRUE_NEGATIVE'
        elif r != g and c == g:
            status = 'TRUE_POSITIVE'
        elif r == g and c != g:
            status = 'FALSE_POSITIVE'
        elif r != g and c == r:
            status = 'FALSE_NEGATIVE'
        else:
            status = 'SUBOPTIMAL_FLIP'
        events.append(DecisionEvent(status, i, r_index, c_index, g, r, c,
                                    _context(gt_tokens, i), _format_options(raw_options),
                                    _format_options(corr_options)))
    # Cover unmatched tails and insertions explicitly. These rows do not add
    # gold events or enter the four single-token decision categories.
    for prefix, words, alignment in (('RAW', raw_tokens, gr), ('CORRECTED', corr_tokens, gc)):
        for i, j in alignment.details.steps:
            if i is not None:
                continue
            ambiguous = alignment.details.hypothesis_options[j] != {None}
            status = prefix + ('_UNMATCHED_AMBIGUOUS' if ambiguous else '_UNMATCHED')
            events.append(DecisionEvent(
                status, None, j if prefix == 'RAW' else None,
                j if prefix == 'CORRECTED' else None, '',
                words[j] if prefix == 'RAW' else '', words[j] if prefix == 'CORRECTED' else '',
                prefix + ': ' + _context(words, j),
                _format_options(alignment.details.hypothesis_options[j]) if prefix == 'RAW' else '',
                _format_options(alignment.details.hypothesis_options[j]) if prefix == 'CORRECTED' else '',
            ))
    return events
