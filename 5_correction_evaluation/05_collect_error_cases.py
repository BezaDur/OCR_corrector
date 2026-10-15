#!/usr/bin/env python3
"""Alignment-derived decision audit; no KenLM or morphology required."""
# Make the repo-root shared modules importable from this subfolder.
# --- make repo_paths importable from any working directory -------------------
import os as _os
import sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(
        _os.path.join(_root, "tools", "stage_paths.py")):
    _root = _os.path.dirname(_root)
# tools/ first so stage_paths and repo_paths resolve, then register every stage
# folder so cross-stage imports work by bare module name. This replaces the
# thirteen root-level shims that used to forward those imports by hand.
_sys.path.insert(0, _os.path.join(_root, "tools"))
import stage_paths  # noqa: E402  (imported after the path bootstrap)
stage_paths.register(_root)
import repo_paths  # noqa: E402
# ----------------------------------------------------------------------------

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import collections
import csv
from dataclasses import asdict
from pathlib import Path

from amharic_canonical_normalizer import normalize_for_metric
from audit_decision import DECISION_STATUSES, audit_tokens

DATASET_DIR = Path(repo_paths.workspace("synthetic_eval"))


def align_three_way(gt_tokens, raw_tokens, corr_tokens):
    """Compatibility wrapper returning focused five-field event tuples.

    Unlike the old positional comparison, unresolved rows have explicit status
    labels and unmatched tails are retained. Use audit_tokens for index/options
    metadata or the CLI for full TSV output including true negatives.
    """
    return [(e.status, e.gold, e.raw, e.corrected, e.context)
            for e in audit_tokens(gt_tokens, raw_tokens, corr_tokens)
            if e.status != 'TRUE_NEGATIVE']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-dir', type=Path, default=DATASET_DIR / 'ground_truth')
    parser.add_argument('--raw-dir', type=Path, default=DATASET_DIR / 'ocr_new_model')
    parser.add_argument('--corrected-dir', type=Path, default=DATASET_DIR / 'ocr_corrected')
    parser.add_argument('--output-tsv', type=Path, default=Path('benchmark_results/correction_decisions.tsv'))
    parser.add_argument('--output-summary', type=Path, default=Path('benchmark_results/correction_patterns_report.txt'))
    parser.add_argument('--normalization', choices=('strict', 'canonical'), default='strict')
    parser.add_argument('--expected-pages', type=int)
    parser.add_argument('--allow-missing', action='store_true', help='Skip missing pairs, reporting every skip')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    gt_files = sorted(args.gt_dir.glob('*.gt.txt'))
    if not gt_files:
        raise ValueError(f'No ground truth files in {args.gt_dir}')
    pages, missing = [], []
    for gt in gt_files:
        base = gt.name[:-len('.gt.txt')]
        raw = args.raw_dir / (base + '.ocr.txt')
        corrected = args.corrected_dir / (base + '.corr.txt')
        if not raw.exists() or not corrected.exists():
            missing.append(base)
        else:
            pages.append((base, gt, raw, corrected))
    if missing and not args.allow_missing:
        raise ValueError('Missing raw/corrected pairs: ' + ', '.join(missing))
    if not pages:
        raise ValueError('No complete page triples')
    if args.expected_pages is not None and len(pages) != args.expected_pages:
        raise ValueError(f'Expected {args.expected_pages} complete pages; found {len(pages)}')
    # Do not allow accidental overwrites of source data or the TSV by summary.
    outputs = {args.output_tsv.resolve(), args.output_summary.resolve()}
    if len(outputs) != 2 or any(path.resolve() in outputs for _, *paths in pages for path in paths):
        raise ValueError('Outputs must be distinct and must not overwrite input files')
    for path in (args.output_tsv, args.output_summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    patterns = {status: collections.Counter() for status in DECISION_STATUSES if status != 'TRUE_NEGATIVE'}
    token_counts = collections.Counter()
    fields = ['PAGE', 'STATUS', 'GOLD_INDEX', 'RAW_INDEX', 'CORRECTED_INDEX',
              'GROUND_TRUTH', 'RAW_OCR', 'CORRECTED', 'CONTEXT', 'RAW_OPTIONS', 'CORRECTED_OPTIONS']
    with args.output_tsv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(fields)
        for base, *paths in pages:
            texts = [path.read_text(encoding='utf-8') for path in paths]
            if args.normalization == 'canonical':
                texts = [normalize_for_metric(text) for text in texts]
            tokens = [text.split() for text in texts]
            for name, words in zip(('gold', 'raw', 'corrected'), tokens):
                token_counts[name] += len(words)
            for event in audit_tokens(*tokens):
                stats[event.status] += 1
                writer.writerow([base, *asdict(event).values()])
                if event.status in patterns:
                    patterns[event.status][(event.gold, event.raw, event.corrected)] += 1
    lines = [
        'AMHARIC OCR ALIGNMENT-DERIVED DECISION AUDIT',
        f'Normalization: {args.normalization}',
        f'Complete pages: {len(pages)}; skipped missing pairs: {len(missing)}',
        'Missing pages: ' + (', '.join(missing) or 'none'),
        f"Token totals: gold={token_counts['gold']}, raw={token_counts['raw']}, corrected={token_counts['corrected']}",
        f'TSV rows (gold positions plus unmatched raw/corrected rows): {sum(stats.values())}',
        'Indices/options are zero-based; <gap> denotes a missing token.',
        'Only invariant one-to-one, cycle-consistent, isolated edits enter TP/FP/FN/flip counts.',
        'DP objective: minimum word edits, then maximum exact matches.',
        'Unresolved rows show representative tokens; consult mapping options, not just the token columns.',
        'These are alignment-derived diagnostics, NOT causal edit provenance or corpus CER/WER.',
        'Boundary/complex/ambiguous events remain in the TSV and are excluded from single-token counts.',
        'RAW/CORRECTED_UNMATCHED rows can describe the same event; do not sum them as edit counts.',
        '',
    ]
    for status in DECISION_STATUSES:
        lines.append(f'{status:28s}: {stats[status]:,}')
    for status in sorted(set(stats) - set(DECISION_STATUSES)):
        lines.append(f'{status:28s}: {stats[status]:,}')
    for status in ('FALSE_POSITIVE', 'TRUE_POSITIVE', 'FALSE_NEGATIVE', 'SUBOPTIMAL_FLIP'):
        lines.extend(['', f'TOP 20 {status} PATTERNS (gold | raw | corrected)'])
        for (g, r, c), count in patterns[status].most_common(20):
            lines.append(f'{count:4d}x | Gold: {g!r} | Raw: {r!r} | Corrected: {c!r}')
    summary = '\n'.join(lines) + '\n'
    args.output_summary.write_text(summary, encoding='utf-8')
    print(summary)
    print(f'TSV: {args.output_tsv}\nSummary: {args.output_summary}')
    return dict(stats)


if __name__ == '__main__':
    main()
