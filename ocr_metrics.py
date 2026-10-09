"""Reference-anchored OCR metrics, independent of KenLM and morphology.

A gold position is alignment-invariant clean if it matches exactly in every
co-optimal raw alignment (minimum word edits, then maximum exact matches).
On that fixed subset, report min/max losses attainable across co-optimal
corrected alignments. Ambiguous raw positions are excluded and counted.
These are alignment-derived estimates, NOT bounds on causal edit damage:
independent alignments cannot establish token provenance. Insertions affect WER
and are reported separately. Tokens include attached punctuation; label the
strict/canonical normalization policy. Intended for page-sized word sequences.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenAlignment:
    distance: int
    matched_reference: frozenset
    insertions: int
    deletions: int
    certain_reference: frozenset
    possible_reference: frozenset
    min_tracked_matches: int
    max_tracked_matches: int


def edit_distance(left, right):
    """Unit-cost Levenshtein distance with linear working memory."""
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, value in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (value != other)))
        previous = current
    return previous[-1]


def align_tokens(reference, hypothesis, tracked_reference=frozenset()):
    """Exact DP: two score rows plus one byte per cell for traceback.

    Scores minimize (edit count, -exact matches). Bitset intersections/unions
    track certainly/possibly matched positions across ALL tied optimal paths.
    Unlike SequenceMatcher, repeated/popular tokens have no heuristic treatment.
    """
    width = len(hypothesis) + 1
    directions = [bytearray(width) for _ in range(len(reference) + 1)]
    directions[0][1:] = bytes([2]) * (width - 1)  # insertion
    previous = [(j, 0) for j in range(width)]
    previous_certain = [0] * width
    previous_possible = [0] * width
    previous_min = [0] * width
    previous_max = [0] * width
    for i, word in enumerate(reference, 1):
        directions[i][0] = 1  # deletion
        current = [(i, 0)]
        certain = [0]
        possible = [0]
        min_matches, max_matches = [0], [0]
        bit = 1 << (i - 1)
        for j, other in enumerate(hypothesis, 1):
            match = word == other
            prior = previous[j - 1]
            diagonal = (prior[0] + (not match), prior[1] - match)
            deletion = (previous[j][0] + 1, *previous[j][1:])
            insertion = (current[-1][0] + 1, *current[-1][1:])
            choices = (diagonal, deletion, insertion)
            direction = min(range(3), key=choices.__getitem__)
            directions[i][j] = direction
            best = choices[direction]
            current.append(best)
            mask = bit if match else 0
            must_choices = (previous_certain[j-1] | mask, previous_certain[j], certain[-1])
            may_choices = (previous_possible[j-1] | mask, previous_possible[j], possible[-1])
            must, may = must_choices[direction], may_choices[direction]
            tracked = int(match and i - 1 in tracked_reference)
            min_choices = (previous_min[j-1] + tracked, previous_min[j], min_matches[-1])
            max_choices = (previous_max[j-1] + tracked, previous_max[j], max_matches[-1])
            low, high = min_choices[direction], max_choices[direction]
            for index, choice in enumerate(choices):
                if choice == best:
                    must &= must_choices[index]
                    may |= may_choices[index]
                    low = min(low, min_choices[index])
                    high = max(high, max_choices[index])
            certain.append(must)
            possible.append(may)
            min_matches.append(low)
            max_matches.append(high)
        previous = current
        previous_certain, previous_possible = certain, possible
        previous_min, previous_max = min_matches, max_matches
    distance = previous[-1][0]
    matches = set()
    insertions = deletions = 0
    i, j = len(reference), len(hypothesis)
    while i or j:
        direction = directions[i][j]
        if direction == 0:
            if reference[i - 1] == hypothesis[j - 1]:
                matches.add(i - 1)
            i -= 1
            j -= 1
        elif direction == 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    def positions(bits):
        return frozenset(i for i in range(len(reference)) if bits & (1 << i))

    return TokenAlignment(distance, frozenset(matches), insertions, deletions,
                          positions(previous_certain[-1]), positions(previous_possible[-1]),
                          previous_min[-1], previous_max[-1])


COUNT_FIELDS = (
    'reference_characters', 'raw_character_errors', 'corrected_character_errors',
    'reference_words', 'raw_word_errors', 'corrected_word_errors',
    'clean_reference_tokens', 'damaged_clean_tokens', 'max_damaged_clean_tokens',
    'ambiguous_raw_reference_tokens',
    'raw_insertions', 'corrected_insertions',
)


def score_texts(gold, raw, corrected):
    """Return additive corpus counts. All three texts must share normalization."""
    gold_words, raw_words, corr_words = gold.split(), raw.split(), corrected.split()
    raw_alignment = align_tokens(gold_words, raw_words)
    clean = raw_alignment.certain_reference
    corr_alignment = align_tokens(gold_words, corr_words, tracked_reference=clean)
    gold_chars, raw_chars, corr_chars = (''.join(text.split()) for text in (gold, raw, corrected))
    return dict(
        reference_characters=len(gold_chars),
        raw_character_errors=edit_distance(gold_chars, raw_chars),
        corrected_character_errors=edit_distance(gold_chars, corr_chars),
        reference_words=len(gold_words),
        raw_word_errors=raw_alignment.distance,
        corrected_word_errors=corr_alignment.distance,
        clean_reference_tokens=len(clean),
        damaged_clean_tokens=len(clean) - corr_alignment.max_tracked_matches,
        max_damaged_clean_tokens=len(clean) - corr_alignment.min_tracked_matches,
        ambiguous_raw_reference_tokens=len(raw_alignment.possible_reference - clean),
        raw_insertions=raw_alignment.insertions,
        corrected_insertions=corr_alignment.insertions,
    )


def percentages(counts):
    """Corpus percentages, not means of page percentages; undefined rates are None."""
    def rate(numerator, denominator):
        return 100.0 * numerator / denominator if denominator else None

    return dict(
        raw_character_accuracy=(100.0 - rate(counts['raw_character_errors'], counts['reference_characters'])
                                if counts['reference_characters'] else None),
        corrected_character_accuracy=(100.0 - rate(counts['corrected_character_errors'], counts['reference_characters'])
                                      if counts['reference_characters'] else None),
        raw_cer=rate(counts['raw_character_errors'], counts['reference_characters']),
        corrected_cer=rate(counts['corrected_character_errors'], counts['reference_characters']),
        raw_wer=rate(counts['raw_word_errors'], counts['reference_words']),
        corrected_wer=rate(counts['corrected_word_errors'], counts['reference_words']),
        clean_token_damage_rate_lower=rate(counts['damaged_clean_tokens'], counts['clean_reference_tokens']),
        clean_token_damage_rate_upper=rate(counts['max_damaged_clean_tokens'], counts['clean_reference_tokens']),
        character_error_reduction=rate(
            counts['raw_character_errors'] - counts['corrected_character_errors'],
            counts['raw_character_errors']),
    )
