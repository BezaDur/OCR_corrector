import itertools
import unittest

from amharic_canonical_normalizer import normalize_for_metric
from ocr_metrics import align_tokens, edit_distance, score_texts, percentages


def enumerate_alignments(reference, hypothesis, i=0, j=0):
    """Independent exponential oracle used only for tiny strings."""
    if i == len(reference) and j == len(hypothesis):
        yield 0, frozenset()
        return
    if i < len(reference) and j < len(hypothesis):
        match = reference[i] == hypothesis[j]
        for cost, matches in enumerate_alignments(reference, hypothesis, i+1, j+1):
            yield cost + (not match), matches | ({i} if match else set())
    if i < len(reference):
        for cost, matches in enumerate_alignments(reference, hypothesis, i+1, j):
            yield cost + 1, matches
    if j < len(hypothesis):
        for cost, matches in enumerate_alignments(reference, hypothesis, i, j+1):
            yield cost + 1, matches


class MetricsTests(unittest.TestCase):
    def test_dp_against_exhaustive_alignment_oracle(self):
        sequences = [seq for n in range(4) for seq in itertools.product('ab', repeat=n)]
        for reference in sequences:
            tracked = frozenset(range(0, len(reference), 2))
            for hypothesis in sequences:
                alignments = list(enumerate_alignments(reference, hypothesis))
                best = min((cost, -len(matches)) for cost, matches in alignments)
                matches = [matched for cost, matched in alignments if (cost, -len(matched)) == best]
                result = align_tokens(reference, hypothesis, tracked)
                with self.subTest(reference=reference, hypothesis=hypothesis):
                    self.assertEqual(result.distance, edit_distance(reference, hypothesis))
                    self.assertEqual(result.certain_reference, frozenset.intersection(*matches))
                    self.assertEqual(result.possible_reference, frozenset.union(*matches))
                    self.assertEqual(result.min_tracked_matches, min(len(m & tracked) for m in matches))
                    self.assertEqual(result.max_tracked_matches, max(len(m & tracked) for m in matches))

    def test_one_clean_word_damaged(self):
        counts = score_texts('ቤት ነው', 'ቤት ነው', 'ቤት ነበር')
        self.assertEqual(counts['clean_reference_tokens'], 2)
        self.assertEqual(counts['damaged_clean_tokens'], 1)
        self.assertEqual(percentages(counts)['clean_token_damage_rate_upper'], 50)

    def test_insertion_does_not_shift_clean_tokens(self):
        counts = score_texts('ቤት ነው', 'ቤት ነው', 'ግብጽ ቤት ነው')
        self.assertEqual(counts['damaged_clean_tokens'], 0)
        self.assertEqual(counts['corrected_insertions'], 1)
        self.assertEqual(counts['corrected_word_errors'], 1)

    def test_deletion_of_duplicate_is_not_hidden(self):
        counts = score_texts('ቤት ቤት', 'ቤት ቤት', 'ቤት')
        self.assertEqual(counts['clean_reference_tokens'], 2)
        self.assertEqual(counts['damaged_clean_tokens'], 1)
        self.assertEqual(counts['max_damaged_clean_tokens'], 1)

    def test_ambiguous_raw_positions_are_reported(self):
        counts = score_texts('ቤት ቤት ነው', 'ቤት ነው', 'ቤት ነው')
        self.assertEqual(counts['clean_reference_tokens'], 1)
        self.assertEqual(counts['ambiguous_raw_reference_tokens'], 2)
        self.assertEqual(counts['max_damaged_clean_tokens'], 0)

    def test_unchanged_output_cannot_damage_clean_subset(self):
        sequences = [' '.join(seq) for n in range(3) for seq in itertools.product('ab', repeat=n)]
        for gold in sequences:
            for raw in sequences:
                counts = score_texts(gold, raw, raw)
                self.assertEqual(counts['damaged_clean_tokens'], 0)
                self.assertEqual(counts['max_damaged_clean_tokens'], 0)

    def test_canonicalization_applied_before_alignment(self):
        texts = ('ቤት፡ነው፡፡', 'ቤት፡ነው፡፡', 'ቤት ነው።')
        self.assertGreater(score_texts(*texts)['damaged_clean_tokens'], 0)
        canonical = score_texts(*(normalize_for_metric(text) for text in texts))
        self.assertEqual(canonical['clean_reference_tokens'], 2)
        self.assertEqual(canonical['max_damaged_clean_tokens'], 0)

    def test_undefined_rates(self):
        rates = percentages(score_texts('', '', ''))
        self.assertTrue(all(value is None for value in rates.values()))
        rates = percentages(score_texts('a', 'x', 'a'))
        self.assertIsNone(rates['clean_token_damage_rate_upper'])
        self.assertEqual(rates['character_error_reduction'], 100)

    def test_corpus_counts_not_page_average(self):
        left = score_texts('a', 'a', 'b')
        right = score_texts('a b c', 'a b c', 'a b c')
        totals = {key: left[key] + right[key] for key in left}
        self.assertEqual(percentages(totals)['clean_token_damage_rate_upper'], 25)


if __name__ == '__main__':
    unittest.main()
