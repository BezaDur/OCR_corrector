import contextlib
import importlib.util
import io
import itertools
from pathlib import Path
import tempfile
import unittest

from ocr_decision_audit import audit_tokens, DECISION_STATUSES
from ocr_metrics import align_tokens

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('decision_cli', ROOT / '5_correction_evaluation' / '05_collect_error_cases.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def all_paths(left, right, i=0, j=0):
    if i == len(left) and j == len(right):
        yield 0, 0, ()
    if i < len(left) and j < len(right):
        match = left[i] == right[j]
        for cost, matches, steps in all_paths(left, right, i + 1, j + 1):
            yield cost + (not match), matches + match, ((i, j),) + steps
    if i < len(left):
        for cost, matches, steps in all_paths(left, right, i + 1, j):
            yield cost + 1, matches, ((i, None),) + steps
    if j < len(right):
        for cost, matches, steps in all_paths(left, right, i, j + 1):
            yield cost + 1, matches, ((None, j),) + steps


class DecisionAuditTests(unittest.TestCase):
    def test_mapping_options_against_exhaustive_oracle(self):
        sequences = [seq for n in range(4) for seq in itertools.product('ab', repeat=n)]
        for left in sequences:
            for right in sequences:
                paths = list(all_paths(left, right))
                best = min((cost, -matches) for cost, matches, _ in paths)
                optimal = [steps for cost, matches, steps in paths if (cost, -matches) == best]
                ref = [set() for _ in left]
                hyp = [set() for _ in right]
                insertions, deletions = set(), set()
                complex_ref, complex_hyp = set(), set()
                for steps in optimal:
                    row = col = 0
                    for i, j in steps:
                        if i is None:
                            insertions.add(row)
                        else:
                            ref[i].add(j)
                            row += 1
                        if j is None:
                            deletions.add(col)
                        else:
                            hyp[j].add(i)
                            col += 1
                    for first, second in zip(steps, steps[1:]):
                        def match(step):
                            i, j = step
                            return i is not None and j is not None and left[i] == right[j]
                        if not match(first) and not match(second):
                            for i, j in (first, second):
                                if i is not None:
                                    complex_ref.add(i)
                                if j is not None:
                                    complex_hyp.add(j)
                details = align_tokens(left, right, include_details=True).details
                with self.subTest(left=left, right=right):
                    self.assertEqual(details.reference_options, tuple(frozenset(s) for s in ref))
                    self.assertEqual(details.hypothesis_options, tuple(frozenset(s) for s in hyp))
                    self.assertEqual(details.insertion_slots, insertions)
                    self.assertEqual(details.deletion_slots, deletions)
                    self.assertEqual(details.complex_reference, complex_ref)
                    self.assertEqual(details.complex_hypothesis, complex_hyp)
                    self.assertIn(details.steps, optimal)

    def test_isolated_decisions(self):
        cases = [('g', 'g', 'g', 'TRUE_NEGATIVE'),
                 ('g', 'r', 'g', 'TRUE_POSITIVE'),
                 ('g', 'g', 'c', 'FALSE_POSITIVE'),
                 ('g', 'r', 'r', 'FALSE_NEGATIVE'),
                 ('g', 'r', 'c', 'SUBOPTIMAL_FLIP')]
        for g, r, c, status in cases:
            events = audit_tokens(['a', g, 'z'], ['a', r, 'z'], ['a', c, 'z'])
            self.assertEqual(events[1].status, status)

    def test_insertion_does_not_shift_later_tokens(self):
        events = audit_tokens('a b c'.split(), 'x a b c'.split(), 'x a b c'.split())
        self.assertEqual([e.status for e in events[:3]], ['WORD_BOUNDARY', 'TRUE_NEGATIVE', 'TRUE_NEGATIVE'])
        self.assertEqual(events[1].raw_index, 2)
        self.assertEqual([e.status for e in events[3:]], ['RAW_UNMATCHED', 'CORRECTED_UNMATCHED'])

    def test_duplicate_alignment_is_not_a_confident_damage(self):
        events = audit_tokens(['a', 'a'], ['a', 'a'], ['a'])
        self.assertTrue(all(e.status == 'AMBIGUOUS_ALIGNMENT' for e in events))

    def test_empty_and_unmatched_tails_are_retained(self):
        self.assertEqual(audit_tokens([], [], []), [])
        events = audit_tokens([], ['x'], ['y'])
        self.assertEqual([e.status for e in events], ['RAW_UNMATCHED', 'CORRECTED_UNMATCHED'])
        events = audit_tokens(['a', 'b'], ['a'], ['a'])
        self.assertEqual(len(events), 2)
        self.assertTrue(all(e.status == 'WORD_BOUNDARY' for e in events))

    def test_split_and_equal_length_boundary_shift_excluded(self):
        events = audit_tokens(['ab', 'z'], ['a', 'b', 'z'], ['a', 'b', 'z'])
        self.assertFalse(any(e.status == 'FALSE_NEGATIVE' for e in events))
        events = audit_tokens(['ab', 'c', 'z'], ['a', 'bc', 'z'], ['a', 'bc', 'z'])
        self.assertEqual([e.status for e in events[:2]], ['WORD_BOUNDARY', 'WORD_BOUNDARY'])

    def test_multiword_replacement_not_forced_into_token_typos(self):
        events = audit_tokens(['a', 'b'], ['x', 'y'], ['x', 'y'])
        self.assertTrue(all(e.status == 'COMPLEX_EDIT_BLOCK' for e in events))

    def test_reordering_not_forced_into_single_token_categories(self):
        events = audit_tokens(['minister', 'prime'], ['prime', 'minister'], ['prime', 'minister'])
        self.assertFalse(any(e.status in DECISION_STATUSES[1:] for e in events))

    def test_unchanged_output_never_a_flip_or_damage(self):
        sequences = [seq for n in range(3) for seq in itertools.product('ab', repeat=n)]
        for gold in sequences:
            for raw in sequences:
                events = audit_tokens(gold, raw, raw)
                self.assertFalse(any(e.status in ('TRUE_POSITIVE', 'FALSE_POSITIVE', 'SUBOPTIMAL_FLIP') for e in events))

    def run_fixture(self, folder, normalization='strict', extra=()):
        root = Path(folder)
        return cli.main(['--gt-dir', folder, '--raw-dir', folder, '--corrected-dir', folder,
                         '--output-tsv', str(root / 'events.tsv'), '--output-summary', str(root / 'summary.txt'),
                         '--normalization', normalization, *extra])

    def test_cli_normalization_and_full_tsv(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            for suffix, text in [('gt', 'ቤት፡፡'), ('ocr', 'ቤት፡፡'), ('corr', 'ቤት።')]:
                (root / f'p.{suffix}.txt').write_text(text, encoding='utf-8')
            stats = self.run_fixture(folder)
            self.assertEqual(stats['FALSE_POSITIVE'], 1)
            stats = self.run_fixture(folder, 'canonical')
            self.assertEqual(stats['TRUE_NEGATIVE'], 1)
            self.assertIn('Normalization: canonical', (root / 'summary.txt').read_text())
            self.assertIn('GOLD_INDEX', (root / 'events.tsv').read_text())
            # Current canonical policy does not conflate slash and dot abbreviations.
            for suffix, text in [('gt', 'ዓ/ም'), ('ocr', 'ዓ/ም'), ('corr', 'ዓ.ም')]:
                (root / f'p.{suffix}.txt').write_text(text, encoding='utf-8')
            self.assertEqual(self.run_fixture(folder, 'canonical')['FALSE_POSITIVE'], 1)

    def test_missing_and_page_count_fail_before_outputs(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            (root / 'p.gt.txt').write_text('a')
            with self.assertRaisesRegex(ValueError, 'Missing'):
                self.run_fixture(folder)
            self.assertFalse((root / 'events.tsv').exists())
            (root / 'p.ocr.txt').write_text('a')
            (root / 'p.corr.txt').write_text('a')
            with self.assertRaisesRegex(ValueError, 'Expected 2'):
                self.run_fixture(folder, extra=('--expected-pages', '2'))
            (root / 'missing.gt.txt').write_text('a')
            self.run_fixture(folder, extra=('--allow-missing',))
            self.assertIn('Missing pages: missing', (root / 'summary.txt').read_text())

    def test_outputs_cannot_overwrite_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for suffix in ('gt', 'ocr', 'corr'):
                (root / f'p.{suffix}.txt').write_text('a')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                self.run_fixture(folder, extra=('--output-tsv', str(root / 'p.gt.txt')))
            self.assertEqual((root / 'p.gt.txt').read_text(), 'a')


if __name__ == '__main__':
    unittest.main()
