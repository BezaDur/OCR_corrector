import unittest

from test_page1_fixes import TARGETS, check_targets, run_benchmark


class FakeCorrector:
    def __init__(self, repair=True):
        self.repair = repair
        self.cleared = False
        self.sentence_calls = 0
        self.saw_profile = False

    def clear_caches(self):
        self.cleared = True

    def correct_sentence(self, text):
        # Every cold pass must be explicitly cleared; warm pass must not be.
        assert self.cleared == (self.sentence_calls % 2 == 0)
        assert not self.saw_profile
        self.cleared = False
        self.sentence_calls += 1
        if self.repair:
            for observed, expected in TARGETS.items():
                text = text.replace(observed, expected)
        return text

    def generate_candidates(self, token):
        self.saw_profile = True
        candidates = [(token, 0)]
        if self.repair and token in TARGETS:
            candidates.append((TARGETS[token], 1))
        return candidates


class PhaseATests(unittest.TestCase):
    def test_targets_checked_at_position(self):
        checks = check_targets('አና አና', 'እና አና')
        self.assertEqual([check['passed'] for check in checks], [True, False])
        self.assertFalse(check_targets('አና ቤት', 'እና')[0]['passed'])

    def test_pass_and_cache_separation(self):
        report = run_benchmark(FakeCorrector(), repeats=2)
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['sentences'][0]['cold_ms']), 2)
        self.assertEqual(len(report['sentences'][0]['warm_ms']), 2)
        # First sentence has two conjunction positions plus prefix repair.
        self.assertEqual(len(report['sentences'][0]['checks']), 12)

    def test_failure_is_recorded(self):
        report = run_benchmark(FakeCorrector(repair=False), repeats=1)
        self.assertFalse(report['passed'])
        self.assertFalse(all(report['target_candidate_recall'].values()))

    def test_invalid_repeats(self):
        with self.assertRaises(ValueError):
            run_benchmark(FakeCorrector(), repeats=0)


if __name__ == '__main__':
    unittest.main()
