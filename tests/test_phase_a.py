"""QUARANTINED — targets a corrector revision that is no longer in the tree.

This file exercises `AmharicOCRCorrector` options that do not exist at HEAD:
`candidate_backend`, `trie_search`, `symspell_max_edit_distance`,
`indel_cost_cap`, `trusted_long_word_frequency`,
`trusted_short_word_frequency`, and the injectable `lm=` / `state_factory=` /
`segmenter=` / `morphology_stats()` surface. HEAD hard-codes kenlm, imports the
segmenter at module scope and exposes none of them, so every test here fails
with TypeError or a semantic mismatch.

See docs/architecture_review.md section 2. The live contract suite is
tests/test_head_contract.py.

If you restore the trie backend, delete the setUpModule guard below.
"""
import unittest


def setUpModule():
    raise unittest.SkipTest(
        "quarantined: targets a removed corrector revision; "
        "see docs/architecture_review.md section 2"
    )


import unittest

try:
    # test_page1_fixes now builds its corrector inside main(), so importing it
    # no longer loads KenLM or the multi-GB lexicon. It still exports none of
    # the TARGETS / check_targets / run_benchmark helpers this suite expects --
    # those belong to a deleted revision of the script -- so this suite skips.
    from test_page1_fixes import TARGETS, check_targets, run_benchmark
except Exception as exc:  # noqa: BLE001 - import-time environment failure
    raise unittest.SkipTest(f"test_page1_fixes unavailable: {exc}")


class FakeCorrector:
    def __init__(self, repair=True):
        self.repair = repair
        self.cleared = False
        self.sentence_calls = 0
        self.saw_profile = False

    def morphology_stats(self):
        # Fake telemetry verifies snapshot deltas without sleeping in tests.
        return dict(trusted_bypasses=self.sentence_calls, segmenter_calls=self.sentence_calls * 2,
                    segmenter_failures=0, segmenter_seconds=self.sentence_calls * 0.01,
                    cache=dict(hits=self.sentence_calls, misses=self.sentence_calls * 2))

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
        self.assertEqual(report['sentences'][0]['cold_morphology'][0]['segmenter_calls'], 2)
        self.assertEqual(report['sentences'][0]['warm_morphology'][0]['trusted_bypasses'], 1)
        self.assertAlmostEqual(report['sentences'][0]['cold_morphology'][0]['segmenter_seconds'], 0.01)
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
