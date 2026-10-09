"""Portable regression tests; fake LM tests control scoring, not language accuracy."""
import contextlib
import enum
import importlib.util
import io
import itertools
from pathlib import Path
import tempfile
import unittest

from amharic_ocr_corrector import AmharicOCRCorrector, EthiopicPunctuationManager

ROOT = Path(__file__).resolve().parents[1]


class State:
    def __init__(self):
        self.last = "<s>"


class LM:
    def __init__(self, scores=None):
        self.scores = scores or {}
        self.calls = []

    def BeginSentenceWrite(self, state):
        state.last = "<s>"

    def BaseScore(self, previous, word, next_state):
        self.calls.append((previous.last, word))
        next_state.last = word
        return self.scores.get((previous.last, word), 0.0)


class Morph:
    def __init__(self, pos="N", failures=0):
        self.pos = pos
        self.failures = failures
        self.calls = 0

    def segment_word(self, word):
        self.calls += 1
        if self.failures:
            self.failures -= 1
            raise RuntimeError("temporary failure")
        return {"pos": self.pos}


class CorrectorTests(unittest.TestCase):
    def make(self, words=(), **kwargs):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        lexicon = Path(folder.name) / 'words.txt'
        lexicon.write_text('\n'.join(f'{w} {f}' for w, f in words), encoding='utf-8')
        options = dict(kenlm_path="unused", confusion_matrix_path=str(ROOT / 'confusion_matrix.json'),
                       dict_vocab_path=str(lexicon), lm=LM(), state_factory=State, segmenter=None)
        options.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return AmharicOCRCorrector(**options)

    def test_cost_override_and_cache_key(self):
        c = self.make([('በሽታ', 1)], max_optical_cost=2.6)
        self.assertNotIn('በሽታ', dict(c.generate_candidates('በሸታ')))
        self.assertIn('በሽታ', dict(c.generate_candidates('በሸታ', max_cost=4)))
        self.assertNotIn('በሽታ', dict(c.generate_candidates('በሸታ')))
        self.assertEqual(c._generate_candidates_cached.cache_info().hits, 1)
        with self.assertRaises(ValueError):
            c.generate_candidates('በሸታ', float('nan'))

    def test_adaptive_trie_budget_is_separate_from_acceptance_ceiling(self):
        c = self.make([('ቤት', 1)])
        self.assertEqual(c._trie_radius('ቤት', 10), 2.4)
        self.assertEqual(c._trie_radius('ክተማ', 10), 2.6)
        self.assertEqual(c._trie_radius('አበከደገ', 10), 3.0)
        self.assertEqual(c._trie_radius('አበከደገለመነ', 10), 2.2)
        self.assertEqual(c._trie_radius('አበከደገ', 1), 1)
        c.trie_search = 'full'
        self.assertEqual(c._trie_radius('አበከደገ', 10), 10)

    def test_previously_unreachable_short_word(self):
        c = self.make([('ከተማ', 1)])
        self.assertAlmostEqual(dict(c.generate_candidates('ክተማ'))['ከተማ'], 2.7451)

    def test_single_fidel_admission(self):
        c = self.make([('በ', 1000), ('ከ', 1000)])
        self.assertIn('በ', c.vocab)
        self.assertIn('ከ', c.vocab)

    def test_frequent_real_word_neighbor(self):
        c = self.make([('በሽታ', 10000), ('በሸታ', 10000)])
        self.assertIn('በሸታ', c.common_words)
        self.assertIn('በሽታ', dict(c.generate_candidates('በሸታ')))

    def test_original_reserved_and_protected_abbreviation(self):
        c = self.make([('በሽታ', 10000)], max_candidates=1)
        self.assertEqual(c.generate_candidates('በሸታ'), [('በሸታ', 0)])
        self.assertEqual(c.generate_candidates('ዶ/ር', 100), [('ዶ/ር', 0)])

    def test_zero_cost_alternative_cannot_evict_original(self):
        c = self.make([('በC', 10000)], max_candidates=1)
        self.assertEqual(c.optical_distance('በC', 'በ'), 0)
        self.assertEqual(c.generate_candidates('በ', 0), [('በ', 0)])

    def test_prefix_heuristic_respects_budget_and_limit(self):
        c = self.make([('እንደሚያደርጉ', 1)], max_candidates=2)
        self.assertEqual(c.generate_candidates('አንደሚያደርጉ', 0), [('አንደሚያደርጉ', 0)])
        self.assertIn('እንደሚያደርጉ', dict(c.generate_candidates('አንደሚያደርጉ', 4)))
        c.max_candidates = 1
        c.clear_caches()
        self.assertEqual(c.generate_candidates('አንደሚያደርጉ', 4), [('አንደሚያደርጉ', 0)])

    def test_directional_optical_distance(self):
        c = self.make(indel_cost_cap=None)
        self.assertAlmostEqual(c.optical_distance('እ', 'አ'), 2.1666)
        self.assertAlmostEqual(c.optical_distance('አ', 'እ'), 6.576)
        self.assertEqual(c.optical_distance('በ', 'በ'), 0)
        self.assertEqual(c.optical_distance('', 'በ'), c._get_ins_cost('በ'))
        self.assertEqual(c.optical_distance('በ', ''), c._get_del_cost('በ'))

    def test_trie_matches_exhaustive_weighted_search(self):
        words = [''.join(chars) for n in range(1, 4) for chars in itertools.product('እአበ', repeat=n)]
        c = self.make([(word, 1) for word in words], max_candidates=100, indel_cost_cap=None, trie_search="full")
        for observed in ('እአ', 'በአ', 'አአ'):
            expected = {word: c.optical_distance(word, observed) for word in words}
            expected = {word: cost for word, cost in expected.items() if cost <= 4}
            expected[observed] = 0
            self.assertEqual(dict(c.generate_candidates(observed)), expected)

    def test_morphology_cached_and_instance_local(self):
        morph = Morph()
        c = self.make(segmenter=morph)
        for _ in range(100):
            self.assertTrue(c.validate_morphology('የቤታችን'))
        self.assertEqual(morph.calls, 1)
        other = self.make(segmenter=Morph('UNKNOWN'))
        self.assertFalse(other.validate_morphology('የቤታችን'))
        c.clear_caches()
        c.validate_morphology('የቤታችን')
        self.assertEqual(morph.calls, 2)

    def test_frequency_trust_boundaries(self):
        # "Over" is strict, not >=. Three-fidel tokens use the short-word tier.
        for word, threshold in [('ቤት', 4000), ('ከተማ', 4000), ('መንግስት', 1000),
                                ('አበከደ', 1000), ('የቤታችን', 1000)]:
            for frequency in (threshold - 1, threshold, threshold + 1):
                with self.subTest(word=word, frequency=frequency):
                    morph = Morph('UNKNOWN')
                    c = self.make([(word, frequency)], segmenter=morph)
                    trusted = frequency > threshold
                    self.assertEqual(c.is_trusted_dictionary_word(word), trusted)
                    self.assertEqual(c.validate_morphology(word), trusted)
                    self.assertEqual(morph.calls, 0 if trusted else 1)
                    self.assertEqual(word in c.common_words, trusted)

    def test_single_fidel_and_low_frequency_words_use_analyzer(self):
        morph = Morph()
        c = self.make([('በ', 100000), ('መንግስት', 1)], segmenter=morph)
        for word in ('በ', 'መንግስት', 'የቤታችን'):
            self.assertFalse(c.is_trusted_dictionary_word(word))
            for _ in range(3):
                self.assertTrue(c.validate_morphology(word))
        self.assertEqual(morph.calls, 3)
        self.assertEqual(c.morphology_stats()['cache']['hits'], 6)
        self.assertEqual(c.morphology_stats()['cache']['maxsize'], 65536)

    def test_trust_bypass_does_not_fill_deep_cache(self):
        morph = Morph('UNKNOWN')
        c = self.make([('መንግስት', 1001)], segmenter=morph)
        for _ in range(100):
            self.assertTrue(c.validate_morphology('መንግስት'))
        self.assertEqual(morph.calls, 0)
        stats = c.morphology_stats()
        self.assertEqual(stats['trusted_bypasses'], 100)
        self.assertEqual(stats['cache']['currsize'], 0)
        self.assertEqual(stats['segmenter_seconds'], 0)
        c.clear_caches()
        self.assertEqual(c.morphology_stats()['trusted_bypasses'], 100)

    def test_trust_without_segmenter_and_configurable_cutoffs(self):
        c = self.make([('ቤት', 4001), ('መንግስት', 1001), ('ከተማ', 1)])
        self.assertTrue(c.validate_morphology('ቤት'))
        self.assertTrue(c.validate_morphology('መንግስት'))
        self.assertFalse(c.validate_morphology('ከተማ'))
        c = self.make([('ቤት', 11), ('መንግስት', 21)],
                      trusted_short_word_frequency=10, trusted_long_word_frequency=20,
                      morphology_cache_size=8)
        self.assertTrue(c.validate_morphology('ቤት'))
        self.assertTrue(c.validate_morphology('መንግስት'))
        self.assertEqual(c.morphology_stats()['cache']['maxsize'], 8)

    def test_trusted_ambiguous_words_keep_neighbors(self):
        c = self.make([('አና', 10000), ('እና', 10000)], segmenter=Morph('UNKNOWN'))
        self.assertTrue(c.validate_morphology('አና'))
        self.assertNotIn('አና', c.common_words)
        self.assertIn('እና', dict(c.generate_candidates('አና')))

    def test_frequency_promotion_beats_stale_cached_rejection(self):
        morph = Morph('UNKNOWN')
        c = self.make([('መንግስት', 1000)], segmenter=morph)
        self.assertFalse(c.validate_morphology('መንግስት'))
        c._add_to_trie('መንግስት', 1001)
        self.assertTrue(c.validate_morphology('መንግስት'))
        self.assertIn('መንግስት', c.common_words)
        self.assertEqual(morph.calls, 1)

    def test_dictionary_and_morph_frequencies_are_not_added(self):
        morph = Morph('UNKNOWN')
        morph.vocab_freq = {'መንግስት': 800}
        c = self.make([('መንግስት', 900)], segmenter=morph)
        self.assertEqual(c.vocab['መንግስት'], 900)
        self.assertFalse(c.validate_morphology('መንግስት'))

    def test_sentence_scoring_bypasses_trusted_candidates_only(self):
        morph = Morph('UNKNOWN')
        c = self.make([('በሸታ', 4001), ('በሽታ', 4001)], segmenter=morph)
        c.correct_sentence('በሸታ')
        self.assertEqual(morph.calls, 0)
        self.assertGreaterEqual(c.morphology_stats()['trusted_bypasses'], 2)
        # A low-frequency alternative still needs actual analyzer validation.
        c = self.make([('በሸታ', 4001), ('በሽታ', 4000)], segmenter=morph)
        c.correct_sentence('በሸታ')
        c.correct_sentence('በሸታ')
        self.assertEqual(morph.calls, 1)

    def test_unknown_enum_and_retry(self):
        class POS(enum.Enum):
            UNKNOWN = 0
        c = self.make(segmenter=Morph(POS.UNKNOWN))
        self.assertFalse(c.validate_morphology('ቤት'))
        morph = Morph(failures=1)
        c = self.make(segmenter=morph)
        self.assertFalse(c.validate_morphology('ቤት'))
        self.assertTrue(c.validate_morphology('ቤት'))
        self.assertEqual(morph.calls, 2)
        stats = c.morphology_stats()
        self.assertEqual(stats['segmenter_calls'], 2)
        self.assertEqual(stats['segmenter_failures'], 1)
        self.assertGreaterEqual(stats['segmenter_seconds'], 0)

    def test_eos_reranks_final_word(self):
        lm = LM({('በሸታ', '</s>'): -10, ('በሽታ', '</s>'): 0})
        c = self.make([('በሽታ', 1)], lm=lm, beam_width=2)
        self.assertEqual(c.correct_sentence('በሸታ'), 'በሽታ')
        c.score_eos = False
        self.assertEqual(c.correct_sentence('በሸታ'), 'በሸታ')

    def test_surface_marks_and_backpointer_order(self):
        c = self.make([('ቤት', 10000), ('ነው', 10000)])
        self.assertEqual(c.correct_sentence('“ቤት፟ ነው።”'), '“ቤት፟ ነው።”')
        self.assertEqual(c.correct_sentence('፟'), '፟')
        self.assertEqual(c.correct_sentence(''), '')

    def test_backpointer_survives_later_context_reranking(self):
        lm = LM({('በሽታ', 'ነው'): 10})
        c = self.make([('በሽታ', 1), ('ነው', 10000)], lm=lm)
        self.assertEqual(c.correct_sentence('“በሸታ ነው።”'), '“በሽታ ነው።”')

    def test_singleton_skips_morph_reward_but_advances_lm(self):
        morph = Morph()
        lm = LM()
        c = self.make([('ቤት', 10000)], segmenter=morph, lm=lm)
        c.correct_sentence('ቤት')
        self.assertEqual(morph.calls, 0)
        self.assertIn(('<s>', 'ቤት'), lm.calls)

    def test_normalization_keeps_numeric_colon_and_commas(self):
        self.assertEqual(EthiopicPunctuationManager.normalize_sentence_text('12:30'), '12:30')
        self.assertEqual(EthiopicPunctuationManager.normalize_sentence_text('ቤት፣፣'), 'ቤት፣፣')
        self.assertEqual(EthiopicPunctuationManager.normalize_sentence_text('ቤት፡፡'), 'ቤት።')

    def test_invalid_configuration(self):
        for kwargs in ({'beam_width': 0}, {'max_candidates': 0}, {'candidate_backend': 'bad'},
                       {'max_optical_cost': -1}, {'indel_cost_cap': -1},
                       {'trusted_long_word_frequency': -1}, {'trusted_short_word_frequency': 1.5},
                       {'morphology_cache_size': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make(**kwargs)

    @unittest.skipUnless(importlib.util.find_spec('symspellpy'), 'optional symspellpy not installed')
    def test_symspell_scoring_uses_frequency_trust_and_retains_ambiguity(self):
        morph = Morph('UNKNOWN')
        lm = LM({('እና', '</s>'): 10})
        c = self.make([('አና', 4001), ('እና', 4001)], segmenter=morph, lm=lm,
                      candidate_backend='symspell')
        self.assertEqual(c.correct_sentence('አና'), 'እና')
        self.assertEqual(morph.calls, 0)
        self.assertGreaterEqual(c.morphology_stats()['trusted_bypasses'], 2)

    @unittest.skipUnless(importlib.util.find_spec('symspellpy'), 'optional symspellpy not installed')
    def test_symspell_real_word_and_optical_reranking(self):
        c = self.make([('በሽታ', 1000), ('በሸታ', 10000)], candidate_backend='symspell')
        self.assertEqual(c.trie, {})  # No second large dictionary index.
        self.assertAlmostEqual(dict(c.generate_candidates('በሸታ'))['በሽታ'], 3.3322)
        self.assertNotIn('በሽታ', dict(c.generate_candidates('በሸታ', 2)))
        self.assertEqual(c.generate_candidates('ዶ/ር'), [('ዶ/ር', 0)])

    @unittest.skipUnless(importlib.util.find_spec('symspellpy'), 'optional symspellpy not installed')
    def test_symspell_does_not_use_transposition_distance_as_optical_cost(self):
        c = self.make([('በከ', 1)], candidate_backend='symspell')
        self.assertNotIn('በከ', dict(c.generate_candidates('ከበ', 0)))
        self.assertEqual(dict(c.generate_candidates('ከበ', 20))['በከ'], c.optical_distance('በከ', 'ከበ'))


if __name__ == '__main__':
    unittest.main()
