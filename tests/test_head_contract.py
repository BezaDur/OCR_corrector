"""
Portable contract tests for the AmharicOCRCorrector at HEAD.

These run WITHOUT kenlm, the external `segmenter` package, `morphemes.json`
or the production lexicon. They use:

  * a fake LM (injectable via `lm=` / `state_factory=`),
  * a fake morphological analyzer (injectable via `segmenter=`),
  * tiny on-disk lexicons,
  * the REAL checked-in confusion_matrix.json,
  * the real symspellpy (install it with `pip install -r requirements-symspell.txt`).

Their job is to pin the interface contract that 04_evaluate_full_corrector.py
and the rest of the tooling depend on, and to guard the specific regressions
called out in docs/architecture_review.md.

Run:  python -m unittest discover -s tests -p 'test_head_contract.py' -v
"""
import contextlib
import enum
import logging
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFUSION = str(ROOT / "confusion_matrix.json")


# --------------------------------------------------------------------------- #
# Test doubles                                                                 #
# --------------------------------------------------------------------------- #
class State:
    def __init__(self):
        self.last = "<s>"


class FakeLM:
    """Deterministic stand-in for kenlm.Model; records every scored word."""

    def __init__(self, scores=None):
        self.scores = scores or {}
        self.calls = []

    def BeginSentenceWrite(self, state):
        state.last = "<s>"

    def BaseScore(self, previous, word, next_state):
        self.calls.append((previous.last, word))
        next_state.last = word
        return self.scores.get((previous.last, word), 0.0)


class FakeSegmenter:
    """Counts analyzer invocations; can return enums, strings, or raise."""

    def __init__(self, pos="N", failures=0, stems=None):
        self.pos = pos
        self.failures = failures
        self.stems = stems or {}
        self.calls = 0

    def segment_word(self, word):
        self.calls += 1
        if self.failures:
            self.failures -= 1
            raise RuntimeError("transient analyzer failure")
        return {"pos": self.pos, "stem": self.stems.get(word, word)}


class POS(enum.Enum):
    UNKNOWN = 0
    NOUN = 1


# --------------------------------------------------------------------------- #
# Fixture                                                                      #
# --------------------------------------------------------------------------- #
class CorrectorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The missing-morphemes.json warning is expected here (that path is
        # tested explicitly); don't drown the runner in it.
        logging.getLogger("amharic_ocr_corrector").setLevel(logging.ERROR)

    def make_from_path(self, path, **kwargs):
        options = dict(
            kenlm_path="unused-because-lm-is-injected",
            confusion_matrix_path=CONFUSION,
            dict_vocab_path=path,
            lm=FakeLM(),
            state_factory=State,
            segmenter=None,
            verbose=False,
        )
        options.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return AmharicOCRCorrector(**options)

    def make(self, words=(), **kwargs):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        lexicon = Path(folder.name) / "words.txt"
        lexicon.write_text("\n".join(f"{w} {f}" for w, f in words), encoding="utf-8")

        options = dict(
            kenlm_path="unused-because-lm-is-injected",
            confusion_matrix_path=CONFUSION,
            dict_vocab_path=str(lexicon),
            lm=FakeLM(),
            state_factory=State,
            segmenter=None,
            verbose=False,
        )
        options.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return AmharicOCRCorrector(**options)


from amharic_ocr_corrector import AmharicOCRCorrector  # noqa: E402  (after doubles)


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
class TestDependencyInjection(CorrectorTestCase):
    def test_module_imports_without_kenlm_or_symspellpy(self):
        """The corrector module must be importable (and introspectable) even
        when the heavy native deps are missing -- otherwise no test, linter or
        diagnostic tool can touch it."""
        import amharic_ocr_corrector as mod
        self.assertIsNotNone(mod.AmharicOCRCorrector)
        self.assertIn("AmharicNormalizer" if mod.HAS_NORM else "missing",
                      "AmharicNormalizer" if mod.HAS_NORM else "missing")

    def test_fake_lm_is_used_and_no_kenlm_model_is_built(self):
        lm = FakeLM()
        c = self.make([("በሽታ", 10)], lm=lm)
        self.assertIs(c.lm, lm)
        self.assertIs(c._state_factory, State)

    def test_correct_sentence_runs_with_a_fake_lm(self):
        lm = FakeLM({("<s>", "በሽታ"): -1.0})
        c = self.make([("በሽታ", 10000), ("በሸታ", 1)], lm=lm, segmenter=None)
        out = c.correct_sentence("በሸታ")
        self.assertIsInstance(out, str)
        self.assertTrue(lm.calls, "the fake LM should have been consulted")

    def test_segmenter_none_disables_morphology_without_crashing(self):
        c = self.make([("መንግስት", 1)], segmenter=None)
        self.assertFalse(c.validate_morphology("የማይታወቅቃላ"))
        self.assertEqual(c.morphology_stats()["segmenter_calls"], 0)


class TestMorphologyContract(CorrectorTestCase):
    def test_enum_pos_unknown_is_recognised(self):
        """Regression guard for the HEAD regression described in §2 of the
        review: `str(pos) not in ("UN", "UNKNOWN")` misses enum members
        (str(POS.UNKNOWN) == "POS.UNKNOWN"), which would mark EVERY token as
        known, suppress the pairwise search and grant the anchor to garbage."""
        c = self.make(segmenter=FakeSegmenter(POS.UNKNOWN))
        self.assertFalse(c.validate_morphology("ቤት"))

    def test_enum_pos_known_is_accepted(self):
        c = self.make(segmenter=FakeSegmenter(POS.NOUN))
        self.assertTrue(c.validate_morphology("የማይታወቅ"))

    @unittest.skipUnless(os.environ.get("AMHARIC_KNOWN_POS_VALUES"),
                         "set AMHARIC_KNOWN_POS_VALUES to pin real analyzer tags")
    def test_real_analyzer_pos_values(self):
        """Documents the contract; enable once the real analyzer is importable."""
        c = self.make(segmenter=FakeSegmenter(os.environ["AMHARIC_KNOWN_POS_VALUES"]))
        self.assertTrue(c.validate_morphology("መንግስት"))

    def test_unknown_pos_string_variants(self):
        for bad in ("", "UN", "UNKNOWN", "UNK", None, "unknown", "Un"):
            with self.subTest(pos=bad):
                c = self.make(segmenter=FakeSegmenter(bad))
                self.assertFalse(c.validate_morphology("አልታወቀም"))

    def test_analysis_is_cached_not_recomputed(self):
        seg = FakeSegmenter("N")
        c = self.make(segmenter=seg)
        for _ in range(100):
            self.assertTrue(c.validate_morphology("የቤታችን"))
        self.assertEqual(seg.calls, 1, "analyzer must be called once per word")

    def test_analysis_carries_the_structured_result(self):
        seg = FakeSegmenter("N", stems={"የቤታችን": "ቤት"})
        c = self.make(segmenter=seg)
        a = c.analyze("የቤታችን")
        self.assertTrue(a["known"])
        self.assertEqual(a["stem"], "ቤት")
        self.assertEqual(a["pos"], "N")
        self.assertIsInstance(a["analysis"], dict)

    def test_in_vocab_words_skip_the_analyzer_entirely(self):
        seg = FakeSegmenter("N")
        c = self.make([("መንግስት", 5000)], segmenter=seg)
        self.assertTrue(c.validate_morphology("መንግስት"))
        self.assertEqual(seg.calls, 0, "vocabulary hits must not hit the analyzer")

    def test_transient_failure_is_recorded_and_survived(self):
        seg = FakeSegmenter("N", failures=1)
        c = self.make(segmenter=seg)
        self.assertFalse(c.validate_morphology("ቤት"))       # raises -> unknown
        self.assertTrue(c.validate_morphology("ቤት"))        # failure NOT cached -> retry
        self.assertEqual(seg.calls, 2)
        stats = c.morphology_stats()
        self.assertEqual(stats["segmenter_failures"], 1)
        self.assertGreaterEqual(stats["segmenter_seconds"], 0.0)

    def test_negative_analyses_are_cached_but_errors_are_not(self):
        """A real 'unknown' verdict is memoized; an exception is not, so a
        transient analyzer fault cannot permanently blacklist a word."""
        seg = FakeSegmenter("UNKNOWN")
        c = self.make(segmenter=seg)
        for _ in range(5):
            self.assertFalse(c.validate_morphology("የማይታወቅ"))
        self.assertEqual(seg.calls, 1, "negative analysis should be memoized")

        seg2 = FakeSegmenter("N", failures=5)
        c2 = self.make(segmenter=seg2)
        for _ in range(3):
            self.assertFalse(c2.validate_morphology("ቤት"))
        self.assertEqual(seg2.calls, 3, "errors must not be memoized")

    def test_morphology_stats_reports_counts_and_cache(self):
        seg = FakeSegmenter("N")
        c = self.make(segmenter=seg, morphology_cache_size=8)
        for w in ("ሀ", "ሁ", "ሂ", "ሄ"):  # distinct under normalization
            c.validate_morphology(w)
        stats = c.morphology_stats()
        self.assertEqual(stats["segmenter_calls"], 4)
        self.assertEqual(stats["cache"]["maxsize"], 8)
        self.assertEqual(stats["cache"]["size"], 4)
        self.assertIn("trusted_bypasses", stats)

    def test_clear_caches_resets_analyzer_memo(self):
        seg = FakeSegmenter("N")
        c = self.make(segmenter=seg)
        c.validate_morphology("ቤት")
        c.validate_morphology("ቤት")
        self.assertEqual(seg.calls, 1)
        c.clear_caches()
        c.validate_morphology("ቤት")
        self.assertEqual(seg.calls, 2)

    def test_analyzer_receives_the_normalized_form(self):
        """§3.4: the analyzer must not be handed raw orthographic variants
        (ሐ/ዐ/ፀ) it may not know."""
        seen = []

        class Recorder:
            def segment_word(self, word):
                seen.append(word)
                return {"pos": "N", "stem": word}

        c = self.make(segmenter=Recorder())
        c.validate_morphology("ሐገር")          # ሐ normalizes to ሀ
        self.assertEqual(seen, ["ሀገር"])


class TestDecoderMorphologyCall(CorrectorTestCase):
    """§7.2 -- the decoder must not invoke the analyzer per candidate."""

    def _candidate_pairs(self, c, tokens):
        pairs = []
        for tok in tokens:
            for cand, _cost in c.generate_candidates(tok):
                pairs.append((tok, cand))
        return pairs

    def test_morphology_score_matches_legacy_expression_exactly(self):
        """The fast path must be semantically identical to the code it replaced:
        `lambda_morph if is_known_word(cand) else 0.0`."""
        words = [("በሽታ", 900), ("በሸታ", 3), ("ከተማ", 800), ("ክተማ", 2),
                 ("እንደሚያደርጉ", 50), ("አንደሚያደርጉ", 1), ("መንግስት", 400)]
        seg = FakeSegmenter("N")
        c = self.make([(w, f) for w, f in words], segmenter=seg, lambda_morph=1.5)
        tokens = [w for w, _ in words]
        pairs = self._candidate_pairs(c, tokens)
        self.assertTrue(pairs, "fixture should produce candidates")
        for tok, cand in pairs:
            with self.subTest(tok=tok, cand=cand):
                legacy = c.lambda_morph if c.is_known_word(cand) else 0.0
                fast = c._morphology_score(tok, cand, c.is_known_word(tok))
                self.assertAlmostEqual(fast, legacy, places=9)

    def test_decoder_analyzer_calls_do_not_scale_with_candidate_count(self):
        """Widening the candidate cap must not multiply analyzer calls."""
        words = [("በሽታ", 900), ("በሸታ", 3), ("ከተማ", 800), ("ክተማ", 2),
                 ("ውስጥ", 700), ("ውሰጥ", 2), ("ኢትዮጵያ", 600), ("ኢትዮጵያን", 5)]
        sentence = " ".join(w for w, _ in words[:6])
        counts = {}
        for cap in (4, 24):
            seg = FakeSegmenter("N")
            c = self.make([(w, f) for w, f in words], segmenter=seg,
                          max_candidates=cap, lambda_morph=1.5)
            c.correct_sentence(sentence)
            counts[cap] = seg.calls
        self.assertLess(counts[24], 6 * 24,
                        "analyzer calls must not scale with the candidate cap")
        self.assertLessEqual(counts[24], max(counts[4], 1) + 2,
                             "widening the cap should add ~no analyzer calls")

    def test_lambda_morph_zero_skips_all_morphology_work(self):
        seg = FakeSegmenter("N")
        c = self.make([("በሽታ", 9), ("በሸታ", 1)], segmenter=seg, lambda_morph=0)
        c.correct_sentence("በሸታ በሽታ")
        self.assertEqual(seg.calls, 0)


class TestVendoredSegmenter(CorrectorTestCase):
    """The vendored suite must be importable, or morphology is dead code.

    See vendor/segmenter/VENDORING.md. These tests skip (rather than fail) when
    the vendor tree is absent, so the suite still runs on a checkout without it.
    """

    def setUp(self):
        import amharic_ocr_corrector as m
        self.mod = m
        if not m.HAS_MORPH:
            self.skipTest("segmenter suite not importable")

    def test_segmenter_is_importable(self):
        self.assertTrue(self.mod.HAS_MORPH)
        from segmenter import AmharicMorphologicalSegmenter  # noqa: F401

    def test_vendored_path_is_the_lowest_priority_fallback(self):
        """Live checkouts must win over the frozen snapshot."""
        paths = self.mod.MORPH_PATHS
        vendored = [p for p in paths if os.path.join("vendor", "segmenter") in p]
        if not vendored:
            self.skipTest("vendored copy not present")
        # The loader inserts at sys.path[0] in list order, so the LAST entry
        # has the highest priority. Vendored must come before any live path.
        self.assertEqual(vendored[0], paths[0])

    def test_vendored_morphemes_json_is_discovered(self):
        c = self.make([("በሽታ", 1)])
        self.assertTrue(os.path.exists(c.morphemes_json_path),
                        c.morphemes_json_path)
        self.assertTrue(c.protected_suffixes)
        self.assertTrue(c.protected_prefixes)

    def test_segment_word_returns_a_structured_analysis(self):
        seg = self.mod.AmharicMorphologicalSegmenter()
        r = seg.segment_word("ለአለማቀፉ")
        self.assertIsInstance(r, dict)
        for key in ("word", "prefix", "stem", "suffix", "pos"):
            self.assertIn(key, r)

    def test_explicit_morphemes_path_is_honoured_even_when_missing(self):
        """The vendored fallback must not mask an explicitly requested path."""
        c = self.make([("በሽታ", 1)],
                      morphemes_json_path="/nonexistent/morphemes.json")
        self.assertEqual(c.morphemes_json_path, "/nonexistent/morphemes.json")
        self.assertEqual(c.protected_suffixes, set())


class TestMorphologySwitch(CorrectorTestCase):
    """use_morphology makes an accidental import result an explicit switch."""

    def test_explicit_off_disables_the_analyzer(self):
        c = self.make([("በሽታ", 1)], segmenter=Ellipsis, use_morphology=False)
        self.assertIsNone(c.segmenter)

    def test_explicit_on_requires_an_importable_analyzer(self):
        import amharic_ocr_corrector as m
        if not m.HAS_MORPH:
            with self.assertRaises(RuntimeError):
                self.make([("በሽታ", 1)], segmenter=Ellipsis, use_morphology=True)
            return
        c = self.make([("በሽታ", 1)], segmenter=Ellipsis, use_morphology=True)
        self.assertIsNotNone(c.segmenter)

    def test_default_preserves_historical_behaviour(self):
        """use_morphology=None must not shift any existing setup."""
        import amharic_ocr_corrector as m
        c = self.make([("በሽታ", 1)], segmenter=Ellipsis)
        self.assertEqual(c.segmenter is not None, m.HAS_MORPH)

    def test_disabled_morphology_never_accepts_oov(self):
        """Without the analyzer, unknown really means unknown."""
        c = self.make([("በሽታ", 1)], segmenter=Ellipsis, use_morphology=False)
        a = c.analyze("የማይታወቅበት")
        self.assertFalse(a["known"])
        self.assertEqual(a["error"], "no_segmenter")

    def test_injected_segmenter_is_not_overridden_by_the_switch(self):
        seg = FakeSegmenter("N")
        c = self.make([("በሽታ", 1)], segmenter=seg, use_morphology=False)
        self.assertIs(c.segmenter, seg)


class TestLexiconCap(CorrectorTestCase):
    """SymSpell costs ~1.03 KB/type; a multi-million-type lexicon OOMs."""

    WORDS = [("በሽታ", 900), ("ከተማ", 800), ("ውስጥ", 700), ("ሀገር", 600),
             ("ሕዝብ", 500), ("መንግስት", 400), ("ትምህርት", 300), ("ገበያ", 200),
             ("ወረቀት", 90), ("መጽሐፍ", 40), ("አደባባይ", 9), ("ቁምጣ", 3),
             ("ዘንጋዳ", 2), ("ኵስታሪ", 1)]

    def test_no_cap_keeps_everything(self):
        c = self.make(self.WORDS)
        self.assertEqual(len(c.vocab), len(self.WORDS))

    def test_max_lexicon_types_keeps_the_most_frequent(self):
        c = self.make(self.WORDS, max_lexicon_types=5)
        self.assertEqual(len(c.vocab), 5)
        self.assertIn("በሽታ", c.vocab)
        self.assertNotIn("ዘንጋዳ", c.vocab)
        # surface_form is pruned in step with vocab, or lookups go stale
        self.assertEqual(set(c.surface_form), set(c.vocab))

    def test_dict_min_freq_drops_rare_types(self):
        c = self.make(self.WORDS, dict_min_freq=100)
        self.assertEqual(len(c.vocab), 8)  # the 8 words with freq >= 100
        self.assertNotIn("መጽሐፍ", c.vocab)
        self.assertEqual(set(c.surface_form), set(c.vocab))

    def test_caps_actually_shrink_symspell(self):
        """The cap must bound the expensive structure, not just the dicts."""
        big = [(f"word{i:06d}", i) for i in range(1, 4001)]
        uncapped = self.make(big)
        capped = self.make(big, max_lexicon_types=500)
        self.assertEqual(len(uncapped.vocab), 4000)
        self.assertEqual(len(capped.vocab), 500)

    def test_capped_corrector_still_corrections(self):
        """Cap must not break the pipeline: the surviving word is still
        reachable as a candidate and the decoder still runs."""
        c = self.make(self.WORDS, max_lexicon_types=6)
        self.assertIn("በሽታ", c.vocab)          # survived the cap
        cands = {w for w, _cost in c.generate_candidates("በሸታ")}
        self.assertIn("በሽታ", cands)             # still offered by SymSpell
        self.assertIsInstance(c.correct_sentence("በሸታ"), str)

    def test_affix_memo_is_bounded(self):
        """An unbounded memo grows for the life of a long-running process."""
        c = self.make([("በሽታ", 1)])
        self.assertLessEqual(c._affix_seq_memo.maxsize, 1 << 20)
        for i in range(5000):
            for aff in ("ም", "ን", "ስ", "እ"):
                c._affix_seq_memo[(f"w{i}{aff}", id(c._suffixes_sorted))] = True
        self.assertLessEqual(len(c._affix_seq_memo._data), c._affix_seq_memo.maxsize)


class TestChunkedLexiconCap(CorrectorTestCase):
    """With a cap, the read is chunked so memory is bounded during it.

    The selection is exact, not approximate: if a word is in the global top-N
    then fewer than N words beat it globally, hence fewer than N beat it in
    its own chunk, hence it survives its chunk's top-N.
    """

    PAIRS = [("በሽታ", 900), ("ከተማ", 800), ("ውስጥ", 700), ("ሀገር", 600),
             ("ሕዝብ", 500), ("መንግስት", 400), ("ትምህርት", 300), ("ገበያ", 200),
             ("ወረቀት", 90), ("መጽሐፍ", 40), ("አደባባይ", 9), ("ቁምጣ", 3),
             ("ዘንጋዳ", 2), ("ኵስታሪ", 1)]

    def _write(self, pairs):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        p = os.path.join(d, "lex.tsv")
        with open(p, "w", encoding="utf-8") as f:
            for w, c in pairs:
                f.write(f"{w}\t{c}\n")
        return p

    def test_chunked_selection_matches_materialise_then_cap(self):
        """Chunked top-N must equal top-N of the fully materialised dict."""
        import random
        rnd = random.Random(3)
        pairs = []
        for i in range(600):
            pairs.append((f"w{i:05d}", rnd.randint(1, 1000)))
            if rnd.random() < 0.3:      # duplicates: exercises the merge
                pairs.append((f"w{i:05d}", rnd.randint(1, 1000)))
        rnd.shuffle(pairs)
        path = self._write(pairs)

        full = self.make_from_path(path)
        for n in (5, 17, 50, 200):
            with self.subTest(n=n):
                expected = set(sorted(full.vocab, key=full.vocab.__getitem__,
                                      reverse=True)[:n])
                got = self.make_from_path(path, max_lexicon_types=n)
                self.assertEqual(set(got.vocab), expected)
                self.assertTrue(all(got.vocab[k] == full.vocab[k]
                                    for k in got.vocab),
                                "frequencies must survive the merge")

    def test_chunked_cap_keeps_surface_form_in_step(self):
        path = self._write(self.PAIRS)
        c = self.make_from_path(path, max_lexicon_types=6)
        self.assertEqual(set(c.surface_form), set(c.vocab))
        self.assertEqual(len(c.vocab), 6)

    def test_chunked_cap_still_leaves_a_working_corrector(self):
        path = self._write(self.PAIRS)
        c = self.make_from_path(path, max_lexicon_types=14)
        cands = {w for w, _cost in c.generate_candidates("በሸታ")}
        self.assertIn("በሽታ", cands)


class TestSymSpellDistance(CorrectorTestCase):
    """--symspell-distance was accepted at HEAD but rejected by the
    constructor, so every run using it died with a TypeError. Restored."""

    def test_default_is_one(self):
        c = self.make([("በሽታ", 1)])
        self.assertEqual(c.symspell_max_edit_distance, 1)

    def test_distance_is_configurable(self):
        c = self.make([("በሽታ", 1)], symspell_max_edit_distance=2)
        self.assertEqual(c.symspell_max_edit_distance, 2)

    def test_distance_two_finds_strictly_more(self):
        """Sanity check that the knob is actually wired to SymSpell."""
        words = [("ኢትዮጵያ", 900), ("በሽታ", 800), ("መንግስት", 700)]
        d1 = self.make(words, symspell_max_edit_distance=1)
        d2 = self.make(words, symspell_max_edit_distance=2)
        probe = "ኢትዮጵያ"[::-1]  # far from anything at distance 1
        n1 = len(d1.symspell.lookup(probe, 2, max_edit_distance=1))
        n2 = len(d2.symspell.lookup(probe, 2, max_edit_distance=2))
        self.assertGreaterEqual(n2, n1)

    def test_evaluator_accepts_the_flag(self):
        """The CLI must accept it, and reject retired ones with a reason."""
        import subprocess
        import sys
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(here, "04_evaluate_full_corrector.py")
        r = subprocess.run(
            [sys.executable, script, "--symspell-distance", "1", "--help"],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--symspell-distance", r.stdout)

    def test_retired_flags_fail_with_an_explanation(self):
        import subprocess
        import sys
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(here, "04_evaluate_full_corrector.py")
        for flag, value in (("--trie-search", None),
                            ("--trusted-long-word-frequency", "5"),
                            ("--trusted-short-word-frequency", "5")):
            with self.subTest(flag=flag):
                cmd = [sys.executable, script, flag]
                if value is not None:
                    cmd.append(value)
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                self.assertNotEqual(r.returncode, 0, f"{flag} should fail")
                self.assertIn("removed option", r.stderr,
                              f"{flag} gave: {r.stderr[-200:]}")


class TestMorphemesJson(CorrectorTestCase):
    def test_missing_file_warns_and_degrades_the_peel_guard(self):
        """§3.5 / §6.2.2.

        The guard no longer goes fully inert when morphemes.json is absent:
        the pragmatic enclitics (-ም -ስ -ን -እ) and the proclitics
        (የ- በ- ለ- ...) are compiled in, mirroring MaximalLemmaArbitrator's own
        PRAGMATIC_SISTERS / PROCLITIC_PEEL_TARGETS constants, because those
        four enclitics account for the overwhelming majority of real damage.
        What is lost without the file is the file-derived layered affixes.
        """
        c = self.make([("በሽታ", 1)], morphemes_json_path="/nonexistent/morphemes.json")
        self.assertEqual(c.protected_suffixes, set())
        self.assertEqual(c.protected_prefixes, set())
        # File-derived layered affixes are unavailable -> guard falls through.
        self.assertFalse(c.is_unsupported_morpheme_peel("መጽሐፍቱ", "መጽሐፍ"))
        # Built-in inventories still guard the high-frequency damage.
        self.assertTrue(c.is_unsupported_morpheme_peel("ተከትሎም", "ተከትሎ"))

    def test_guard_blocks_every_observed_peel_false_positive(self):
        """§6.2.2: the top-5 observed over-peels, all now vetoed."""
        c = self.make([("በሽታ", 1)], morphemes_json_path="/nonexistent/morphemes.json")
        for orig, cand in [("ተከትሎም", "ተከትሎ"),
                           ("የሚታይም", "የሚታይ"),
                           ("እርምጃን", "እርምጃ"),
                           ("ለአለማቀፉ", "አለማቀፉ"),
                           ("የጥቅምት", "ጥቅምት")]:
            with self.subTest(orig=orig, cand=cand):
                self.assertTrue(c.is_unsupported_morpheme_peel(orig, cand))

    def test_guard_does_not_block_genuine_repairs(self):
        """Vetoing a peel must not veto a real repair."""
        c = self.make([("በሽታ", 1)], morphemes_json_path="/nonexistent/morphemes.json")
        for orig, cand in [("ተከትሎም", "ተከትሎጽ"),
                           ("እርምጃን", "እርምጃን"),
                           ("ገበያ", "ገብያ")]:
            with self.subTest(orig=orig, cand=cand):
                self.assertFalse(c.is_unsupported_morpheme_peel(orig, cand))

    def test_missing_file_can_be_made_fatal(self):
        """§3.5: an empty affix inventory silently disables the peel guard,
        which is indistinguishable from a modelling bug. Let callers opt in to
        failing loudly."""
        with self.assertRaises(FileNotFoundError):
            self.make([("በሽታ", 1)], morphemes_json_path="/nonexistent/morphemes.json",
                      require_morphemes_json=True)

    def test_affix_inventory_is_loaded_and_applied(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "morphemes.json"
        path.write_text('{"suffixes": ["ም", "ን"], "prefixes": ["የ", "ለ"]}',
                        encoding="utf-8")
        c = self.make([("ተከትሎ", 10)], morphemes_json_path=str(path))
        self.assertEqual(c.protected_suffixes, {"ም", "ን"})
        self.assertEqual(c.protected_prefixes, {"የ", "ለ"})
        self.assertTrue(c.is_unsupported_morpheme_peel("ተከትሎም", "ተከትሎ"))
        self.assertTrue(c.is_unsupported_morpheme_peel("የጥቅምት", "ጥቅምት"))


class TestOpticalCost(CorrectorTestCase):
    def test_boundary_uses_the_indel_tables_not_the_flat_defaults(self):
        """dp[i][0] = every observed char spurious -> ins[observed];
        dp[0][j]   = every candidate char dropped -> del[candidate].
        The original code used the flat defaults, with the two tables swapped."""
        c = self.make([("ሀ", 1)])
        # pure deletion: candidate is the empty string, observed is one char
        self.assertAlmostEqual(c.compute_optical_cost("ሀ", ""), c._ins_cost("ሀ"))
        # pure insertion
        self.assertAlmostEqual(c.compute_optical_cost("", "ሀ"), c._del_cost("ሀ"))

    def test_indel_cap_is_configurable(self):
        c = self.make([("ሀ", 1)], indel_cost_cap=2.4)
        self.assertLessEqual(c._ins_cost("ሀ"), 2.4)
        c2 = self.make([("ሀ", 1)], indel_cost_cap=None)
        self.assertGreaterEqual(c2._ins_cost("ሀ"), c._ins_cost("ሀ"))

    def test_directed_substitution_is_not_symmetric(self):
        c = self.make([("ሀ", 1)])
        forward = c.compute_optical_cost("እ", "አ")   # OCR saw አ, truth was እ
        backward = c.compute_optical_cost("አ", "እ")
        self.assertNotAlmostEqual(forward, backward, places=6)

    def test_optical_distance_alias_matches(self):
        c = self.make([("ሀ", 1)])
        self.assertAlmostEqual(c.optical_distance("በሽታ", "በሸታ"),
                               c.compute_optical_cost("በሸታ", "በሽታ"))


class TestSearchAndScoring(CorrectorTestCase):
    def test_max_candidates_caps_the_returned_set(self):
        c = self.make([("በሽታ", 100)], max_candidates=1)
        self.assertLessEqual(len(c.generate_candidates("በሸታ")), 1)

    def test_score_eos_toggles_end_of_sentence_reranking(self):
        lm = FakeLM({("<s>", "በሽታ"): -1.0})
        with_eos = self.make([("በሽታ", 5)], lm=lm, score_eos=True)
        without = self.make([("በሽታ", 5)], lm=FakeLM({("<s>", "በሽታ"): -1.0}),
                            score_eos=False)
        self.assertTrue(with_eos.score_eos)
        self.assertFalse(without.score_eos)
        # both must run without error and produce identical path shapes
        self.assertEqual(with_eos.correct_sentence("በሽታ"),
                         without.correct_sentence("በሽታ"))

    def test_original_token_is_always_a_candidate(self):
        c = self.make([("በሽታ", 100)])
        cands = {w for w, _ in c.generate_candidates("በሸታ")}
        self.assertIn("በሸታ", cands)

    def test_cache_info_aliases_exist_for_the_evaluator(self):
        c = self.make([("በሽታ", 100)])
        c.generate_candidates("በሸታ")
        c.validate_morphology("በሽታ")
        info = c._generate_candidates_cached.cache_info()
        self.assertEqual(info.maxsize, c._candidate_cache.maxsize)
        self.assertIsNotNone(c._validate_morphology_cached.cache_info())

    def test_empty_and_punctuation_only_input(self):
        c = self.make([("በሽታ", 100)])
        self.assertEqual(c.correct_sentence(""), "")
        self.assertEqual(c.correct_sentence("   "), "")
        self.assertEqual(c.correct_sentence("።"), "።")


if __name__ == "__main__":
    unittest.main(verbosity=2)


# --------------------------------------------------------------------------- #
# Optimized DP must be exactly equivalent to the straightforward recurrence     #
# --------------------------------------------------------------------------- #
def _reference_optical_cost(corrector, ocr_str, cand_str, correct_boundary=True):
    """Straightforward transcription of the recurrence, used as an oracle."""
    m, n = len(ocr_str), len(cand_str)
    sub = corrector.costs.get("sub", {})
    ins_t = corrector.costs.get("ins", {})
    del_t = corrector.costs.get("del", {})
    ds, di, dd = (corrector.default_sub, corrector.default_ins, corrector.default_del)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    if correct_boundary:
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + corrector._ins_cost(ocr_str[i - 1])
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + corrector._del_cost(cand_str[j - 1])
    else:  # the original (swapped, flat-default) boundaries
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + dd
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + di
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            co, cc = ocr_str[i - 1], cand_str[j - 1]
            s = 0.0 if co == cc else float(sub.get(cc, {}).get(co, ds))
            dp[i][j] = min(dp[i - 1][j] + corrector._ins_cost(co),
                           dp[i][j - 1] + corrector._del_cost(cc),
                           dp[i - 1][j - 1] + s)
    return dp[m][n]


class TestOptimizedDPEquivalence(CorrectorTestCase):
    """The inlined + early-abandoned DP must agree with the oracle exactly."""

    def _pairs(self):
        import random
        random.seed(11)
        alphabet = list({c for m in self.costs_alphabet()
                         for c in m}) or list("ሀሁሂሃሄህሆአኡኢኤእኦ")
        pairs = []
        for _ in range(400):
            a = "".join(random.choice(alphabet) for _ in range(random.randint(0, 8)))
            b = "".join(random.choice(alphabet) for _ in range(random.randint(0, 8)))
            pairs.append((a, b))
        pairs += [("", ""), ("ሀ", ""), ("", "ሀ"), ("በሽታ", "በሸታ"),
                  ("ከተማ", "ክተማ"), ("ውስጥ", "ውሰጥ"), ("በሽታ", "በሽታ")]
        return pairs

    @staticmethod
    def costs_alphabet():
        import json
        with open(CONFUSION, encoding="utf-8") as fh:
            d = json.load(fh)
        out = []
        out.append(list(d.get("sub", {}).keys()))
        out.append(list(d.get("del", {}).keys()))
        out.append(list(d.get("ins", {}).keys()))
        return out

    def test_matches_oracle_exactly(self):
        c = self.make([("ሀ", 1)])
        for ocr, cand in self._pairs():
            with self.subTest(ocr=ocr, cand=cand):
                self.assertAlmostEqual(
                    c.compute_optical_cost(ocr, cand),
                    _reference_optical_cost(c, ocr, cand),
                    places=12,
                )

    def test_cutoff_never_discards_a_qualifying_candidate(self):
        c = self.make([("ሀ", 1)])
        for cutoff in (1.0, 2.4, 3.6, 5.76, 12.0):
            for ocr, cand in self._pairs():
                with self.subTest(cutoff=cutoff, ocr=ocr, cand=cand):
                    got = c.compute_optical_cost(ocr, cand, cutoff=cutoff)
                    ref = _reference_optical_cost(c, ocr, cand)
                    if ref <= cutoff:
                        # soundness: must never reject a qualifying candidate
                        self.assertAlmostEqual(got, ref, places=12,
                                               msg="cutoff rejected a candidate "
                                                   "that qualifies")
                    else:
                        # pruning is optional; but whatever it returns must be
                        # either "definitely too expensive" or the exact value.
                        if got != float("inf"):
                            self.assertAlmostEqual(got, ref, places=12)

    def test_boundary_fix_changes_only_boundary_alignments(self):
        """Documents the behaviour change: pure insertion/deletion alignments
        now use the per-character ins/del tables instead of the flat defaults
        with the two tables swapped."""
        c = self.make([("ሀ", 1)])
        self.assertAlmostEqual(c.compute_optical_cost("ሀ", ""),
                               _reference_optical_cost(c, "ሀ", "", True))
        self.assertNotAlmostEqual(c.compute_optical_cost("ሀ", ""),
                                  _reference_optical_cost(c, "ሀ", "", False))


class TestEvaluatorContract(CorrectorTestCase):
    """§2 guard: 04_evaluate_full_corrector.py must be able to build HEAD.

    The committed benchmark JSON was produced by an earlier revision whose
    options no longer exist; this pins the option set that the evaluator
    currently sends so the two cannot drift apart again silently.
    """

    EVALUATOR_OPTIONS = dict(beam_width=5, max_optical_cost=4.8, lambda_optical=0.9,
                             lambda_lm=2.0, lambda_morph=1.5, max_candidates=25,
                             score_eos=True, require_morphemes_json=False)

    def test_evaluator_option_set_is_accepted(self):
        import inspect
        from amharic_ocr_corrector import AmharicOCRCorrector
        params = set(inspect.signature(AmharicOCRCorrector.__init__).parameters)
        unknown = set(self.EVALUATOR_OPTIONS) - params
        self.assertEqual(unknown, set(),
                         f"evaluator passes options HEAD does not accept: {unknown}")
        c = self.make([("በሽታ", 100)], **self.EVALUATOR_OPTIONS)
        self.assertEqual(c.max_candidates, 25)
        self.assertTrue(c.score_eos)

    def test_evaluator_module_exposes_compatible_defaults(self):
        """The evaluator's own defaults must not name removed options.

        `symspell_max_edit_distance` is deliberately NOT in this list: it was
        dead at HEAD (accepted by argparse, rejected by the constructor), but
        has since been restored as a real constructor parameter, so the
        evaluator passing it is now correct.
        """
        src = (ROOT / "04_evaluate_full_corrector.py").read_text(encoding="utf-8")
        for dead in ("candidate_backend", "trie_search",
                     "trusted_long_word_frequency", "trusted_short_word_frequency"):
            self.assertNotIn(f"{dead}=", src,
                             f"evaluator still passes removed option {dead!r}")

    def test_restored_option_is_accepted_by_the_constructor(self):
        """The restored flag must round-trip, not just parse."""
        import inspect
        from amharic_ocr_corrector import AmharicOCRCorrector
        params = set(inspect.signature(AmharicOCRCorrector.__init__).parameters)
        self.assertIn("symspell_max_edit_distance", params)

    def test_morphology_stats_serialises_for_the_json_report(self):
        import json
        c = self.make([("በሽታ", 100)], segmenter=FakeSegmenter("N"))
        c.validate_morphology("የማይታወቅ")
        payload = json.dumps(c.morphology_stats(), sort_keys=True)
        self.assertIn("segmenter_calls", payload)
