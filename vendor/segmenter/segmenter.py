"""
AmharicMorphologicalSegmenter: Declarative, Frequency-Governed Morphological Segmenter.
Equipped with Position-Class Morphotactic Slot Grammar, Coordinated Circumfix Engine,
Authoritative 2-Character Lexical Firewall, and Proclitic De-conflation.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import heapq
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from maximal_lemma_arbitrator import MaximalLemmaArbitrator
import pos_tags
from morph_types import (
    IMPERF_STEM_SHAPES,
    PERF_STEM_SHAPES,
    NominalCohortEvidence,
    VerbalCohortEvidence,
    VerbalShape,
    SimpleLRUCache,
)
from morpheme_disambiguator import AmharicMorphemeDisambiguator, MorphemeRole
from morphology import AmharicMorphologyEngine
from seg_normalizer import AmharicNormalizer  # VENDORED: renamed to avoid colliding with tess's normalizer.py
from phonology import AmharicBoundaryPhonology
from pos_tagger import PosTagger
from project_paths import MAIN_DICTIONARY_PATH

DEFAULT_DICT_PATH = str(MAIN_DICTIONARY_PATH)

# ---------------------------------------------------------------------------
# POS tag syntax (see pos_tags.py for the canonical notation)
#   ':'  subset       VB:PERFECTIVE:RELATIVE is a kind of VB:PERFECTIVE
#   '|'  disjunction  between complete tags, with the superfamily repeated
#                     (ይጠጣ = VB:IMPERFECTIVE|VB:JUSSIVE_IMPERATIVE)
# PosTagger emits canonical tags directly; pos_tags.canonical_tag() is applied
# at this boundary as an idempotent safety net for any legacy spelling that
# still reaches the public API.
# ---------------------------------------------------------------------------
from pos_tags import canonical_tag as format_pos_tag  # noqa: E402  (API alias)


@dataclass(frozen=True)
class Candidate:
    stem: str
    prefixes: Tuple[str, ...] = ()
    suffixes: Tuple[str, ...] = ()
    rules: Tuple[str, ...] = ()
    strength: int = 0

    @property
    def prefix(self) -> str:
        return "+".join(self.prefixes)

    @property
    def suffix(self) -> str:
        return "+".join(self.suffixes)

    def key(self) -> Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
        return (self.stem, self.prefixes, self.suffixes, self.rules)


class AmharicMorphologicalSegmenter:
    STRONG = 3
    NORMAL = 2
    WEAK = 1

    def __init__(
        self,
        dictionary_path: Optional[str] = DEFAULT_DICT_PATH,
        vocab_dict: Optional[Dict[str, int]] = None,
        two_char_words_path: Optional[str] = None,
        cohort_min_freq: int = 50,
        stem_min_freq: int = 3,
        two_char_corpus_min_freq: int = 50,
        popular_stem_threshold: int = 1005,
        cn_triangulation_min: int = 3,
        mn_triangulation_min: int = 2,
        pn_triangulation_min: int = 1,
        verbal_triangulation_min: int = 2,
        proclitic_ratio_threshold: float = 20.0,
        prefer_morph_segmentation: bool = True,
        auto_normalize: bool = True,
        raw_suffix_fallback: bool = False,
        max_recursion_depth: int = 8,
        phonology: Optional[AmharicBoundaryPhonology] = None,
        morphology: Optional[AmharicMorphologyEngine] = None,
        normalizer: Optional[AmharicNormalizer] = None,
        listwise_arbitration: bool = True,
    ):
        self.phonology = phonology or AmharicBoundaryPhonology()
        self.morphology = morphology or AmharicMorphologyEngine(
            two_char_words_path=two_char_words_path, phonology=self.phonology
        )
        self.normalizer = normalizer or AmharicNormalizer()
        self.disambiguator = AmharicMorphemeDisambiguator(morphology=self.morphology)

        self.cohort_min_freq = cohort_min_freq
        self.stem_min_freq = stem_min_freq
        self.two_char_corpus_min_freq = two_char_corpus_min_freq

        self.popular_stem_threshold = popular_stem_threshold
        self.cn_triangulation_min = cn_triangulation_min
        self.mn_triangulation_min = mn_triangulation_min
        self.pn_triangulation_min = pn_triangulation_min
        self.verbal_triangulation_min = verbal_triangulation_min
        self.proclitic_ratio_threshold = proclitic_ratio_threshold
        self.prefer_morph_segmentation = prefer_morph_segmentation
        self.auto_normalize = auto_normalize
        self.raw_suffix_fallback = raw_suffix_fallback
        self.max_depth = max_recursion_depth

        self.vocab_freq: Dict[str, int] = {}
        if vocab_dict is not None:
            self.vocab_freq = {
                (self._norm(k) if auto_normalize else k): v
                for k, v in vocab_dict.items()
            }
        elif dictionary_path and os.path.exists(dictionary_path):
            with open(dictionary_path, encoding="utf-8", errors="ignore") as f:
                for n, line in enumerate(f, 1):
                    p = line.strip().split()
                    if not p or (n == 1 and p[0].isdigit()):
                        continue
                    w = self._norm(p[0]) if auto_normalize else p[0]
                    freq = int(p[-1]) if len(p) > 1 and p[-1].isdigit() else 1
                    self.vocab_freq[w] = max(self.vocab_freq.get(w, 0), freq)

        if self.auto_normalize:
            self.morphology.normalize_inventories(self._norm)
            # The disambiguator's inverted index is a snapshot of morpheme strings
            # taken at construction time (above), so it must be rebuilt against the
            # normalized inventories. Otherwise the disambiguator and this segmenter
            # disagree about the same morpheme.
            self.disambiguator.rebuild_index()

        # Decoupled instantiation: pass predicate callback and morphology engine
        self.maximal_lemma_arbitrator = MaximalLemmaArbitrator(
            vocab_freq=self.vocab_freq,
            min_freq=self.cohort_min_freq,
            phonology=self.phonology,
            verbal_noun_checker=self.morphology.is_valid_verbal_noun_stem,
            morphology=self.morphology,
            normalizer=self._norm,
        )

        # Decoupled two-layer POS tagger
        self.pos_tagger = PosTagger(
            morphology=self.morphology,
            phonology=self.phonology,
            vocab_freq=self.vocab_freq,
            cohort_min_freq=self.cohort_min_freq,
            popular_stem_threshold=self.popular_stem_threshold,
            cn_triangulation_min=self.cn_triangulation_min,
            mn_triangulation_min=self.mn_triangulation_min,
            verbal_triangulation_min=self.verbal_triangulation_min,
            normalizer=self._norm,
            arbitrator=self.maximal_lemma_arbitrator,
        )

        # Instance-level caches (prevents process-lifetime memory leaks from @lru_cache on self)
        self._effective_stem_freq_cache = SimpleLRUCache(16384)
        self._verbal_lemma_freq_cache = SimpleLRUCache(16384)

        # Listwise candidate-pool arbitration by MaximalLemmaArbitrator.
        # Set False to reproduce the legacy pointwise ranking (A/B evaluation).
        self.listwise_arbitration = listwise_arbitration
        self.last_arbitration = None  # PoolArbitration of the most recent word

    # ------------------------------------------------------------------
    # Public component aliases / dynamic live-inventory properties
    # ------------------------------------------------------------------

    @property
    def lemma_identifier(self) -> Optional[MaximalLemmaArbitrator]:
        """Backward-compatible alias for :attr:`maximal_lemma_arbitrator`."""
        return self.maximal_lemma_arbitrator

    @lemma_identifier.setter
    def lemma_identifier(self, value: Optional[MaximalLemmaArbitrator]) -> None:
        self.maximal_lemma_arbitrator = value

    @property
    def sfx_nominal(self) -> Set[str]:
        return getattr(self.morphology, "suffix_nominal", set())

    @property
    def pfx_nominal(self) -> Set[str]:
        return getattr(self.morphology, "prefix_nominal", set())

    @property
    def sfx_verb(self) -> Set[str]:
        return getattr(self.morphology, "all_verbal_suffixes", set())

    @property
    def pfx_verb(self) -> Set[str]:
        return getattr(self.morphology, "all_verbal_prefixes", set())

    @staticmethod
    def _flatten_tuple(t: Tuple[Any, ...]) -> Tuple[str, ...]:
        flat: List[str] = []
        for item in t:
            if isinstance(item, (tuple, list)):
                flat.extend(AmharicMorphologicalSegmenter._flatten_tuple(tuple(item)))
            elif item:
                flat.append(str(item))
        return tuple(flat)

    def _norm(self, text: str) -> str:
        if not text or not self.normalizer:
            return text
        if hasattr(self.normalizer, "normalize_word_strict"):
            return self.normalizer.normalize_word_strict(text)
        if hasattr(self.normalizer, "normalize_text"):
            return self.normalizer.normalize_text(text)
        return text

    def _has_cohort_evidence(
        self, forms: Tuple[str, ...], exclude_word: Optional[str] = None
    ) -> bool:
        if not self.vocab_freq:
            return False
        norm_exclude = self._norm(exclude_word) if exclude_word else None
        for f in forms:
            norm_f = self._norm(f) if self.auto_normalize else f
            if norm_exclude and norm_f == norm_exclude:
                continue
            if self.vocab_freq.get(norm_f, 0) >= self.cohort_min_freq:
                return True
        return False

    def _evaluate_nominal_cohort(self, stem: str) -> NominalCohortEvidence:
        return self.pos_tagger.evaluate_nominal_cohort(stem)

    def _score_nominal_cohort(self, stem: str) -> int:
        return self.pos_tagger.evaluate_nominal_cohort(stem).total_slots

    def _evaluate_verbal_cohort(
        self, stem: str, exclude_word: Optional[str] = None
    ) -> VerbalCohortEvidence:
        return self.pos_tagger.evaluate_verbal_cohort(stem, exclude_word=exclude_word)

    def _score_verbal_cohort(
        self, stem: str, exclude_word: Optional[str] = None
    ) -> Tuple[int, int]:
        ev = self.pos_tagger.evaluate_verbal_cohort(stem, exclude_word=exclude_word)
        return ev.core_concord_hits * 30, ev.total_slots * 10

    def _is_macro_verbal_noun_candidate(self, c: Candidate) -> bool:
        if not c.prefixes or not self.morphology.is_valid_verbal_noun_stem(c.stem):
            return False
        if any(p in self.morphology.all_subject_prefixes for p in c.prefixes):
            return False
        surface = "".join(c.prefixes) + c.stem
        return any(
            surface.startswith(macro) for macro in self.morphology.macro_prefix_vn
        )

    def _classify_nominal_stem(
        self, stem: str, candidate: Optional[Candidate] = None
    ) -> Optional[str]:
        return self.pos_tagger.subclassify_nominal_stem(stem, candidate=candidate)

    def _classify_verbal_stem(
        self, stem: str, exclude_word: Optional[str] = None
    ) -> bool:
        return self.pos_tagger.is_known_verbal_stem(stem, exclude_word=exclude_word)

    def _is_known_nominal_stem(self, stem: str) -> bool:
        return self.pos_tagger.is_known_nominal_stem(stem)

    def _is_known_verbal_stem(self, stem: str) -> bool:
        return self.pos_tagger.is_known_verbal_stem(stem)

    def _is_known_stem(self, stem: str) -> bool:
        return self.pos_tagger.is_known_stem(stem)


    def get_effective_stem_freq(self, stem: str, shape: Optional[VerbalShape] = None) -> int:
        if not stem or not self.vocab_freq:
            return 0
        cache_key = (stem, shape)
        if cache_key in self._effective_stem_freq_cache:
            return self._effective_stem_freq_cache[cache_key]

        norm_stem = self._norm(stem) if self.auto_normalize else stem
        direct_freq = self.vocab_freq.get(norm_stem, 0)

        citations = self.morphology.generate_lemma_candidates(stem, shape=shape)
        citation_freq = 0
        if citations:
            citation_freq = max(
                (
                    self.vocab_freq.get(self._norm(c) if self.auto_normalize else c, 0)
                    for c in citations
                ),
                default=0,
            )
        res = max(direct_freq, citation_freq)
        self._effective_stem_freq_cache[cache_key] = res
        return res

    def get_verbal_lemma_freq(self, stem: str, shape: Optional[VerbalShape] = None) -> int:
        if not stem or not self.vocab_freq:
            return 0
        cache_key = (stem, shape)
        if cache_key in self._verbal_lemma_freq_cache:
            return self._verbal_lemma_freq_cache[cache_key]

        citations = self.morphology.generate_lemma_candidates(stem, shape=shape)
        if not citations:
            return 0
        res = max(
            (
                self.vocab_freq.get(self._norm(c) if self.auto_normalize else c, 0)
                for c in citations
            ),
            default=0,
        )
        self._verbal_lemma_freq_cache[cache_key] = res
        return res


    def _is_corpus_licensed_2char_stem(self, norm_stem: str) -> bool:
        """Corpus attestation for biconsonantal stems the static registry misses.

        ``data/2_char_true_words.tsv`` is a *sample* of genuine 2-radical words,
        not a closed inventory.  Hollow and final-weak verbs supply stems such as
        ``መላ``, ``ቈጣ`` and ``ከሳ`` that are absent from that sample, so a
        ``ይ-``/``የ-`` peel of ``ይመላ``/``የቈጣ`` used to be rejected and the whole
        word was then misread as a bare perfective.  Requiring real corpus
        frequency keeps the firewall against invented biconsonantal bases.
        """
        if self.two_char_corpus_min_freq <= 0 or not self.vocab_freq:
            return False
        if not norm_stem or len(norm_stem) != 2:
            return False
        if self.vocab_freq.get(norm_stem, 0) < self.two_char_corpus_min_freq:
            return False
        # Attestation alone is not enough: the firewall exists to keep invented
        # biconsonantal bases out of the lattice, so the stem must also satisfy a
        # verbal template.  This admits the hollow/final-weak bases above while
        # still rejecting accidental fragments such as ``መነ`` or ``ድበ``.
        return (
            self.morphology.is_valid_imperfective_stem(norm_stem)
            or self.morphology.is_valid_jussive_stem(norm_stem)
            or self.morphology.is_valid_perfective_template(norm_stem)
        )

    def _valid_stem(self, stem: str) -> bool:
        if not stem:
            return False
        norm_s = self._norm(stem) if self.auto_normalize else stem
        # Dynamically license verified sparse verbs (including 1-character stems like 'ሻ') from morphemes.json
        if norm_s in getattr(self.morphology, "confirmed_sparse_verbs", set()):
            return True
        if stem == "ል":
            return True
        if len(stem) < 2:
            return False
        # Stems can never harbor unpeeled relative prefixes
        if stem.startswith(("የሚ", "የምት", "የምን", "የምታ")):
            return False
        # 2-character licensing: data-driven inventories plus corpus attestation
        if len(stem) == 2:
            if norm_s in self.morphology.pronoun_stems:
                return True
            if getattr(self.morphology, "is_licensed_2char_stem", None) and self.morphology.is_licensed_2char_stem(norm_s):
                return True
            if self.morphology.is_licensed_defective_root(norm_s):
                return True
            return self._is_corpus_licensed_2char_stem(norm_s)

        return True

    def _decompose(
        self, word: str, prefixes: Tuple[str, ...], suffixes: Tuple[str, ...]
    ):
        yield from self.morphology.decompose_all_morphology(
            word,
            prefixes=prefixes,
            suffixes=suffixes,
            known_stem_checker=self._is_known_stem,
            known_verbal_stem_checker=self._is_known_verbal_stem,
            include_raw_suffix_fallback=self.raw_suffix_fallback,
        )

    def _is_valid_affix_combination(
        self,
        prefixes: Tuple[str, ...],
        suffixes: Tuple[str, ...] = (),
        stem: str = "",
        rules: Tuple[str, ...] = (),
    ) -> bool:
        if len(prefixes) > 4 or len(suffixes) > 5:
            return False

        if stem:
            if not self._valid_stem(stem):
                return False
            if not self.morphology.is_valid_stem_for_affixes(stem, prefixes, suffixes):
                return False

            # Strict 2-Character Category Concord:
            # Verbal affixes require verbal columns (0, 1, 3, 6); nominal affixes require Column 20
            if len(stem) == 2:
                norm_s = self._norm(stem) if self.auto_normalize else stem
                has_v_affixes = (
                    any(p in self.morphology.all_verbal_prefixes or p in ("ተ", "አስ", "ተን", "አን", "ታ") for p in prefixes)
                    or any(s in self.morphology.all_verbal_suffixes or s in ("አል", "ዋል", "ኦ", "ቼ", "ተው", "ቶ", "ታ", "አ") for s in suffixes)
                )
                has_n_affixes = (
                    any(p in self.morphology.all_nominal_prefixes for p in prefixes)
                    or any(s in self.morphology.all_nominal_suffixes for s in suffixes)
                )
                if has_v_affixes:
                    is_v_stem = (
                        self.morphology.is_licensed_2char_verbal_stem(norm_s)
                        or self.morphology.is_licensed_defective_root(norm_s)
                        or norm_s in self.morphology.pronoun_stems
                        or self._is_corpus_licensed_2char_stem(norm_s)
                    )
                    if not is_v_stem:
                        return False
                elif has_n_affixes:
                    if not self.morphology.is_licensed_2char_nominal(norm_s):
                        return False
            # Ideophone Morphotactic Firewall: Ideophones cannot take compound prefixes like 'ባለ-' or 'ስነ-'
            if stem in getattr(self.morphology, "closed_class_ideophones", set()):
                if prefixes:
                    return False
                pragmatic_set = {"ም", "ስ", "ና", "ማ"}
                if any(s not in pragmatic_set for s in suffixes):
                    return False

        # Prefix Stacking Multiplicity Laws
        if sum(1 for p in prefixes if p in self.morphology.prefix_universal) > 1:
            return False
        if sum(1 for p in prefixes if p in self.morphology.prefix_verb_modal) > 1:
            return False
        if (
            sum(1 for p in prefixes if p in self.morphology.prefix_verb_atomic_negative)
            > 1
        ):
            return False
        if (
            sum(1 for p in prefixes if p in self.morphology.prefix_nominal_associatives)
            > 1
        ):
            return False

        # Suffix Stacking Multiplicity Laws
        if (
            sum(1 for s in suffixes if s in self.morphology.suffix_nominal_possessive)
            > 1
        ):
            return False
        if sum(1 for s in suffixes if s in self.morphology.suffix_nominal_plural) > 1:
            return False
        if (
            sum(1 for s in suffixes if s in self.morphology.suffix_applicative_clitics)
            > 1
        ):
            return False

        # Stranded Applicative Firewall: Applicatives (-ል-, -ብ-) obligatorily require an Object Clitic dependent.
        # They can NEVER appear as bare terminal suffixes at the rightmost boundary.
        for idx, s in enumerate(suffixes):
            if s in ("ል", "ብ"):
                if idx == len(suffixes) - 1:
                    return False  # Stranded at the rightmost boundary!
                next_s = suffixes[idx + 1]
                if next_s not in (
                    "ኝ",
                    "ህ",
                    "ሽ",
                    "ት",
                    "ው",
                    "አት",
                    "ን",
                    "አችሁ",
                    "አቹ",
                    "አቸው",
                    "ዎ",
                    "ዎት",
                ):
                    return False
        if (
            sum(
                1 for s in suffixes if s in self.morphology.suffix_verb_object_exclusive
            )
            > 1
        ):
            return False
        if sum(1 for s in suffixes if s in self.morphology.suffix_verb_auxiliary) > 1:
            return False
        if sum(1 for s in suffixes if s in ("ና", "ም", "ስ", "ማ")) > 1:
            return False

        # Morphotactic Partitions: Asymmetric One-Way Plural Superpower Gate
        # 1. If candidate carries nominal plural suffixes (-ዎች, -ኦች, -ቶች, -ሎች),
        # it is unconditionally certified as NOMINAL; verbal templates cannot override it.
        # Morphotactic Partitions: Asymmetric One-Way Plural Superpower Gate
        # 1. If candidate carries nominal plural suffixes (-ዎች, -ኦች, -ቶች, -ሎች),
        # it is unconditionally certified as NOMINAL; verbal templates cannot override it.
        has_nom_plural_sfx = any(s in self.morphology.suffix_nominal_exclusive_plural for s in suffixes)
        norm_s = self._norm(stem) if self.auto_normalize else stem
        
        # 2. Verbal stems are licensed ONLY if they do NOT carry exclusive nominal plurals
        licensed_shape = self.morphology.classify_licensed_verbal_shape(prefixes, suffixes, stem=stem)
        is_verb_stem = (
            not has_nom_plural_sfx
            and (
                licensed_shape in PERF_STEM_SHAPES
                or licensed_shape in IMPERF_STEM_SHAPES
                or self.morphology.is_valid_perfective_template(stem)
                or self._classify_verbal_stem(stem)
                or any(p in self.morphology.exclusive_verbal_prefixes for p in prefixes)
            )
        )


        # 3. Broad NOM Superset (Pronouns, Interrogatives, Demonstratives, Nominals, Abstracts, VNs)
        is_pure_nom = (
            has_nom_plural_sfx
            or stem in self.morphology.pronoun_stems
            or self.morphology.get_closed_class_pos(norm_s) in ("PRO", "INT", "ADV", "CONJ", "ADP", "VB:IDEO")
            or self.morphology.is_abstract_noun_stem(norm_s)
            or self.morphology.is_valid_verbal_noun_stem(norm_s)
            or stem in getattr(self.morphology, "geez_plurals", set())
            or stem in getattr(self.morphology, "confirmed_sparse_nouns", set())
            or (self._is_known_nominal_stem(stem) and not is_verb_stem)
        )

        is_nom_stem = is_pure_nom or self._is_known_nominal_stem(stem)

        # Relative prefixes cannot co-occur with main-clause tense auxiliaries (*የሚሰብራል, *የሚቻል)
        has_rel_marker = any(p in self.morphology.prefix_relative_pronoun or p in ("የሚ", "የምት", "የምን", "የምታ") for p in prefixes)
        if has_rel_marker and any(s in ("አል", "ዋል") or s in self.morphology.suffix_verb_auxiliary for s in suffixes):
            return False

        # Incomplete relative peeling firewall (e.g. የ + ሚላጭ is invalid; must peel as የሚ- or የ+እም+ይ)
        if prefixes == ("የ",) and stem.startswith(("ሚ", "ምት", "ምን", "ምታ")):
            return False

        # Preposition 'ያለ' ("without") strictly requires a nominal base, never bare finite verbs
        if "ያለ" in prefixes and not is_pure_nom:
            return False

        # 4. Morphotactic Firewall: Block verbal inflection from pure NOM layer
        if is_pure_nom and not is_verb_stem:
            # Applicatives (-ል-, -ብ-) and Auxiliaries (-አል, -ዋል) are strictly verbal
            if any(
                s in ("ል", "ብ", "አል", "ዋል")
                or s in self.morphology.suffix_applicative_clitics
                or s in self.morphology.suffix_verb_auxiliary
                for s in suffixes
            ):
                return False

            # Past subject concord (-ሁ, -ክ, -ኧች, -ነን)
            if any(s in self.morphology.suffix_verb_exclusive_subject for s in suffixes):
                return False

            # Exclusive verbal 1S object clitic -ኝ
            if any(s == "ኝ" for s in suffixes):
                return False

            # Pure verbal subordinating prefixes (ሲ-, ቢ-, ሊ-, እንድ-, እስክ-)
            # Note: 'እስከ' is excluded here because it legitimately attaches to nominals (e.g. እስከ አዲስ አበባ)
            pure_v_subord = {"ሲ", "ቢ", "ሊ", "እንድ", "እንዲ", "እስክ", "እስኪ"}
            if any(p in pure_v_subord for p in prefixes):
                return False

            # Imperfective subject concord prefixes (ይ-, ት-, እ-, እን-)
            if any(p in self.morphology.prefix_verb_iv for p in prefixes):
                return False

        # Exclusive Converb vowels (ኦ, ቼ, ቶ, ተው, ተን, ታ) cannot co-occur with subject concord
        # Note: Polysemous suffixes (ው, ህ, ሽ, ን) are legitimate object clitics on imperfectives (e.g. ይ-ቸግር-ው-አል)
        exclusive_coverb_vowels = (
            getattr(self.morphology, "suffix_verb_coverb_exclusive", set())
            | {"ኦ", "ቼ", "ተው", "ቶ", "ታ"}
        )
        has_exclusive_coverb_vowel = any(s in exclusive_coverb_vowels for s in suffixes)

        # Labiovelar stems ending in ቋ, ኳ, ጓ inherently contain the fused coverb vowel 'ኦ'
        # Labiovelar stems ending in ቋ, ኳ, ጓ only act as fused coverbs if NO subject prefixes are present.
        # When subject concord (ይ-, ት-, እ-, እን-, ያ-) is present, stems like ንኳኳ/ጓጓ/መሰኳ are authentic imperfectives (e.g. ይ-ንኳኳ-አል).
        has_subject_pfx = any(p in self.morphology.all_subject_prefixes for p in prefixes)
        if not has_exclusive_coverb_vowel and not has_subject_pfx and stem and stem[-1] in ("ቋ", "ኳ", "ጓ", "ኋ") and any(s in ("አል", "ዋል") for s in suffixes):
            has_exclusive_coverb_vowel = True

        if has_exclusive_coverb_vowel and has_subject_pfx:
            return False

        # ------------------------------------------------------------------
        # Active Morphotactic Concord Table (Semitic Verbal Super-Families)
        # ------------------------------------------------------------------
        if licensed_shape != VerbalShape.UNKNOWN:
            # Shape C (Imperfective): Requires subject prefix, forbids perfective suffixes
            if licensed_shape in IMPERF_STEM_SHAPES and licensed_shape != VerbalShape.SHAPE_D_JUSSIVE_IMPERATIVE:
                if any(s in self.morphology.suffix_verb_exclusive_subject for s in suffixes):
                    return False
                if any(r.startswith("possessive") for r in rules):
                    return False

            # Shape A (Perfective): Forbids imperfective subject prefixes and bare auxiliaries
            elif licensed_shape == VerbalShape.SHAPE_A_PERFECTIVE:
                if any(p in self.morphology.prefix_verb_iv for p in prefixes):
                    return False
                # Direct auxiliary cannot attach to Shape A without coverb vowel
                if any(s in ("አል", "ዋል") for s in suffixes) and not any(s in ("ኦ", "ቼ", "ተው") for s in suffixes):
                    return False
                # Suffix -ኡ on Shape A verbs with verbal context is strictly subject concord
                if any(r.startswith("possessive") for r in rules):
                    return False

            # Shape D (Jussive / Imperative): Forbids tense auxiliaries and relative particles
            elif licensed_shape == VerbalShape.SHAPE_D_JUSSIVE_IMPERATIVE:
                if any(s in ("አል", "ዋል") or s in self.morphology.suffix_verb_auxiliary for s in suffixes):
                    return False
                if any(p in self.morphology.prefix_relative_pronoun or p in self.morphology.rel_imperfective_prefixes for p in prefixes):
                    return False

            # Shape B (Converb): Forbids subject prefixes and relative complementizers
            elif licensed_shape == VerbalShape.SHAPE_B_COVERB:
                if any(p in self.morphology.all_subject_prefixes for p in prefixes):
                    return False
                if any(p in self.morphology.prefix_relative_pronoun for p in prefixes):
                    return False


        # Scoped Disambiguation: Resolve ambiguous affixes based on determined stem class
        resolved_suffix_roles = []
        has_verb_obj_rule_flag = any(r.startswith(("verb_object", "verb_obj")) for r in rules)
        for s in suffixes:
            if s == "ው" and has_verb_obj_rule_flag and is_verb_stem:
                role = MorphemeRole.VERB_OBJECT_CLITIC
            else:
                role = self.disambiguator.resolve_suffix(
                    stem=stem,
                    morpheme=s,
                    stem_is_nominal=is_nom_stem,
                    stem_is_verbal=is_verb_stem,
                    adjacent_suffixes=suffixes,
                )
            resolved_suffix_roles.append(role)

            # Authoritative Role Enforcement Firewall
            if rules:
                if role == MorphemeRole.VERB_SUBJECT_AGREEMENT and any(
                    r.startswith(("possessive", "definite")) for r in rules
                ):
                    return False
                if role in (
                    MorphemeRole.POSSESSIVE_PRONOUN,
                    MorphemeRole.DEFINITE_ARTICLE,
                ) and any(
                    r.startswith(("perfective_subject", "verb_plural_u", "imp_subject"))
                    for r in rules
                ):
                    return False

        # Exclusive Partition Firewalls (strictly governed by morphemes.json exclusive sets)
        has_exclusive_v_pfx = any(
            p in self.morphology.exclusive_verbal_prefixes for p in prefixes
        )
        has_exclusive_n_pfx = any(
            p in self.morphology.exclusive_nominal_prefixes for p in prefixes
        )
        has_exclusive_v_sfx = any(
            s in self.morphology.exclusive_verbal_suffixes for s in suffixes
        )
        has_exclusive_n_sfx = any(
            s in self.morphology.exclusive_nominal_suffixes for s in suffixes
        )
        has_exclusive_v_subj = any(
            s in self.morphology.suffix_verb_exclusive_subject for s in suffixes
        )
        has_exclusive_v_obj = any(
            s
            in (
                self.morphology.suffix_verb_object_exclusive
                | self.morphology.suffix_verb_applicative_exclusive
            )
            for s in suffixes
        )

        if has_exclusive_v_sfx and has_exclusive_n_sfx:
            return False
        if has_exclusive_v_pfx and has_exclusive_n_sfx:
            return False
        if has_exclusive_n_pfx and (
            has_exclusive_v_subj or has_exclusive_v_obj or has_exclusive_v_sfx
        ):
            return False
        if has_exclusive_n_pfx and has_exclusive_v_pfx:
            return False

        # Associative prefixes (የነ-, እነ-, ከነ-, በነ-) attach exclusively to nominals/names, never finite verbs or object clitics
        has_assoc_pfx = any(p in self.morphology.prefix_nominal_associatives for p in prefixes)
        if has_assoc_pfx and (
            is_verb_stem
            or has_exclusive_v_sfx
            or has_exclusive_v_subj
            or any(s in ("ው", "ት", "ኝ", "አት", "አቸው", "አችሁ", "አቹ", "ህ", "ሽ", "ን", "ነን") for s in suffixes)
        ):
            return False

        if has_exclusive_n_sfx and stem:
            if not self._is_known_nominal_stem(stem):
                return False

        for i, p in enumerate(prefixes):
            if p in self.morphology.prefix_universal and i > 0:
                return False

        if any(p in self.morphology.prefix_universal for p in prefixes) and any(
            s in self.morphology.suffix_verb_exclusive_subject for s in suffixes
        ):
            if not any(
                p in self.morphology.prefix_relative_pronoun or p.endswith("ተ")
                for p in prefixes
            ):
                return False

        # Preposition-Verb Clitic Morphotactic Firewall:
        # Prepositions (በ-, ለ-, ከ-) CANNOT attach to bare finite verbs or imperatives with object clitics
        # unless licensed by a relative pronoun (የ-) or passive voice (ተ-).
        if any(p in self.morphology.prefix_universal for p in prefixes):
            if any(
                r.startswith("imperative") for r in rules
            ) and not self.morphology.is_valid_verbal_noun_stem(stem):
                return False

        # Morphotactic Firewall: Imperfective subject prefixes (ይ, ት, እ, እን) cannot attach
        # to an intact 3MS perfective stem when NO auxiliary is present (e.g. *ይ+አደረገ is ungrammatical; must be relative የ+አደረገ -> ያደረገ).
        # When an auxiliary is present (ይ-ችል-አል, ይ-ሻል-አል), it is an authentic compound imperfective.
        has_aux_suffix = any(s in ("አል", "ዋል") or s in self.morphology.suffix_verb_auxiliary for s in suffixes)
        if not has_aux_suffix and any(p in self.morphology.prefix_verb_iv for p in prefixes):
            _, stem_last_ord = self.phonology.get_char_info(stem[-1]) if stem else ("", -1)
            if stem_last_ord in (0, 3):
                if self.morphology.is_valid_perfective_template(stem) and not self.morphology.is_valid_imperfective_stem(stem):
                    return False

        # Morphotactic Ordering: Subject concord must strictly precede Object clitics on verbs
        if not self._validate_position_class_slots(
            prefixes, suffixes, resolved_suffix_roles
        ):
            return False

        if "እን" in prefixes and "ት" in suffixes:
            return False

        # Verbal Noun Negation Law: 'አለ-' strictly requires a verbal noun stem
        if "አለ" in prefixes and not self.morphology.is_valid_verbal_noun_stem(stem):
            return False
        
        return True

    def _get_prefix_slot(self, p: str) -> float:
        # Slot -3.0: Outer Adpositional Proclitics (የ-, በ-, ለ-, ከ-, ስለ-, እንደ-, ወደ-, እስከ-)
        if p in self.morphology.prefix_universal or p in self.morphology.prefix_nominal:
            return -3.0
        # Slot -2.0: Subordinating Conjunctions, Negation, and Relative Particles (እም-, ስ-, ብ-, ሊ-, እንድ-, እስክ-)
        if (
            p in self.morphology.prefix_relative_pronoun
            or p in self.morphology.prefix_verb_exclusive_subordinate
            or p in self.morphology.prefix_verb_atomic_negative
            or p in self.morphology.prefix_rel_imperfective_particle
            or p in self.morphology.prefix_rel_imperfective_particle_allomorphs
            or p in ("እስክ", "እስኪ", "እንድ", "እንዲ", "ስ", "ሲ", "ብ", "ቢ", "ል", "ሊ", "ክ", "ኪ")
        ):
            return -2.0
        # Slot -1.5: Subject Agreement Concord (ይ-, ት-, እ-, እን-)
        if (
            p in self.morphology.all_subject_prefixes
            or p in self.morphology.prefix_rel_imperfective_agreement
        ):
            return -1.5
        # Slot -1.2: Causative Voice Primitive (አስ-)
        if p == "አስ":
            return -1.2
        # Slot -1.0: Non-Active / Passive Voice Primitive (ተ-, ተን-, ታ-, ተም-)
        if p in ("ተ", "ተን", "ታ", "ተም") or p in self.morphology.prefix_verb_voice:
            return -1.0
        return -0.5

    def _get_suffix_slot(self, s: str, role: MorphemeRole) -> int:
        # Slot 6: Accusative Case (-ን) and Pragmatic Enclitics (-ም, -ስ, -ና, -ማ)
        if s in ("ም", "ስ", "ና", "ማ") or role == MorphemeRole.PRAGMATIC_CLITIC:
            return 6
        if role == MorphemeRole.ACCUSATIVE_CASE:
            return 6
        # Slot 5: Tense Auxiliaries (-አል, -ዋል, -አለሁ, -አለች, -አሉ)
        # Slot 3: Applicative Prepositional Clitics (-ል- benefactive, -ብ- malfactive)
        # Checked BEFORE auxiliary so polysemous 'ል' disambiguated as applicative gets Slot 3
        if (
            role == MorphemeRole.VERB_APPLICATIVE_CLITIC
            or (s in ("ል", "ብ") and role != MorphemeRole.VERB_AUXILIARY)
            or s in self.morphology.suffix_applicative_clitics
        ):
            return 3
        # Slot 4: Object Pronominal Clitics (-ኝ, -ህ, -ሽ, -ት, -ው, -አት, -ን, -አችሁ, -አቸው)
        if (
            role == MorphemeRole.VERB_OBJECT_CLITIC
            or s in self.morphology.suffix_verb_object_exclusive
            or s in ("ኝ", "ህ", "ሽ", "ት", "ው", "አት", "አችሁ", "አቹ", "አቸው", "ዎ", "ዎት")
        ):
            return 4
        # Slot 5: Tense Auxiliaries (-አል, -ዋል, -አለሁ, -አለች, -አሉ)
        if (
            role == MorphemeRole.VERB_AUXILIARY
            or s in ("አል", "ዋል")
            or (s in self.morphology.suffix_verb_auxiliary and role != MorphemeRole.VERB_APPLICATIVE_CLITIC)
        ):
            return 5
        # Slot 2: Subject Agreement / Possessive / Definite Article (-ኩ, -ች, -ኧች, -ኡ, -ዬ, -ችን, -ዋ, -ው)
        if (
            role
            in (
                MorphemeRole.VERB_SUBJECT_AGREEMENT,
                MorphemeRole.POSSESSIVE_PRONOUN,
                MorphemeRole.DEFINITE_ARTICLE,
            )
            or s in self.morphology.suffix_verb_exclusive_subject
            or s in self.morphology.suffix_nominal_exclusive_possessive
            or s in self.morphology.suffix_nominal_exclusive_definiteness
        ):
            return 2
        # Slot 1: Derivational / Abstract / Nominal Plural (-ኛ, -ነት, -ስና, -ኦች, -ዎች)
        if (
            role
            in (
                MorphemeRole.NOMINAL_PLURAL,
                MorphemeRole.DERIVATIONAL_SUFFIX,
                MorphemeRole.ABSTRACT_SUFFIX,
            )
            or s in self.morphology.suffix_nominal_plural
            or s in self.morphology.suffix_nominal_derivational
            or s in self.morphology.suffix_nominal_abstract
        ):
            return 1
        return 2

    def _validate_position_class_slots(
        self,
        prefixes: Tuple[str, ...],
        suffixes: Tuple[str, ...],
        resolved_suffix_roles: List[MorphemeRole],
    ) -> bool:
        prev_p_slot = -99.0
        for p in prefixes:
            slot = self._get_prefix_slot(p)
            if slot <= prev_p_slot:
                return False
            prev_p_slot = slot

        prev_s_slot = 0
        for s, role in zip(suffixes, resolved_suffix_roles):
            slot = self._get_suffix_slot(s, role)
            # Case (-ን) and Pragmatic clitics (-ም, -ስ, -ና) can both legitimately occupy outer Slot 6
            if slot == 6 and prev_s_slot == 6:
                continue
            if slot <= prev_s_slot:
                return False
            prev_s_slot = slot

        return True

    def _generate(self, word: str) -> List[Candidate]:
        """
        Best-First Search (Priority Queue) over morphological decomposition states.
        Prioritizes high-strength Semitic rules and coherent stem lengths first,
        ensuring optimal candidate discovery without hitting arbitrary DFS depth traps.
        """
        found: List[Candidate] = []
        # Dedup keyed on the full derivation, not just the surface state.
        # Several rules legitimately reach the same (stem, prefixes, suffixes)
        # state (e.g. መምህሩ -> መምህር+ኡ via possessive_3MS, pure_vowel_S5_V1 and
        # verb_plural_u_S6_imperative).  Keying on the state alone made the
        # first-popped rule silently shadow the others, so a valid reading could
        # be lost whenever the winning rule happened to be rejected later.
        visited: Set[Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]] = set()

        # Priority Queue: (priority_tuple, counter, (cur, pfx_tuple, sfx_tuple, rules, strength, depth))
        # priority_tuple: (-strength, depth, -len(cur)) -> explores strong rules, shorter derivation paths first
        queue: List[
            Tuple[
                Tuple[int, int, int],
                int,
                Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], int, int],
            ]
        ] = []
        counter = 0

        initial_state = (word, (), (), (), 0, 0)
        heapq.heappush(queue, ((0, 0, -len(word)), counter, initial_state))

        MAX_EXPANSIONS = 800  # Best-first finds the target within 100-200 expansions
        expansions = 0

        while queue and expansions < MAX_EXPANSIONS:
            _, _, (cur, pfx_tuple, sfx_tuple, rules, strength, depth) = heapq.heappop(
                queue
            )

            state_key = (cur, pfx_tuple, sfx_tuple, rules)
            if state_key in visited:
                continue
            visited.add(state_key)
            expansions += 1

            if self._valid_stem(cur):
                found.append(
                    Candidate(
                        stem=cur,
                        prefixes=pfx_tuple,
                        suffixes=sfx_tuple,
                        rules=rules,
                        strength=strength,
                    )
                )

            if depth >= self.max_depth or len(cur) < 2:
                continue

            # Defective Verbal Hook
            if (
                hasattr(self.morphology, "defective_verbal_forms")
                and cur in self.morphology.defective_verbal_forms
            ):
                meta = self.morphology.defective_verbal_forms[cur]
                def_pfx = pfx_tuple + tuple(meta.get("prefixes", []))
                def_sfx = tuple(meta.get("suffixes", [])) + sfx_tuple
                new_strength = max(strength, self.STRONG)
                counter += 1
                heapq.heappush(
                    queue,
                    (
                        (-new_strength, depth + 1, -len(meta["stem"])),
                        counter,
                        (
                            meta["stem"],
                            def_pfx,
                            def_sfx,
                            rules + ("defective_verb_hook",),
                            new_strength,
                            depth + 1,
                        ),
                    ),
                )

            if self.morphology.is_reduplicative_quadriradical(cur):
                continue

            has_exclusive_verb_pfx = any(
                p in self.morphology.exclusive_verbal_prefixes for p in pfx_tuple
            )
            has_exclusive_verb_sfx = any(
                s in self.morphology.exclusive_verbal_suffixes or s == "ኢ"
                for s in sfx_tuple
            )

            for split in self._decompose(cur, pfx_tuple, sfx_tuple):
                if (
                    has_exclusive_verb_pfx or has_exclusive_verb_sfx
                ) and split.rule.startswith(("possessive", "plural")):
                    continue

                new_pfx = self._flatten_tuple(pfx_tuple + split.prefixes)
                new_sfx = self._flatten_tuple(split.suffixes + sfx_tuple)
                if len(new_pfx) <= 4 and (
                    split.stem != cur or new_pfx != pfx_tuple or new_sfx != sfx_tuple
                ):
                    new_strength = max(strength, split.strength)
                    counter += 1
                    # Best-first heuristic: higher rule strength and longer stems evaluated first
                    priority = (-new_strength, depth + 1, -len(split.stem))
                    heapq.heappush(
                        queue,
                        (
                            priority,
                            counter,
                            (
                                split.stem,
                                new_pfx,
                                new_sfx,
                                rules + (split.rule,),
                                new_strength,
                                depth + 1,
                            ),
                        ),
                    )

        unique: Dict[Tuple, Candidate] = {}
        for c in found:
            k = c.key()
            if k not in unique or c.strength > unique[k].strength:
                unique[k] = c
        return list(unique.values())

    def _score(
        self, c: Candidate, orig_word: str
    ) -> Tuple[int, int, int, int, int, int, int, int]:
        if not self._is_valid_affix_combination(
            c.prefixes, c.suffixes, c.stem, c.rules
        ):
            return (-1, 0, 0, 0, 0, 0, 0, 0)

        has_subord_pfx = any(
            p in self.morphology.prefix_verb_exclusive_subordinate and p != "የ"
            for p in c.prefixes
        )
        if has_subord_pfx and any(
            r.startswith(("possessive", "definite")) for r in c.rules
        ):
            return (-1, 0, 0, 0, 0, 0, 0, 0)

        dict_is_loaded = bool(self.vocab_freq)
        norm_orig = self._norm(orig_word) if self.auto_normalize else orig_word
        orig_freq = self.vocab_freq.get(norm_orig, 0)

        stem_is_vn = self.morphology.is_valid_infinitive_stem(c.stem)
        has_macro_vn = self._is_macro_verbal_noun_candidate(c)

        # De-conflate universal proclitics (የ-, በ-, ለ-, ከ-) from exclusive verbal prefixes
        has_exclusive_verb_pfx = any(
            p in self.morphology.exclusive_verbal_prefixes for p in c.prefixes
        )
        has_verb_pfx = any(p in self.morphology.all_verbal_prefixes for p in c.prefixes)

        pfx_str = "".join(c.prefixes)
        sfx_str = "".join(c.suffixes)
        has_macro_verb = (
            pfx_str in self.morphology.macro_prefix_pv
            or pfx_str in self.morphology.macro_prefix_iv
            or "እም" in c.prefixes
            or sfx_str in self.morphology.macro_suffix_benefactive_aux
            or sfx_str in self.morphology.macro_suffix_malfactive_aux
            or sfx_str in self.morphology.macro_suffix_coverb_aux
        )
        has_macro_nom = (
            pfx_str in self.morphology.macro_prefix_nominal
            or sfx_str in self.morphology.macro_suffix_cn
            or sfx_str in self.morphology.macro_suffix_mn
        )

        is_unsegmented = int(c.stem == orig_word and not c.prefixes and not c.suffixes)

        # Evaluate licensed verbal shape from morphotactic concord
        licensed_shape = self.morphology.classify_licensed_verbal_shape(c.prefixes, c.suffixes, stem=c.stem)

        # Only exclusive verbal prefixes strictly force the stem to be verbal
        if has_exclusive_verb_pfx and not stem_is_vn:
            if (
                not self._classify_verbal_stem(c.stem, exclude_word=orig_word)
                and not has_macro_verb
            ):
                return (-1, 0, 0, 0, 0, 0, 0, 0)
            freq = self.get_verbal_lemma_freq(c.stem, shape=licensed_shape)
        elif (
            has_verb_pfx
            and not any(p in self.morphology.prefix_universal for p in c.prefixes)
            and not stem_is_vn
        ):
            if (
                not self._classify_verbal_stem(c.stem, exclude_word=orig_word)
                and not has_macro_verb
            ):
                return (-1, 0, 0, 0, 0, 0, 0, 0)
            freq = self.get_verbal_lemma_freq(c.stem, shape=licensed_shape)
        else:
            freq = self.get_effective_stem_freq(c.stem, shape=licensed_shape)

            # =========================================================================
            # GATE 1: LEXICAL ATTESTATION FILTER (Fast O(1) Stem Filter)
            # =========================================================================
            # Any segmented candidate stem MUST exist in the dictionary (freq >= 3)
            # or belong to certified closed-class lexicons. Unattested pseudo-stems
            # (e.g. 'ይደለ') are eliminated here before invoking MaximalLemmaArbitrator.
            if dict_is_loaded and not is_unsegmented:
                norm_c_stem = self._norm(c.stem) if self.auto_normalize else c.stem
                direct_stem_freq = self.vocab_freq.get(norm_c_stem, 0)
                is_certified_closed = (
                    self.morphology.get_closed_class_pos(c.stem) is not None
                    or c.stem in self.morphology.pronoun_stems
                    or c.stem in getattr(self.morphology, "defective_verbal_forms", {})
                    or c.stem in getattr(self.morphology, "geez_plurals", set())
                    or c.stem in getattr(self.morphology, "confirmed_sparse_nouns", set())
                    or c.stem in getattr(self.morphology, "confirmed_sparse_verbs", set())
                    or c.stem in self.morphology.verb_stems
                )
                if direct_stem_freq < self.stem_min_freq and freq < self.stem_min_freq and not is_certified_closed:
                    return (-1, 0, 0, 0, 0, 0, 0, 0)
                # Full citation stems (not ending in 6th-order Sadis) must have direct corpus presence >= stem_min_freq
                _, last_o = self.phonology.get_char_info(c.stem[-1]) if c.stem else ("", -1)
                if last_o != 5 and direct_stem_freq < self.stem_min_freq and not is_certified_closed:
                    return (-1, 0, 0, 0, 0, 0, 0, 0)


        # NOTE: Candidate-vs-candidate lemma decisions (autonomous lemma shield,
        # single-letter proclitic anchors, pragmatic / object-clitic / possessive
        # sister commutation) are NOT made here one candidate at a time. They are
        # made listwise over the whole pool by MaximalLemmaArbitrator.arbitrate_pool()
        # in _rank_candidates(). _score() only rejects candidates that are
        # impossible on their own.

        # Protect autonomous auxiliary lemmas (e.g. አይደለ, አይደል, አይደሉ, አለ) from being peeled into pseudo-roots
        if c.prefixes:
            unpeeled_cand = c.prefixes[0] + c.stem
            is_aux_lemma = (
                unpeeled_cand in getattr(self.morphology, "auxiliary_verbs_standalone", set())
                or unpeeled_cand.startswith(("አይደል", "አይደለ"))
            )
            if is_aux_lemma:
                return (-1, 0, 0, 0, 0, 0, 0, 0)

        if self.maximal_lemma_arbitrator:
            # Suffix Morphotactic Gates (candidate-intrinsic)
            if c.suffixes:
                last_sfx = c.suffixes[-1]

                # =====================================================================
                # MaximalLemmaArbitrator Terminal Consonant & Coverb Slicing Shield
                # Prevents slicing root-final consonants from genuine verbs
                # (e.g. ይለምን -> *ይለም+ን, ተመልከት -> *ተመልከ+ት, ተላላክ -> *ተላላ+ክ, ተበላሽ -> *ተበላ+ሽ)
                # =====================================================================
                if last_sfx in ("ን", "ክ", "ት", "አ", "ሽ", "ኝ"):
                    # Check if host c.stem claims to be a finite verb
                    # "Finite verb host" must be a lexical/morphological claim, not a
                    # purely orthographic one: is_valid_jussive_stem() accepts any
                    # 2-char form with a Sadis coda, which would treat nouns such as
                    # ቤት as finite verbs and block the legitimate ቤት+ኦች+ን parse.
                    is_finite_verb_host = (
                        c.stem.startswith(("ይ", "ት", "እ", "እን", "ተ", "አ"))
                        or self.morphology.is_valid_perfective_template(c.stem)
                        or self._classify_verbal_stem(c.stem, exclude_word=orig_word)
                        or any(p in self.morphology.all_verbal_prefixes for p in c.prefixes)
                    )
                    if is_finite_verb_host and not any(p in ("የ", "የሚ") for p in c.prefixes):
                        # Extract the root whether prefix is subject (ይ/ት), passive (ተ), or causative (አ)
                        orig_root = orig_word[1:] if orig_word.startswith(("ይ", "ት", "ተ", "አ")) else orig_word
                        orig_orders = tuple(self.phonology.get_vocalic_orders(orig_root))
                        if (
                            orig_orders in self.morphology.BOUND_JUSSIVE_STEM_PATTERNS
                            or orig_orders in self.morphology.LESLAU_JUSSIVE_PATTERNS
                            or orig_orders in self.morphology.PASSIVE_JUSSIVE_ROOT_PATTERNS
                            or orig_orders in self.morphology.CAUSATIVE_JUSSIVE_ROOT_PATTERNS
                        ):
                            return (-1, 0, 0, 0, 0, 0, 0, 0)
                        # Host stem must prove it is an active verb by commuting with sister object clitics
                        if not self.maximal_lemma_arbitrator.verify_verbal_clitic_commutation(c.stem):
                            return (-1, 0, 0, 0, 0, 0, 0, 0)

                # Slot 3: Verbal Object Clitic Host Gate (e.g. መጋቢት vs መጋቢ + ት)
                has_verb_obj_rule = any(
                    r.startswith(("verb_object", "verb_obj")) for r in c.rules
                )
                if has_verb_obj_rule:
                    # Verbal Host Law: Host must be certified as a finite verbal base.
                    # -ን is both the object clitic and the accusative case marker, and
                    # the peel is always tagged "verb_object_clitic".  When an exclusive
                    # nominal morpheme is also present the concord resolves the form as
                    # nominal (licensed_shape == UNKNOWN), so this is an accusative
                    # parse of a noun (ቤት + ኦች + ን) and the verbal-host law must not veto it.
                    is_nominal_accusative_parse = (
                        licensed_shape == VerbalShape.UNKNOWN
                        and self._is_known_nominal_stem(c.stem)
                        and any(
                            s in self.morphology.exclusive_nominal_suffixes
                            for s in c.suffixes
                        )
                    )
                    if (
                        not is_nominal_accusative_parse
                        and not self._classify_verbal_stem(
                            c.stem, exclude_word=orig_word
                        )
                        and not self.morphology.is_valid_perfective_template(c.stem)
                    ):
                        return (-1, 0, 0, 0, 0, 0, 0, 0)

                    # Sister-clitic commutation (is the host an active verb?) is now
                    # decided listwise by MaximalLemmaArbitrator.arbitrate_pool().

        # Proclitic Ratio Guard (Fallback secondary safety)
        is_bare_proclitic = (
            bool(c.prefixes)
            and not c.suffixes
            and all(
                p in self.morphology.all_prepositional_or_nominal_prefixes
                for p in c.prefixes
            )
        )
        stem_direct_freq = self.vocab_freq.get(self._norm(c.stem), 0)
        if (
            is_bare_proclitic
            and not self._is_known_stem(c.stem)
            and orig_freq > 0
            and orig_freq > self.proclitic_ratio_threshold * max(stem_direct_freq, 1)
        ):
            return (-1, 0, 0, 0, 0, 0, 0, 0)

        is_known = self._is_known_stem(c.stem) or (
            bool(is_unsegmented) and orig_freq >= self.popular_stem_threshold
        )
        stem_is_known = int(is_known) if dict_is_loaded else 1

        has_strong_morph_rule = any(
            r.startswith(
                (
                    "matrix_cell",
                    "possessive",
                    "plural",
                    "verb_auxiliary",
                    "perfective_subject",
                    "imp_subject",
                    "verb_object",
                    "verb_obj",
                    "applicative",
                    "definite",
                    "compound_",
                    "coverb_",
                    "schwa_epenthesis",
                    "adjectivizer",
                    "abstract_",
                    "prefix_literal",
                    "negative_finiteness",
                    "case_accusative",
                    "verb_plural_u",
                    "imperative_2FS",
                    "pfx_",
                    "rel_",
                    "glide_vocal",
                )
            )
            for r in c.rules
        )

        pragmatic_set = getattr(
            self.morphology,
            "suffix_pragmatic_clitics",
            getattr(self.morphology, "suffix_universal_clitics", set()),
        )
        is_bare_pragmatic = (
            not c.prefixes
            and bool(c.suffixes)
            and all(s in pragmatic_set for s in c.suffixes)
            and all(r.startswith("clitic_pragmatic") for r in c.rules)
        )

        starts_with_prep = any(
            orig_word.startswith(p) for p in self.morphology.prefix_universal
        )
        norm_stem = self._norm(c.stem) if self.auto_normalize else c.stem
        is_unsegmented_preposition_initial_jussive = (
            bool(is_unsegmented)
            and starts_with_prep
            and self.pos_tagger._evaluate_verbal_double_lock(norm_stem)
            == "JUSSIVE_IMPERATIVE"
            and not self._is_known_nominal_stem(norm_stem)
        )

        if len(c.stem) == 2:
            is_popular_2char = (freq >= self.popular_stem_threshold) or (
                orig_freq > 0 and freq > orig_freq
            )
            is_verified_2char = (
                norm_stem in self.morphology.valid_2char_stems
                or self.morphology.is_licensed_defective_root(norm_stem)
                or norm_stem in self.morphology.pronoun_stems
                or self.morphology.is_valid_perfective_template(norm_stem)
                or self.morphology.is_valid_imperfective_stem(norm_stem)
            )
            has_stem_authority = is_popular_2char or is_verified_2char
        else:
            has_stem_authority = bool(stem_is_known)

        # Citation bonus applies ONLY to clean base roots (no unstripped prepositions like የ-, በ-, ባለ-)
        # Citation bonus applies ONLY to clean base roots (no unstripped prepositions like የ-, በ-, ባለ-)
        has_unpeeled_coalescence = False
        if not c.prefixes and len(c.stem) >= 2:
            for inv_stem, inv_pfx, _ in self.phonology.invert_prefix_boundary(c.stem):
                if self._is_known_stem(inv_stem):
                    has_unpeeled_coalescence = True
                    break

        is_true_base = (
            (
                self.maximal_lemma_arbitrator.is_uninflected_base(c.stem)
                or self.maximal_lemma_arbitrator.is_autonomous_lemma(c.stem)
            )
            if self.maximal_lemma_arbitrator
            else False
        )
        is_unstripped_prep_stem = (
            any(c.stem.startswith(p) for p in self.morphology.prefix_universal)
            and not is_true_base
        )

        has_unpeeled_proclitic = (
            not c.prefixes
            and is_unstripped_prep_stem
            and not is_unsegmented_preposition_initial_jussive
        )

        # Flag candidates retaining unstripped verbal plural -u (e.g. 'በለጸጉ' without peeling 'ኡ')
        # A 3MS perfective verb lemma never ends in 2nd order (-u)
        _, stem_last_order = (
            self.phonology.get_char_info(c.stem[-1]) if c.stem else ("", -1)
        )
        has_unpeeled_verbal_u = (
            stem_last_order == 1
            and len(c.stem) >= 3
            and c.stem
            not in getattr(self.morphology, "inherent_terminal_u_stems", set())
            and self.morphology.is_valid_perfective_template(
                c.stem[:-1] + self.phonology.shift_order(c.stem[-1], 0)
            )
        )
        # Detect unpeeled object clitic -w on past verbs (ሰበረው), contingents (ይደንቀው), and jussives (ይድነቀው)
        has_coda_shifted_w = False
        if c.stem.endswith("ው") and len(c.stem) >= 3 and not c.suffixes:
            penult_c = c.stem[-2]
            _, penult_ord = self.phonology.get_char_info(penult_c)
            if penult_ord == 0:
                restored_c6 = c.stem[:-2] + self.phonology.shift_order(penult_c, 5)
                orders_c6 = tuple(self.phonology.get_vocalic_orders(restored_c6))
                if (
                    self.morphology.is_valid_jussive_stem(restored_c6)
                    or self.morphology.is_valid_imperfective_stem(restored_c6)
                    or orders_c6 in self.morphology.LESLAU_JUSSIVE_PATTERNS
                    or orders_c6 in self.morphology.LESLAU_IMPERFECTIVE_PATTERNS
                    or orders_c6 in ((5, 0, 5), (5, 5, 5), (0, 5, 5))
                ):
                    has_coda_shifted_w = True

        has_unpeeled_object_w = (
            has_coda_shifted_w
            or (
                c.stem.endswith("ው")
                and len(c.stem) >= 3
                and (
                    self.morphology.is_valid_3ms_perfective_form(c.stem[:-1])
                    or c.stem[:-1] in ("ነበረ", "ነበር")
                )
            )
        )
        has_unpeeled_demonstrative_prep = (
            c.stem.startswith(("በ", "ለ", "ከ", "የ"))
            and c.stem[1:] in self.morphology.pronoun_stems
        )
        has_unpeeled_rel_imp_prefix = (
            c.prefixes == ("የ",)
            and c.stem.startswith(("ሚ", "ምት", "ምን", "ምታ"))
        )
        
        # A valid segmented stem can never harbor an unpeeled tense auxiliary (e.g. -ቷል, -ሯል, -ቋል, -ሏል, -አል, -ዋል)
        # Any stem ending in these sequences where the suffix was not peeled is an intermediate/dirty stem.
        # Exempt authentic radical roots ending in radical 'ል' (hollow (3, 5) like ዛል, ጣል, ማል, ሻል and reciprocal (3, 3, 5) like ዋዋል).
        stem_orders_eval = tuple(self.phonology.get_vocalic_orders(c.stem)) if c.stem else ()
        is_authentic_radical_l = c.stem.endswith("ል") and (
            stem_orders_eval in ((3, 5), (3, 3, 5), (7, 1, 5))
            or self.morphology.is_valid_jussive_stem(c.stem)
            or c.stem in getattr(self.morphology, "verb_stems", set())
        )

        is_type_a_jussive_l = c.stem in getattr(self.morphology, "biradical_type_a_jussive_stems", set())
        has_unpeeled_auxiliary = not c.suffixes and not is_type_a_jussive_l and (
            c.stem.endswith(("አል", "ዋል", "አለሁ", "አለች", "አለን", "አላችሁ", "አሉ"))
            or (len(c.stem) >= 2 and c.stem[-1] == "ል" and self.phonology.get_char_info(c.stem[-2])[1] in (3, 7))
        )
        # Flag candidates retaining unstripped subject prefixes when an auxiliary is present (e.g. ይቻል+አል vs ይ+ቻል+አል)
        has_unpeeled_subj_prefix = (
            not c.prefixes
            and any(
                s in ("አል", "ዋል") or s in self.morphology.suffix_verb_auxiliary
                for s in c.suffixes
            )
            and len(c.stem) >= 2
            and c.stem.startswith(("ይ", "ት", "እ", "እን", "ያ"))
        )

        is_verbal_u_citation = (
            "verb_plural_u_S0_perfective" in c.rules
            and self.morphology.is_valid_perfective_template(c.stem)
            and not is_unstripped_prep_stem
            and not has_unpeeled_coalescence
        )

        # Compound Imperfective Construction (e.g. ይ-መስል-አል -> ይመስላል, ይ-ሆን-አል -> ይሆናል)
        has_aux_sfx = any(s in ("አል", "ዋል") or s in self.morphology.suffix_verb_auxiliary for s in c.suffixes)
        aux_is_at_end = True
        if has_aux_sfx and c.suffixes:
            for idx, s in enumerate(c.suffixes):
                if s in ("አል", "ዋል") or s in self.morphology.suffix_verb_auxiliary:
                    trailing = c.suffixes[idx + 1:]
                    if any(t not in ("ም", "ስ", "ና", "ማ") for t in trailing):
                        aux_is_at_end = False
                        break

        stem_is_authentic_imp = (
            (len(c.stem) >= 3 and self.morphology.is_valid_imperfective_stem(c.stem))
            or (len(c.stem) == 2 and (
                c.stem in getattr(self.morphology, "TWO_CHAR_CONTINGENT_STEMS", set())
                or self.morphology.is_valid_imperfective_stem(c.stem)
                or self.morphology.is_licensed_defective_root(c.stem)
                or (any(p in ("ያ", "ና", "ታ") for p in c.prefixes) and c.stem in self.morphology.verb_stems)
            ))
        )

        # Lexical Attestation Gate: An authentic compound imperfective MUST have an attested verbal root
        # Prevents phantom roots like 'ባባ' (in ይባባሉ), 'ዋዋ' (in ይዋዋል), or 'አመል' (in ያመላልስ) from receiving +5000 priority
        stem_has_verbal_backing = (
            c.stem in self.morphology.verb_stems
            or self.get_verbal_lemma_freq(c.stem, shape=VerbalShape.SHAPE_C_IMPERFECTIVE) >= self.stem_min_freq
            or self._classify_verbal_stem(c.stem, exclude_word=orig_word)
        )

        is_compound_imperfective = (
            has_aux_sfx
            and aux_is_at_end
            and any(p in self.morphology.all_subject_prefixes for p in c.prefixes)
            and stem_is_authentic_imp
            and stem_has_verbal_backing
        )

        # Inflection Exhaustion: A parse decomposing down to a canonical singular root (ሰው from ሰዎች+ን)
        # receives structural priority over candidates stopping at intermediate plural forms.
        has_plural_root_unpacked = any(
            r.startswith("matrix_cell") or "plural" in r for r in c.rules
        ) and any(
            s in self.morphology.suffix_nominal_plural or s == "ኦች" for s in c.suffixes
        )
        has_definite_root_unpacked = any(
            "definite" in r or r.startswith("matrix_cell") or r.startswith("pure_vowel") for r in c.rules
        ) and any(s in ("ው", "ኡ", "ዋ") for s in c.suffixes)

        if is_compound_imperfective:
            # Fully-unpacked compound imperfective construction (Subject Concord + Stem + Aux)
            # strictly outranks intermediate candidates claiming accidental homograph roots (e.g. ዋል, ቃል, ባል)
            morph_priority = 4
            freq += 5000
        elif is_verbal_u_citation:
            morph_priority = 3
            freq += 1000
        # (A verified root lemma is lifted to morph_priority 3 once per word in
        #  _rank_candidates(), from PoolArbitration.root_verified.)
        elif (
            has_unpeeled_coalescence
            or has_unpeeled_subj_prefix
            or has_unpeeled_proclitic
            or has_unpeeled_verbal_u
            or has_unpeeled_object_w
            or has_unpeeled_demonstrative_prep
            or has_unpeeled_rel_imp_prefix
            or has_unpeeled_auxiliary
        ):
            # Saturated intermediate stems yield priority to fully unpacked base lemmas
            morph_priority = 1
        elif has_plural_root_unpacked or has_definite_root_unpacked:
            morph_priority = 3

        elif is_unsegmented_preposition_initial_jussive and stem_is_known:
            # A whole form with a validated jussive reading should not be demoted
            # as an unpeeled ለ-/በ-/ከ- proclitic. Preserve that reading unless the
            # nominal lexicon/cohort above identifies a competing lexical analysis.
            morph_priority = 3

        elif is_unsegmented and stem_is_known:
            b_plural_intact = self.morphology.generate_nominal_cohort_bundles(c.stem)[5]
            has_plural_intact = self._has_cohort_evidence(b_plural_intact)
            _, last_order = (
                self.phonology.get_char_info(c.stem[-1]) if c.stem else ("", -1)
            )
            is_terminal_kaeb = last_order == 1 and c.stem not in getattr(
                self.morphology, "inherent_terminal_u_stems", set()
            )
            if has_plural_intact and not starts_with_prep and not is_terminal_kaeb:
                morph_priority = 3
            elif (
                self.morphology.is_valid_perfective_template(c.stem)
                and not starts_with_prep
            ):
                morph_priority = 2
            else:
                morph_priority = 1

        elif (
            has_stem_authority
            and not is_bare_pragmatic
            and (
                has_strong_morph_rule or has_macro_verb or has_macro_nom or has_macro_vn
            )
        ):
            morph_priority = 3

        elif is_bare_pragmatic and has_stem_authority:
            b_plural_orig = self.morphology.generate_nominal_cohort_bundles(orig_word)[
                5
            ]
            morph_priority = 1 if self._has_cohort_evidence(b_plural_orig) else 2

        elif is_unsegmented:
            morph_priority = 1
        else:
            morph_priority = 0

        effective_steps = sum(
            1
            for r in c.rules
            if not r.startswith("pfx_neg_atomic")
            and not r.startswith("rel_imp")
            and not r.startswith("pfx_prep_a_coalescence")
        )
        suffix_penalty = (
            0
            if (has_strong_morph_rule or has_macro_verb or has_macro_nom)
            else len(c.suffixes)
        )

        # Position 3: Segmented parse (1) > Unsegmented fallback (0).
        # A verified root lemma or a whole-form jussive protected from a false
        # proclitic split is lifted to 2.
        if is_unsegmented_preposition_initial_jussive:
            lemma_rank = 2
        elif is_unsegmented:
            lemma_rank = 0
        else:
            lemma_rank = 1

        if dict_is_loaded:
            return (
                1,
                stem_is_known,
                morph_priority,
                lemma_rank,
                freq,                        # Frequency breaks ties among competing segmented parses
                c.strength,
                -suffix_penalty,
                -effective_steps,
            )
        return (
            1,
            0,
            morph_priority,
            lemma_rank,
            c.strength,
            0,
            -suffix_penalty,
            -effective_steps,
        )

    def _rank_candidates(self, w: str) -> List[Tuple[Candidate, Tuple[int, ...]]]:
        """
        Decision pipeline for one normalized word:
          1. GENERATION  -- _generate() builds the full candidate pool
          2. FILTERING   -- _score() rejects candidates impossible on their own
          3. ARBITRATION -- MaximalLemmaArbitrator.arbitrate_pool() compares the whole pool
                            (parent vs. peeled children) and removes the losers
          4. RANKING     -- survivors ordered by the score tuple, with evidence_rank
                            inserted right after morph_priority
        PosTagger tags the ranked candidates in analyze_word().

        Ranking key: (valid, stem_is_known, morph_priority, evidence_rank,
                      lemma_rank, freq, strength, -suffix_penalty, -effective_steps)
        """
        # 1. Generation
        candidates = self._generate(w)

        # 2. Pointwise filtering
        scored: List[Tuple[Candidate, Tuple[int, ...]]] = []
        for c in candidates:
            score = self._score(c, w)
            if score[0] > 0:
                scored.append((c, score))

        # 3. Listwise arbitration: the whole pool is evidence, only filtered ones survive
        arbitration = None
        if scored and self.maximal_lemma_arbitrator and self.listwise_arbitration:
            arbitration = self.maximal_lemma_arbitrator.arbitrate_pool(
                candidates, eligible=[c for c, _ in scored]
            )
            keep = {id(c) for c in arbitration.survivors}
            scored = [(c, s) for c, s in scored if id(c) in keep]
        self.last_arbitration = arbitration

        if not scored:
            freq = self.vocab_freq.get(w, 0)
            fallback = Candidate(stem=w, rules=("fallback",))
            return [(fallback, (1, int(freq > 0), 3, 0, 0, freq, 1, 0, 0))]

        # 4. Ranking
        ev_rank = arbitration.evidence_rank if arbitration else {}
        root_verified = bool(arbitration and arbitration.root_verified)
        derivational_voice_prefixes = (
            set(self.morphology.non_active_prefixes)
            | set(self.morphology.direct_causative_prefixes)
            | set(self.morphology.indirect_causative_prefixes)
        )
        has_licensed_derivational_voice_parse = root_verified and any(
            c.stem != w
            and any(p in derivational_voice_prefixes for p in c.prefixes)
            and self.morphology.classify_licensed_verbal_shape(
                c.prefixes, c.suffixes, stem=c.stem
            )
            != VerbalShape.UNKNOWN
            for c, _ in scored
        )
        keyed: List[Tuple[Candidate, Tuple[int, ...]]] = []
        # A universal/nominal preposition plus a closed-class adpositional stem
        # (ወደ+ፊት, በ+ላይ, ከ+ውስጥ, ...) is a compositional phrase, not an opaque
        # lemma.  When such a parse exists the unsegmented whole word must not
        # receive the verified-lemma promotion below: the phrase parse carries
        # the closed-class stem's own category evidence, while the whole word
        # has no analysis of its own.
        has_prepositional_phrase_parse = any(
            len(c.prefixes) == 1
            and not c.suffixes
            and pos_tags.closed_subtype(self.morphology.get_closed_class_pos(c.stem) or "") == "ADP"
            and (
                c.prefixes[0] in self.morphology.prefix_universal
                or c.prefixes[0]
                in getattr(
                    self.morphology, "prefix_nominal_prepositional_exclusive", set()
                )
            )
            for c, _ in scored
        )
        for c, s in scored:
            morph_priority, lemma_rank = s[2], s[3]
            if (
                root_verified
                and not has_licensed_derivational_voice_parse
                and not has_prepositional_phrase_parse
                and c.stem == w
                and not c.prefixes
                and not c.suffixes
            ):
                # Verified root lemma: decided once for the whole pool, not per candidate.
                # A licensed passive/causative derivation is stronger segmentation
                # evidence than treating the entire form as an opaque root.
                morph_priority, lemma_rank = 3, 2
            keyed.append(
                (c, (s[0], s[1], morph_priority, ev_rank.get(id(c), 0), lemma_rank) + tuple(s[4:]))
            )
        return self._prefer_spelled_reading(
            w, sorted(keyed, key=lambda z: z[1], reverse=True)
        )

    @staticmethod
    def _surface_of(c: Candidate) -> str:
        return "".join(c.prefixes) + c.stem + "".join(c.suffixes)

    def _prefer_spelled_reading(
        self, word: str, ranked: List[Tuple[Candidate, Tuple[int, ...]]]
    ) -> List[Tuple[Candidate, Tuple[int, ...]]]:
        """Let the parse that spells the word as written outrank a respelling.

        Peeling an object clitic off a consonant-final perfective (ቸገረው ->
        ቸገረ + ው) makes the peeler also offer the repaired parse, on the theory
        that the written ረ is the epenthetic vowel of ቸገር + ው.  Both spellings
        occur, so both parses are well formed, and the choice between them is
        left to the corpus frequency of the two stems -- which measures the
        words ቸገረ and ቸገር themselves, not these readings.  A nominal homograph
        then outweighs the verb (ራብ "hunger" is commoner than ራበ) and ራበ-ው, the
        3MS perfective with a 3MS object clitic, is read as the bare jussive
        ራብ-ው.

        A repair must not displace the spelling it repairs.  When some parse
        carries the identical affixes and spells the word exactly as written,
        that is the word's own reading and it takes the rank.
        """
        if len(ranked) < 2:
            return ranked
        top = ranked[0][0]
        if self._surface_of(top) == word:
            return ranked
        for i, (c, _s) in enumerate(ranked[1:], 1):
            if (
                c.prefixes == top.prefixes
                and c.suffixes == top.suffixes
                and self._surface_of(c) == word
            ):
                return [ranked[i]] + ranked[:i] + ranked[i + 1:]
        return ranked

    def explain_arbitration(self, word: str) -> List[str]:
        """Human-readable evidence trace of the listwise arbitration for one word."""
        w = self._norm(word.strip()) if self.auto_normalize else word.strip()
        if not w:
            return []
        self._rank_candidates(w)
        arb = self.last_arbitration
        if arb is None:
            return ["(no arbitration: nothing survived filtering, or arbitration disabled)"]

        def fmt(c: Candidate) -> str:
            return f"[{c.prefix or '∅'}] {c.stem} [{c.suffix or '∅'}]"

        lines = [
            f"root_verified={arb.root_verified} | survivors={len(arb.survivors)} "
            f"| eliminated={len(arb.eliminated)} | edges={len(arb.verdicts)}"
        ]
        for p, c, v in arb.verdicts:
            lines.append(
                f"{fmt(p)} -> {fmt(c)} | peel '{v.peeled}' ({v.family}) => {v.decision}: "
                f"{v.reason} | parent_ev={list(v.parent_evidence)} "
                f"child_ev={list(v.child_evidence)}"
            )
        for c in arb.survivors:
            lines.append(f"  SURVIVOR {fmt(c)} evidence_rank={arb.evidence_rank.get(id(c), 0)}")
        return lines

    def classify_pos(self, c: Candidate) -> str:
        """Alias of get_fine_grained_pos, kept for API compatibility."""
        return self.get_fine_grained_pos(c)

    def get_fine_grained_pos(self, c: Candidate) -> str:
        """Delegates fine-grained POS classification to PosTagger."""
        return format_pos_tag(
            self.pos_tagger.classify_stem(
                stem=c.stem,
                prefixes=c.prefixes,
                suffixes=c.suffixes,
                candidate=c,
            )
        )

    def analyze_word(self, word: str, limit: int = 20) -> List[Dict]:
        if limit <= 0 or not word:
            return []
        w = self._norm(word.strip()) if self.auto_normalize else word.strip()
        if not w:
            return []

        if (
            hasattr(self.morphology, "defective_verbal_forms")
            and w in self.morphology.defective_verbal_forms
        ):
            meta = self.morphology.defective_verbal_forms[w]
            return [
                {
                    "rank": 1,
                    "prefix": "+".join(meta["prefixes"]),
                    "stem": meta["stem"],
                    "suffix": "+".join(meta["suffixes"]),
                    "prefixes": list(meta["prefixes"]),
                    "suffixes": list(meta["suffixes"]),
                    "frequency": self.get_effective_stem_freq(meta["stem"]),
                    "strength": 3,
                    "rules": ("defective_verb_lexical",),
                    "score": (1, 1, 3, 0, 1, 3, 0, 0),
                    "pos": meta["pos"],
                }
            ]

        if w in self.morphology.pronoun_stems:
            return [
                {
                    "rank": 1,
                    "prefix": "",
                    "stem": w,
                    "suffix": "",
                    "prefixes": [],
                    "suffixes": [],
                    "frequency": self.get_effective_stem_freq(w),
                    "strength": 3,
                    "rules": ("pronoun_lexical",),
                    "score": (1, 1, 3, 0, 1, 3, 0, 0),
                    "pos": pos_tags.pronoun(),
                }
            ]

        closed_pos = self.morphology.get_closed_class_pos(w)
        if closed_pos is not None:
            # Special handling for 'ያለ' (Preposition "without" vs. Relative Perfective "who is / that exists")
            if w == "ያለ":
                return [{
                    "rank": 1, "prefix": "የ", "stem": "አለ", "suffix": "",
                    "prefixes": ["የ"], "suffixes": [],
                    "frequency": self.get_effective_stem_freq("አለ"),
                    "strength": 3, "rules": ("relative_existential_ye_ale",),
                    "score": (1, 1, 3, 0, 1, 3, 0, 0), "pos": "VB:PERFECTIVE:RELATIVE"
                }]

            # A short closed-class word can also be a paradigm-licensed ይ- +
            # two-character contingent (ይልቅ is the relevant collision). Do not
            # let the lexical fast path hide that segmentation. Return separate
            # analyses so the verbal and closed-class readings keep their own
            # hierarchical POS tags.
            if (
                w.startswith("ይ")
                and len(w) == 3
                and pos_tags.closed_subtype(closed_pos) != "AUX"
                and self.morphology.is_licensed_2char_contingent_stem(w[1:])
            ):
                ranked = self._rank_candidates(w)
                registered_contingent = None
                if ranked:
                    candidate, score = ranked[0]
                    if (
                        candidate.prefixes == ("ይ",)
                        and not candidate.suffixes
                        and candidate.stem == w[1:]
                    ):
                        shape = self.morphology.classify_licensed_verbal_shape(
                            candidate.prefixes, candidate.suffixes, stem=candidate.stem
                        )
                        if shape == VerbalShape.SHAPE_C_IMPERFECTIVE:
                            registered_contingent = (candidate, score, shape)

                if registered_contingent is not None:
                    candidate, score, shape = registered_contingent
                    analyses = [{
                        "rank": 1,
                        "prefix": candidate.prefix,
                        "stem": candidate.stem,
                        "suffix": candidate.suffix,
                        "prefixes": list(candidate.prefixes),
                        "suffixes": list(candidate.suffixes),
                        "frequency": self.get_effective_stem_freq(candidate.stem, shape=shape),
                        "strength": candidate.strength,
                        "rules": candidate.rules,
                        "score": score,
                        "pos": self.get_fine_grained_pos(candidate),
                    }]
                    if limit > 1:
                        analyses.append({
                            "rank": 2,
                            "prefix": "",
                            "stem": w,
                            "suffix": "",
                            "prefixes": [],
                            "suffixes": [],
                            "frequency": self.get_effective_stem_freq(w),
                            "strength": 3,
                            "rules": ("closed_class_lexical",),
                            "score": (1, 1, 3, 0, 1, 3, 0, 0),
                            "pos": closed_pos,
                        })
                    return analyses[:limit]

            v_lock = self.pos_tagger._evaluate_verbal_double_lock(w)
            if v_lock and pos_tags.closed_subtype(closed_pos) != "AUX":
                # The lexical closed-class reading is the nominal side of the
                # dual (ቀድሞ -> ADV|VB:COVERB, ለምን -> INT|VB:JUSSIVE_IMPERATIVE),
                # so the family is repeated rather than replaced by a noun.
                verbal_readings = pos_tags.disjuncts(
                    format_pos_tag(
                        f"VB:{self.pos_tagger._core_verbal_lock(v_lock)}"
                    )
                )
                pos_val = pos_tags.disjoin(closed_pos, *verbal_readings)
            else:
                pos_val = closed_pos
            return [
                {
                    "rank": 1,
                    "prefix": "",
                    "stem": w,
                    "suffix": "",
                    "prefixes": [],
                    "suffixes": [],
                    "frequency": self.get_effective_stem_freq(w),
                    "strength": 3,
                    "rules": ("closed_class_lexical",),
                    "score": (1, 1, 3, 0, 1, 3, 0, 0),
                    "pos": pos_val,
                }
            ]

        # Isolated Ge'ez Plural Fast-Path (Accommodates optional outer clitics like የ-, በ- and -ን, -ም)
        if not any(w.endswith(sfx) for sfx in ("ዎችን", "ኦችን", "ቶቹን", "ቶች", "ዎች", "ኦች")):
            pfx_geez = ""
            core_w = w
            for p in self.morphology.prefix_universal:
                if len(p) == 1 and w.startswith(p) and len(w) >= 4:
                    pfx_geez = p
                    core_w = w[len(p) :]
                    break

            outer_sfx_geez = []
            if core_w.endswith("ን") and len(core_w) >= 4:
                outer_sfx_geez.append("ን")
                core_w = core_w[:-1]
            elif core_w.endswith("ም") and len(core_w) >= 4:
                outer_sfx_geez.append("ም")
                core_w = core_w[:-1]

            geez_decomp = self.morphology.decompose_geez_plural(core_w)
            if geez_decomp is not None:
                sing_stem, pl_sfx = geez_decomp
                all_sfx = [pl_sfx] + outer_sfx_geez
                all_pfx = [pfx_geez] if pfx_geez else []
                return [
                    {
                        "rank": 1,
                        "prefix": pfx_geez,
                        "stem": sing_stem,
                        "suffix": "+".join(all_sfx),
                        "prefixes": all_pfx,
                        "suffixes": all_sfx,
                        "frequency": self.get_effective_stem_freq(sing_stem),
                        "strength": 3,
                        "rules": ("geez_plural_at_direct",),
                        "score": (1, 1, 3, 0, 1, 3, 0, 0),
                        "pos": pos_tags.noun("CN"),
                    }
                ]

        ranked = self._rank_candidates(w)[:limit]

        results = []
        for i, (c, score) in enumerate(ranked, 1):
            stem_out = c.stem
            licensed_shape = self.morphology.classify_licensed_verbal_shape(c.prefixes, c.suffixes, stem=c.stem)

            is_verbal_candidate = (
                c.stem not in self.morphology.pronoun_stems
                and (self.morphology.is_valid_perfective_template(c.stem) or self._classify_verbal_stem(c.stem))
            )
            if licensed_shape == VerbalShape.SHAPE_A_PERFECTIVE and is_verbal_candidate and stem_out:
                _, last_o = self.phonology.get_char_info(stem_out[-1])
                if last_o == 5 and len(stem_out) >= 2:
                    citations = self.morphology.generate_lemma_candidates(stem_out, shape=licensed_shape)
                    if citations:
                        stem_out = citations[0]

            results.append({
                "rank": i,
                "prefix": c.prefix,
                "stem": stem_out,
                "suffix": c.suffix,
                "prefixes": list(c.prefixes),
                "suffixes": list(c.suffixes),
                "frequency": self.get_effective_stem_freq(c.stem, shape=licensed_shape),
                "strength": c.strength,
                "rules": c.rules,
                "score": score,
                "pos": self.get_fine_grained_pos(c),
            })

        return results

    def segment_word(self, word: str) -> Dict[str, Union[str, List[str]]]:
        orig = (word or "").strip()
        if not orig:
            return {"word": "", "prefix": "", "stem": "", "suffix": "", "prefixes": [], "suffixes": [], "pos": pos_tags.noun("CN")}

        # 1-Character Command Verbs (ና = "come!", ሻ = "desire!")
        if orig in ("ና", "ሻ"):
            return {
                "word": orig,
                "prefix": "",
                "stem": orig,
                "suffix": "",
                "prefixes": [],
                "suffixes": [],
                "pos": "VB:JUSSIVE_IMPERATIVE",
            }

        if len(orig) <= 1:
            return {
                "word": orig,
                "prefix": "",
                "stem": orig,
                "suffix": "",
                "prefixes": [],
                "suffixes": [],
                "pos": pos_tags.noun("CN"),
            }
        ranked = self.analyze_word(orig, limit=1)
        if not ranked:
            # analyze_word returns [] when normalization empties the string
            # (e.g. input of only zero-width joiners), which len(orig) <= 1 misses.
            return {
                "word": orig,
                "prefix": "",
                "stem": orig,
                "suffix": "",
                "prefixes": [],
                "suffixes": [],
                "pos": pos_tags.noun("CN"),
            }
        top = ranked[0]
        return {
            "word": orig,
            "prefix": top["prefix"],
            "stem": top["stem"],
            "suffix": top["suffix"],
            "prefixes": top["prefixes"],
            "suffixes": top["suffixes"],
            "pos": top.get("pos", pos_tags.noun("CN")),
        }
