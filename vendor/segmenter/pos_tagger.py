"""
pos_tagger.py - Two-Layer Amharic Part-of-Speech Tagger.
Encapsulates:
  - Layer 1: Macro-Category Arbiter (2-Axis Parallel Evidence Matrix: E_Nominal x E_Verbal Double-Lock)
  - Layer 2: Fine POS micro-class assignment (CN, MN, PN, VN, IDEO, verbal aspect/voice)
Rescues non-plural nominals (Mass Nouns, Agent Nouns, Calendar Nouns) from
verbal root reconstruction using strict mutually exclusive grammatical footprints.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple

from morph_types import (
    Layer1Result,
    NominalCohortEvidence,
    VerbalCohortEvidence,
    VerbalShape,
)

from morphology import AmharicMorphologyEngine
import pos_tags
from phonology import AmharicBoundaryPhonology


class PosTagger:
    V_LOCK_TO_SHAPE: Dict[str, VerbalShape] = {
        "RELATIVE_IMPERFECTIVE": VerbalShape.SHAPE_C_IMPERFECTIVE,
        "RELATIVE_PERFECTIVE": VerbalShape.SHAPE_A_PERFECTIVE,
        "COMPOUND_IMPERFECT": VerbalShape.SHAPE_C_IMPERFECTIVE,
        "COMPOUND_PERFECT": VerbalShape.SHAPE_B_COVERB,
        "COVERB": VerbalShape.SHAPE_B_COVERB,
        "IMPERFECTIVE": VerbalShape.SHAPE_C_IMPERFECTIVE,
        "IMPERFECTIVE:JUSSIVE": VerbalShape.SHAPE_C_IMPERFECTIVE,
        "JUSSIVE_IMPERATIVE": VerbalShape.SHAPE_D_JUSSIVE_IMPERATIVE,
        "PERFECTIVE": VerbalShape.SHAPE_A_PERFECTIVE,
    }

    def __init__(
        self,
        morphology: AmharicMorphologyEngine,
        phonology: Optional[AmharicBoundaryPhonology] = None,
        vocab_freq: Optional[Dict[str, int]] = None,
        cohort_min_freq: int = 3,
        popular_stem_threshold: int = 1005,
        cn_triangulation_min: int = 3,
        mn_triangulation_min: int = 2,
        verbal_triangulation_min: int = 2,
        normalizer: Optional[Callable[[str], str]] = None,
        arbitrator: Optional[Any] = None,
    ):
        self.morphology = morphology
        self.phonology = phonology or AmharicBoundaryPhonology()
        self.vocab_freq: Dict[str, int] = vocab_freq or {}
        self.cohort_min_freq = cohort_min_freq
        self.popular_stem_threshold = popular_stem_threshold
        self.cn_triangulation_min = cn_triangulation_min
        self.mn_triangulation_min = mn_triangulation_min
        self.verbal_triangulation_min = verbal_triangulation_min
        self.normalizer = normalizer or (lambda s: s)
        # MaximalLemmaArbitrator, used to decide whether a substrate stem
        # still carries its inflectional prefix (see
        # ``_resolve_fused_inflectional_prefix``).
        self.arbitrator = arbitrator

    def _norm(self, text: str) -> str:
        return self.normalizer(text)

    # ------------------------------------------------------------------
    # Relative-imperfective prefix detection
    # ------------------------------------------------------------------

    #: Imperfective particles that, immediately after the relativiser ``የ``,
    #: form the exclusive relative-imperfective marker ``የሚ-``.
    RELATIVE_IMPERFECTIVE_PARTICLES = frozenset(
        {"ሚ", "ሚያ", "ምት", "ምታ", "ምን", "ምና", "ም", "የሚ", "የምት", "የምታ", "የምን", "የምና"}
    )

    def _has_relative_imperfective_prefix(self, prefixes: Tuple[str, ...]) -> bool:
        """Whether ``prefixes`` carry the exclusive ``የሚ-`` relative-imperfective marker.

        The peeler emits the marker atomically as ``("የ", "ሚ")`` (rule
        ``rel_imp_ye_mi``), but older/alternate paths can hand over the fused
        ``"የሚ"`` or an agreement allomorph.  A bare particle is only accepted
        when the relativiser ``የ`` precedes it, so the possessive/pragmatic
        ``ም-`` is never mistaken for the marker.
        """
        if not prefixes:
            return False
        seen_relative = False
        for p in prefixes:
            if not p:
                continue
            if p in ("የሚ", "የምት", "የምታ", "የምን", "የምና"):
                return True
            if p.startswith("የ"):
                seen_relative = True
                continue
            if seen_relative and p in self.RELATIVE_IMPERFECTIVE_PARTICLES:
                return True
        return False

    # ------------------------------------------------------------------
    # Cohort Evidence Diagnostics
    # ------------------------------------------------------------------

    def _corpus_contingent_jussive_reading(self, stem: str) -> Optional[str]:
        """Consult the derived contingent/jussive reading table for ``stem``.

        Returns ``"BOTH"``, ``"VB:IMPERFECTIVE"``, ``"VB:JUSSIVE_IMPERATIVE"`` or
        ``None`` when the stem's shape is not covered.  The causative ``አ-`` is
        elided in writing after the ``ያ-``/``ይ-`` coalescence but the segmenter
        restores it, so both spellings of the same underlying stem are probed.
        """
        if not stem:
            return None
        reading = self.morphology.contingent_jussive_reading(stem)
        if reading is None and stem.startswith("አ") and len(stem) > 2:
            reading = self.morphology.contingent_jussive_reading(stem[1:])
        if reading is None and not stem.startswith("አ"):
            reading = self.morphology.contingent_jussive_reading("አ" + stem)
        return reading

    def _has_cohort_evidence(
        self, forms: Tuple[str, ...], exclude_word: Optional[str] = None
    ) -> bool:
        if not self.vocab_freq:
            return False
        norm_exclude = self._norm(exclude_word) if exclude_word else None
        for f in forms:
            norm_f = self._norm(f)
            if norm_exclude and norm_f == norm_exclude:
                continue
            if self.vocab_freq.get(norm_f, 0) >= self.cohort_min_freq:
                return True
        return False

    def evaluate_nominal_cohort(self, stem: str) -> NominalCohortEvidence:
        b_compound, b_prep, b_assoc, b_def_acc, b_poss, b_plural, b_net = (
            self.morphology.generate_nominal_cohort_bundles(stem)
        )
        max_pl_freq = max(
            (self.vocab_freq.get(self._norm(f), 0) for f in b_plural), default=0
        )
        unambiguous_assoc_forms = [
            p + stem for p in self.morphology.unambiguous_assoc_prefixes
        ]
        has_true_pn_assoc = any(
            self.vocab_freq.get(self._norm(f), 0) >= self.cohort_min_freq
            for f in unambiguous_assoc_forms
        )
        return NominalCohortEvidence(
            compound=self._has_cohort_evidence(b_compound),
            preposition=self._has_cohort_evidence(b_prep),
            associative=self._has_cohort_evidence(b_assoc),
            def_acc=self._has_cohort_evidence(b_def_acc),
            possessive=self._has_cohort_evidence(b_poss),
            plural=self._has_cohort_evidence(b_plural),
            abstract=self._has_cohort_evidence(b_net),
            robust_plural=max_pl_freq >= 100,
            true_pn_assoc=has_true_pn_assoc,
        )

    def evaluate_verbal_cohort(
        self, stem: str, exclude_word: Optional[str] = None
    ) -> VerbalCohortEvidence:
        bundles = self.morphology.generate_verbal_cohort_bundles(stem)
        b_subj, b_obj, b_aux, b_neg, b_voice_asp, b_conv, b_appl = bundles
        is_direct = self.morphology.is_surface_finite_verb(stem) or (
            stem in self.morphology.verb_stems
        )
        return VerbalCohortEvidence(
            exclusive_subject=self._has_cohort_evidence(b_subj, exclude_word=exclude_word),
            exclusive_object=self._has_cohort_evidence(b_obj, exclude_word=exclude_word),
            exclusive_auxiliary=self._has_cohort_evidence(b_aux, exclude_word=exclude_word),
            negative=self._has_cohort_evidence(b_neg, exclude_word=exclude_word),
            voice_aspect=self._has_cohort_evidence(b_voice_asp, exclude_word=exclude_word),
            coverb=self._has_cohort_evidence(b_conv, exclude_word=exclude_word),
            applicative=self._has_cohort_evidence(b_appl, exclude_word=exclude_word),
            is_direct_stem=is_direct,
        )

    # ------------------------------------------------------------------
    # LAYER 1: DOUBLE-LOCK HELPERS
    # ------------------------------------------------------------------

    def _has_overt_exclusive_nominal_affix(self, norm_s: str) -> bool:
        if not norm_s or len(norm_s) < 3:
            return False

        if norm_s.startswith(("የሚ", "የምት", "የምን", "የምታ")):
            return False

        exclusive_prefixes = (
            self.morphology.prefix_nominal_exclusive_associatives
            | self.morphology.prefix_nominal_exclusive_compound
        )
        for pfx in exclusive_prefixes:
            if norm_s.startswith(pfx) and len(norm_s) > len(pfx) + 1:
                return True

        plural_tails = (
            "ዎችን", "ኦችን", "ዎች", "ኦች",
            "ዎቹን", "ዎቹም", "ዎቹስ", "ዎቹና", "ዎቹ",
        )
        for tail in plural_tails:
            if norm_s.endswith(tail) and len(norm_s) > len(tail) + 1:
                return True

        if hasattr(self.morphology, "compound_plural_possessives"):
            for comp_pl in self.morphology.compound_plural_possessives:
                if norm_s.endswith(comp_pl) and len(norm_s) > len(comp_pl) + 1:
                    return True
                if norm_s.endswith(comp_pl + "ን") and len(norm_s) > len(comp_pl) + 2:
                    return True

        abstract_sfxs = self.morphology.suffix_nominal_exclusive_abstract
        for sfx in abstract_sfxs:
            if norm_s.endswith(sfx) and len(norm_s) > len(sfx) + 1:
                return True
            for clitic in ("ን", "ም", "ስ", "ና"):
                if norm_s.endswith(sfx + clitic) and len(norm_s) > len(sfx) + len(clitic) + 1:
                    return True

        for sfx in ("አዊያን", "አዊት", "አዊ", "ኧኝነት", "ኧኞች", "ኧኛ", "ኞች"):
            if norm_s.endswith(sfx) and len(norm_s) > len(sfx) + 1:
                return True

        if norm_s.endswith("ኛ") and len(norm_s) > 2:
            base_core = norm_s
            for pfx in ("የሚ", "የ", "ተ", "አስ"):
                if norm_s.startswith(pfx) and len(norm_s) > len(pfx) + 1:
                    base_core = norm_s[len(pfx):]
                    break

            if (
                base_core in self.morphology.verbs_ending_in_gna
                or norm_s in self.morphology.verbs_ending_in_gna
                or self.morphology.is_valid_perfective_template(base_core)
            ):
                pass
            else:
                return True

        return False

    def _check_relative_imperfective_double_lock(self, norm_s: str) -> bool:
        if not norm_s or len(norm_s) < 3:
            return False

        rel_pfxs = set()
        for rel in getattr(self.morphology, "prefix_rel_imperfective_particle_allomorphs", ("ሚ", "ም", "ሚያ")):
            rel_pfxs.add("የ" + rel)
            rel_pfxs.add("የ" + rel + "ት")
            rel_pfxs.add("የ" + rel + "ን")
            rel_pfxs.add("የ" + rel + "ታ")
        for neg_pfx in getattr(self.morphology, "prefix_verb_negative_exclusive", ()):
            if neg_pfx.startswith("የማ"):
                rel_pfxs.add(neg_pfx)

        for pfx in sorted(rel_pfxs, key=len, reverse=True):
            if norm_s.startswith(pfx) and len(norm_s) >= len(pfx) + 1:
                rem = norm_s[len(pfx):]
                if (
                    self.morphology.is_valid_imperfective_stem(rem)
                    or self.morphology.is_attested_verbal_stem(rem)
                    or rem in self.morphology.verb_stems
                ):
                    return True

                if len(rem) >= 1:
                    n_ev = self.evaluate_nominal_cohort(norm_s)
                    if not (n_ev.plural or n_ev.robust_plural):
                        return True
        return False

    def _check_relative_perfective_double_lock(self, norm_s: str) -> bool:
        if not norm_s or len(norm_s) < 3:
            return False

        if norm_s.startswith("የ") and len(norm_s) > 2:
            rem = norm_s[1:]
            if rem.endswith("ው") and len(rem) >= 3:
                rem = rem[:-1]

            if rem.startswith(("ተን", "ተ", "ታ")):
                return True

            # 1b. Verified Lexical & Defective Roots
            if rem in self.morphology.verb_stems or self.morphology.is_licensed_defective_root(rem):
                return True

            rem_orders = self.phonology.get_vocalic_orders(rem)
            if rem_orders and rem_orders[-1] in (0, 3, 1, 7):
                is_perf_shape = (
                    tuple(rem_orders) in self.morphology.LESLAU_PERFECTIVE_PATTERNS
                    or tuple(rem_orders) in {
                        (0, 0, 3),       # Final-weak: ዘረጋ, ፈነዳ, ከረፋ, ከነዳ, ዘነጋ, ደነባ, ደነፋ, ገረጣ, ገነባ, ጠረቃ, ጠነባ
                        (3, 3),          # Stative biradical: ሟሟ, ላላ, ሣሣ
                        (7, 7),          # Dikala stative biradical: ሟሟ, ጓጓ
                        (4, 0, 0),       # Class B stative: ሴሰነ
                        (0, 3, 5, 0),    # Frequentative Type B: ቀባዥረ
                        (3, 6, 0),       # Reduplicated hollow labiovelar: ጓጎለ, ጓጎጠ
                    }
                    or self.morphology.is_valid_perfective_template(rem)
                    or self.morphology.is_licensed_defective_root(rem)
                    or rem in self.morphology.verb_stems
                )
                if is_perf_shape:
                    n_ev = self.evaluate_nominal_cohort(rem)
                    if not (n_ev.plural or n_ev.robust_plural):
                        return True

        if norm_s.startswith("ያ") and len(norm_s) >= 3:
            rec_lemma = "አ" + norm_s[1:]
            orders_rec = tuple(self.phonology.get_vocalic_orders(rec_lemma))
            is_juss = (
                self.morphology.is_valid_jussive_stem(rec_lemma)
                or orders_rec in self.morphology.LESLAU_JUSSIVE_PATTERNS
                or orders_rec in getattr(self.morphology, "BOUND_JUSSIVE_STEM_PATTERNS", set())
            )
            if (
                self.morphology.is_valid_perfective_template(rec_lemma)
                and not is_juss
                and not self.morphology.is_valid_imperfective_stem(rec_lemma)
            ):
                n_ev = self.evaluate_nominal_cohort(norm_s)
                if not (n_ev.plural or n_ev.robust_plural):
                    return True

        return False

    def _check_compound_imperfect_double_lock(self, norm_s: str) -> bool:
        if not norm_s or len(norm_s) < 3 or not norm_s.endswith("ል"):
            return False

        iv_prefixes = getattr(self.morphology, "prefix_verb_iv", ("ይ", "ት", "እ", "እን"))
        matched_pfx = None
        for p in sorted(iv_prefixes, key=len, reverse=True):
            if norm_s.startswith(p):
                matched_pfx = p
                break

        if not matched_pfx:
            return False

        core = norm_s[len(matched_pfx):-1]
        if not core:
            return False

        if self.morphology.is_valid_imperfective_stem(core) or self.morphology.is_licensed_defective_root(core):
            return True

        for restored_stem, _ in self.phonology.invert_coda_shift_4th_to_6th(norm_s[:-1] + "ል", "ል"):
            if restored_stem.startswith(matched_pfx):
                inner_stem = restored_stem[len(matched_pfx):]
                if (
                    self.morphology.is_valid_imperfective_stem(inner_stem)
                    or self.morphology.is_licensed_defective_root(inner_stem)
                    or inner_stem in self.morphology.verb_stems
                ):
                    return True

        return False

    def _check_compound_perfect_double_lock(self, norm_s: str) -> bool:
        if not norm_s or len(norm_s) < 3:
            return False

        for bound_stem, affixes, rule in self.phonology.invert_labialized_coverb_auxiliary(norm_s):
            # The -ቷል inversion returns the coverb's *pre-vocalic base* and
            # puts -ቶ in ``affixes``.  License the reconstructed coverb
            # (base + -ቶ), not the base by itself: weak/final-weak coverbs can
            # be too short or have a non-coverb vowel until that suffix is
            # restored (e.g. ሙቷል -> ሙቶ + አል, ነኩቷል -> ነኩቶ + አል).
            if rule == "compound_coverb_weak_to_al":
                coverb_suffix = next(
                    (s for s in affixes if s in self.morphology.suffix_verb_coverb),
                    None,
                )
                if coverb_suffix and self.morphology.is_valid_coverb_stem(
                    bound_stem + coverb_suffix
                ):
                    return True

            if (
                self.morphology.is_licensed_defective_root(bound_stem)
                or self.morphology.is_valid_coverb_stem(bound_stem)
                or self.morphology.is_valid_perfective_template(bound_stem)
                or bound_stem in self.morphology.verb_stems
            ):
                return True

        if hasattr(self.morphology, "compound_auxiliary_suffixes"):
            for sfx, meta in self.morphology.compound_auxiliary_suffixes.items():
                if norm_s.endswith(sfx) and len(norm_s) > len(sfx) + 1:
                    stem = norm_s[:-len(sfx)] + meta.get("stem_suffix", "")
                    if (
                        self.morphology.is_valid_coverb_stem(stem)
                        or self.morphology.is_licensed_defective_root(stem)
                    ):
                        return True

        return False

    def _check_coverb_double_lock(self, norm_s: str) -> bool:
        if not norm_s or len(norm_s) < 2:
            return False

        iv_prefixes = getattr(self.morphology, "prefix_verb_iv", ("ይ", "ት", "እ", "እን"))
        if any(norm_s.startswith(p) for p in iv_prefixes):
            return False

        derived_prefixes = ("አስተ", "ተን", "አን", "አስ", "ተ", "አ")

        if norm_s.endswith("ተው") and len(norm_s) >= 3:
            return True

        # 3PL coverbs with bare -w (-ው) after Geez vowel strictly require a triradical+ base (e.g. ሰብረ-ው, መርምረ-ው, len >= 4).
        # In 3-letter words (ተተው, ተለው), 'ው' is the root radical C2 of {ተ-ወ}, not a converb suffix.
        if norm_s.endswith("ው") and len(norm_s) >= 4:
            penult_c = norm_s[-2]
            _, penult_o = self.phonology.get_char_info(penult_c)
            if penult_o == 0:
                c6 = self.phonology.shift_order(penult_c, 5)
                stem_c6 = norm_s[:-2] + c6
                c6_orders = tuple(self.phonology.get_vocalic_orders(stem_c6))
                if c6_orders in self.morphology.LESLAU_COVERB_PATTERNS or self.morphology.is_valid_coverb_stem(stem_c6):
                    return True

        if norm_s.endswith("ት") and len(norm_s) >= 3:
            penult_c = norm_s[-2]
            _, penult_o = self.phonology.get_char_info(penult_c)
            if penult_o == 6:
                c6 = self.phonology.shift_order(penult_c, 5)
                stem_c6 = norm_s[:-2] + c6
                if (
                    tuple(self.phonology.get_vocalic_orders(stem_c6)) in self.morphology.LESLAU_COVERB_PATTERNS
                    or self.morphology.is_valid_coverb_stem(stem_c6)
                ):
                    return True
            elif norm_s.endswith("ቶት") and len(norm_s) >= 4:
                return True

        if norm_s.endswith("ታ") and len(norm_s) >= 4 and not norm_s.startswith(("መ", "ማ")):
            _, penult_o = self.phonology.get_char_info(norm_s[-2])
            if penult_o == 5:
                base_t = norm_s[:-1]
                if self.morphology.is_valid_coverb_stem(base_t):
                    return True

        if norm_s.endswith("ቶ") and len(norm_s) >= 3:
            weak_base = norm_s[:-1]
            citations = self.morphology.generate_lemma_candidates(weak_base)
            if any(
                self.morphology.is_valid_perfective_template(c) or c in self.morphology.verb_stems
                for c in citations
            ) or self.morphology.is_valid_coverb_stem(weak_base):
                return True

            base_orders = self.phonology.get_vocalic_orders(weak_base)
            if base_orders and tuple(base_orders) in {(3, 1), (0, 1), (6, 1)}:
                return True

            for pfx in derived_prefixes:
                if weak_base.startswith(pfx) and len(weak_base) > len(pfx) + 1:
                    inner_core = weak_base[len(pfx):]
                    inner_orders = self.phonology.get_vocalic_orders(inner_core)
                    if inner_orders and inner_orders[-1] in (5, 1, 2, 3, 0):
                        return True

        orders = self.phonology.get_vocalic_orders(norm_s)
        if not orders or orders[-1] != 6:
            return False

        coverb_patterns = self.morphology.LESLAU_COVERB_PATTERNS
        if tuple(orders) in coverb_patterns:
            return True

        for pfx in derived_prefixes:
            if norm_s.startswith(pfx) and len(norm_s) >= len(pfx) + 2:
                core = norm_s[len(pfx):]
                core_orders = tuple(self.phonology.get_vocalic_orders(core))
                if core_orders and core_orders[-1] == 6 and core_orders in coverb_patterns:
                    return True

        if len(norm_s) == 2:
            o0 = orders[0]
            if o0 == 5:
                citations = self.morphology.generate_lemma_candidates(norm_s)
                if any(self.morphology.is_valid_perfective_template(c) or c in self.morphology.verb_stems for c in citations):
                    return True
            elif o0 == 1:
                rec_lemma = self.phonology.shift_order(norm_s[0], 6) + self.phonology.shift_order(norm_s[1], 0)
                if self.morphology.is_valid_perfective_template(rec_lemma) or rec_lemma in self.morphology.verb_stems:
                    return True
            elif o0 in (2, 4):
                c0_hames = self.phonology.shift_order(norm_s[0], 4)
                c0_geez = self.phonology.shift_order(norm_s[0], 0)
                c1_geez = self.phonology.shift_order(norm_s[1], 0)
                for rec_cand in (c0_hames + c1_geez, c0_geez + c1_geez):
                    if self.morphology.is_valid_perfective_template(rec_cand) or rec_cand in self.morphology.verb_stems:
                        return True
            elif o0 == 0:
                c1_geez = self.phonology.shift_order(norm_s[1], 0)
                geminate_rec = norm_s[0] + c1_geez + c1_geez
                if self.morphology.is_valid_perfective_template(geminate_rec) or geminate_rec in self.morphology.verb_stems:
                    return True

        return False

    def _check_imperative_jussive_double_lock(self, norm_s: str) -> Optional[str]:
        if not norm_s:
            return None

        # Suppletive Imperative Commands from Morphemes Config (e.g. ና = "come!")
        if hasattr(self.morphology, "suppletive_stem_templates") and norm_s in self.morphology.suppletive_stem_templates:
            supp_meta = self.morphology.suppletive_stem_templates[norm_s]
            if supp_meta.get("pos") in ("JUSSIVE_IMPERATIVE", "VB:JUSSIVE_IMPERATIVE"):
                return "JUSSIVE_IMPERATIVE"

        if len(norm_s) < 2:
            return None

        if norm_s.startswith(("መ", "ማ")) and len(norm_s) >= 3:
            if self.morphology.is_valid_verbal_noun_stem(norm_s) or (norm_s.endswith("ት") and len(norm_s) >= 4):
                return None

        orders = tuple(self.phonology.get_vocalic_orders(norm_s))
        if orders in {(0, 3), (3, 3)}:
            return None

        SHAPE_D_CORE_PATTERNS = self.morphology.SHAPE_D_JUSSIVE_ORDERS

        # Branch 1: Jussive Modal Prefixes (ል-, እን-, ይ-, ት-, ያ-)
        for j_pfx in ("ይ", "ል", "እን", "ት", "ያ"):
            if norm_s.startswith(j_pfx) and len(norm_s) >= len(j_pfx) + 1:
                # No coalescence restoration here.  The surface ያ- of a
                # causative jussive is peeled as ይ + አ- by the decomposition
                # layer (_peel_prefix_coalescence), so re-inserting the አ- into
                # the remainder would certify the still-unpeeled surface
                # (ያስተምር -> አስተምር) as an autonomous verbal lemma and shadow the
                # correct imperfective/relative decomposition of the same form.
                rem = norm_s[len(j_pfx):]
                if rem.endswith("ኡ") and len(rem) >= 3:
                    rem = rem[:-1]

                if rem.endswith("ው") and len(rem) >= 3:
                    penult_c = rem[-2]
                    _, p_ord = self.phonology.get_char_info(penult_c)
                    if p_ord == 0:
                        rem = rem[:-2] + self.phonology.shift_order(penult_c, 5)
                    elif p_ord in (3, 5):
                        rem = rem[:-1]

                rem_orders = tuple(self.phonology.get_vocalic_orders(rem))

                if self._is_genuinely_syncretic(j_pfx, rem, rem_orders):
                    return "IMPERFECTIVE:JUSSIVE"

                if rem_orders in SHAPE_D_CORE_PATTERNS or self.morphology.is_valid_jussive_stem(rem):
                    return "JUSSIVE_IMPERATIVE"

        if len(norm_s) >= 3 and norm_s.endswith(("ኢ", "ኡ")):
            bare_base = norm_s[:-1]
            base_orders = tuple(self.phonology.get_vocalic_orders(bare_base))
            if base_orders in SHAPE_D_CORE_PATTERNS or self.morphology.is_valid_jussive_stem(bare_base):
                return "JUSSIVE_IMPERATIVE"

        if norm_s.startswith("ያ"):
            return None

        # Bare biconsonantal forms carrying a robust nominal paradigm (ቤት, ሰው,
        # ልብ, ውሃ, እጅ) are nouns, not jussive/imperative bases.  The biconsonantal
        # "hollow jussive" licensing is purely orthographic (final Sadis), so it
        # must yield to the stronger paradigmatic nominal evidence, exactly as
        # _check_perfective_double_lock already does for bare 2-char stems.
        if len(norm_s) == 2 and self.evaluate_nominal_cohort(norm_s).robust_plural:
            return None

        if orders in SHAPE_D_CORE_PATTERNS or self.morphology.is_valid_jussive_stem(norm_s):
            return "JUSSIVE_IMPERATIVE"

        return None

    def _check_perfective_double_lock(self, norm_s: str) -> bool:
        if not norm_s or len(norm_s) < 2:
            return False

        iv_prefixes = getattr(self.morphology, "prefix_verb_iv", ("ይ", "ት", "እ", "እን"))
        if any(norm_s.startswith(p) for p in iv_prefixes) or norm_s.endswith("ቶ"):
            return False

        orders = tuple(self.phonology.get_vocalic_orders(norm_s))
        if not orders or orders[-1] == 6:
            return False

        has_derived_voice = any(norm_s.startswith(p) for p in ("ተ", "ተን", "ታ", "አ", "አን", "አስ", "አስተ"))
        n_ev = self.evaluate_nominal_cohort(norm_s)
        # NOTE: the 2-char TSV registry is a raw whitelist of real words, not a
        # list of verb roots, so membership there is NOT verbal evidence.
        is_known_verb_root = (
            norm_s in self.morphology.verb_stems
            or norm_s in getattr(self.morphology, "suppletive_stem_templates", {})
            or orders in {(0, 3), (3, 3)}
        )
        if (n_ev.robust_plural or n_ev.plural) and not has_derived_voice and len(norm_s) <= 2 and not is_known_verb_root:
            return False

        perf_patterns = self.morphology.LESLAU_PERFECTIVE_PATTERNS
        if orders in perf_patterns:
            return True

        if (
            self.morphology.is_licensed_defective_root(norm_s)
            or self.morphology.is_valid_perfective_template(norm_s)
            or norm_s in self.morphology.verb_stems
        ):
            return True

        derived_prefixes = ("አስተ", "ተን", "አን", "አስ", "ተ", "አ", "ታ")
        for pfx in derived_prefixes:
            if norm_s.startswith(pfx) and len(norm_s) >= len(pfx) + 2:
                core = norm_s[len(pfx):]
                core_orders = tuple(self.phonology.get_vocalic_orders(core))

                if not core_orders or core_orders[-1] not in (0, 3, 1):
                    continue

                if core_orders in perf_patterns or self.morphology.is_valid_perfective_template(core):
                    return True

        return False

    def _is_genuinely_syncretic(
        self, pfx: str, rem: str, rem_orders: Tuple[int, ...]
    ) -> bool:
        """
        3-Tier Morphotactic Dispatcher:
          Tier 1: Exclusive sets -> Instant Pure Single Classification
          Tier 2: Syncretic sets -> Instant Dual Classification (100% homophonous)
          Tier 3: Ambiguous sets -> Cohort & Lexical Alternate Probe
        """
        if not rem or len(rem) < 2:
            return False

        # Concord Firewall: 1S 'እ-' is 100% exclusively imperfective (1S Jussive uses 'ል-')
        if pfx == "እ":
            return False

        # --- Tier 2: Canonical Syncretic Sets (Instant Dual Tag) ---
        # Covers Sabe, Ka'eb-medial, Dikala, Type C, Frequentatives, Inceptives, Reduplicatives
        if rem_orders in getattr(self.morphology, "SYNCRETIC_IMPERFECTIVE_JUSSIVE_PATTERNS", set()):
            return True

        # Inherent Passive Roots under subject concord (ይቀበል, ይመረመር)
        if rem_orders in getattr(self.morphology, "PASSIVE_JUSSIVE_ROOT_PATTERNS", set()):
            return True

        # Explicit Lexical Whitelists (Defectives & Biradical Type B)
        if (
            rem in getattr(self.morphology, "biradical_type_b_verbs", set())
            or rem in getattr(self.morphology, "syncretic_imp_juss_lexemes", set())
            or rem in ("መንን", "ሸኝ", "ለይ", "በጅ", "ተው", "ገጭ", "ፈጅ", "ፈጭ", "ነጭ", "ሻ")
        ):
            return True

        # --- Tier 3: Ambiguous Shapes (Type A/B Split Probe) ---
        # If pattern is NOT in the ambiguous set, it is NOT syncretic!
        if rem_orders not in getattr(self.morphology, "AMBIGUOUS_IMPERFECTIVE_JUSSIVE_PATTERNS", set()):
            return False

        MIN_LEX_FREQ = 3

        # Ambiguous Shape A: Triradicals [0, 5, 5] (ሰብር vs ፈልግ)
        if len(rem) == 3 and rem_orders == (0, 5, 5):
            c1, c2, c3 = rem[0], rem[1], rem[2]
            c1_sadis = self.phonology.shift_order(c1, 5)
            c2_geez = self.phonology.shift_order(c2, 0)

            # Type A Witnesses:
            type_a_juss_stem = c1_sadis + c2_geez + c3
            type_a_juss_word = (pfx or "ይ") + type_a_juss_stem
            type_a_infinitive = "መ" + c1_sadis + c2_geez + c3

            if (
                self.vocab_freq.get(self._norm(type_a_juss_stem), 0) >= MIN_LEX_FREQ
                or self.vocab_freq.get(self._norm(type_a_juss_word), 0) >= MIN_LEX_FREQ
                or self.vocab_freq.get(self._norm(type_a_infinitive), 0) >= MIN_LEX_FREQ
            ):
                return False

            # Type B Witnesses:
            type_b_infinitive = "መ" + c1 + c2_geez + c3
            if (
                self.vocab_freq.get(self._norm(rem), 0) >= MIN_LEX_FREQ
                or self.vocab_freq.get(self._norm(type_b_infinitive), 0) >= MIN_LEX_FREQ
            ):
                return True

            return False

        # Ambiguous Shape B: Final-Weak [0, 3] (ለፋ, ሰማ vs ለካ, ጠጣ)
        if len(rem) == 2 and rem_orders == (0, 3):
            c1, c2 = rem[0], rem[1]
            c1_sadis = self.phonology.shift_order(c1, 5)

            # Type A Witnesses:
            type_a_juss_stem = c1_sadis + c2
            type_a_juss_word = (pfx or "ይ") + type_a_juss_stem
            type_a_infinitive = "መ" + c1_sadis + c2 + "ት"

            if (
                self.vocab_freq.get(self._norm(type_a_juss_stem), 0) >= MIN_LEX_FREQ
                or self.vocab_freq.get(self._norm(type_a_juss_word), 0) >= MIN_LEX_FREQ
                or self.vocab_freq.get(self._norm(type_a_infinitive), 0) >= MIN_LEX_FREQ
            ):
                return False

            if rem in getattr(self.morphology, "biradical_type_b_verbs", set()):
                return True

            type_b_infinitive = "መ" + c1 + c2 + "ት"
            if self.vocab_freq.get(self._norm(type_b_infinitive), 0) >= MIN_LEX_FREQ:
                return True

            return False

        return False
        """
        Discriminates genuinely syncretic imperfective/jussive verbs (Type B, Type C, defectives,
        and reciprocal bound stems) from Type A verbs whose imperfective stem is strictly bound.
        Self-contained: does NOT fall back to raw template intersection.
        """
        if not rem or len(rem) < 2:
            return False

        # 1. Concord Firewall: 1S 'እ-' is 100% exclusively imperfective in Amharic (1S Jussive uses 'ል-')
        if pfx == "እ":
            return False

        # 2. Inherent Syncretic Templates (from leslau_templates.json)
        # Covers Sabe, Ka'eb-medial, Dikala, Type C, Frequentatives, Inceptives, Reduplicatives
        if rem_orders in getattr(self.morphology, "SYNCRETIC_CONTINGENT_JUSSIVE_PATTERNS", set()):
            return True

        # 3. Explicit Lexical Whitelists (Defectives & Biradical Type B)
        if (
            rem in getattr(self.morphology, "biradical_type_b_verbs", set())
            or rem in getattr(self.morphology, "syncretic_imp_juss_lexemes", set())
            or rem in ("መንን", "ሸኝ", "ለይ", "በጅ", "ተው", "ገጭ", "ፈጅ", "ፈጭ", "ነጭ", "ሻ")
        ):
            return True

        # Inherent reciprocal & passive bound stems
        if (
            rem_orders in getattr(self.morphology, "BOUND_JUSSIVE_STEM_PATTERNS", set())
            or rem_orders in getattr(self.morphology, "PASSIVE_JUSSIVE_ROOT_PATTERNS", set())
        ):
            return True

        # Lexical attestation bar for a single word in the dictionary: min 3 occurrences
        MIN_LEX_FREQ = 3

        # 4. Triradicals with order [0, 5, 5] (e.g. ሰብር vs ፈልግ)
        # 4. Triradicals with order [0, 5, 5] (e.g. ሰብር vs ፈልግ)
        if len(rem) == 3 and rem_orders == (0, 5, 5):
            c1, c2, c3 = rem[0], rem[1], rem[2]
            c1_sadis = self.phonology.shift_order(c1, 5)
            c2_geez = self.phonology.shift_order(c2, 0)

            # Type A Witnesses:
            # 1. Type A jussive stem/word: C1(Sadis) C2(Ge'ez) C3(Sadis) e.g. ስበር, ይስበር, ልመድ
            type_a_juss_stem = c1_sadis + c2_geez + c3
            type_a_juss_word = (pfx or "ይ") + type_a_juss_stem
            # 2. Type A infinitive: mä + C1(Sadis) C2(Ge'ez) C3(Sadis) e.g. መስበር, መልቀም, መድረስ
            type_a_infinitive = "መ" + c1_sadis + c2_geez + c3

            norm_juss_stem = self._norm(type_a_juss_stem)
            norm_juss_word = self._norm(type_a_juss_word)
            norm_type_a_inf = self._norm(type_a_infinitive)

            if (
                self.vocab_freq.get(norm_juss_stem, 0) >= MIN_LEX_FREQ
                or self.vocab_freq.get(norm_juss_word, 0) >= MIN_LEX_FREQ
                or self.vocab_freq.get(norm_type_a_inf, 0) >= MIN_LEX_FREQ
            ):
                return False

            # Type B Witnesses:
            # 1. Bare Type B command: C1(Ge'ez) C2(Sadis) C3(Sadis) e.g. ፈልግ, ቀድስ, ለምን
            norm_rem = self._norm(rem)
            # 2. Type B infinitive: mä + C1(Ge'ez) C2(Ge'ez) C3(Sadis) e.g. መፈለግ, መቀደስ, መለመን
            type_b_infinitive = "መ" + c1 + c2_geez + c3
            norm_type_b_inf = self._norm(type_b_infinitive)

            if (
                self.vocab_freq.get(norm_rem, 0) >= MIN_LEX_FREQ
                or self.vocab_freq.get(norm_type_b_inf, 0) >= MIN_LEX_FREQ
            ):
                return True

            # Absence of corpus evidence -> Asymmetry Principle: default to Type A (NOT syncretic)
            return False

        # 5. Final-Weak Biradicals with order [0, 3] (e.g. ለፋ, ሰማ vs ለካ, ጠጣ)
        if len(rem) == 2 and rem_orders == (0, 3):
            c1, c2 = rem[0], rem[1]
            c1_sadis = self.phonology.shift_order(c1, 5)

            # Type A Witnesses:
            # 1. Type A jussive: C1(Sadis) C2(Rabe) e.g. ልፋ, ስማ, ምጣ, ብላ
            type_a_juss_stem = c1_sadis + c2
            type_a_juss_word = (pfx or "ይ") + type_a_juss_stem
            # 2. Type A infinitive: mä + C1(Sadis) C2(Rabe) + t e.g. መልፋት, መስማት, መምጣት, መብላት
            type_a_infinitive = "መ" + c1_sadis + c2 + "ት"

            norm_juss_stem = self._norm(type_a_juss_stem)
            norm_juss_word = self._norm(type_a_juss_word)
            norm_type_a_inf = self._norm(type_a_infinitive)

            if (
                self.vocab_freq.get(norm_juss_stem, 0) >= MIN_LEX_FREQ
                or self.vocab_freq.get(norm_juss_word, 0) >= MIN_LEX_FREQ
                or self.vocab_freq.get(norm_type_a_inf, 0) >= MIN_LEX_FREQ
            ):
                return False

            # If in biradical Type B whitelist -> syncretic
            if rem in getattr(self.morphology, "biradical_type_b_verbs", set()):
                return True

            # Type B infinitive: mä + C1(Ge'ez) C2(Rabe) + t e.g. መለካት, መጠጣት
            type_b_infinitive = "መ" + c1 + c2 + "ት"
            norm_type_b_inf = self._norm(type_b_infinitive)
            if self.vocab_freq.get(norm_type_b_inf, 0) >= MIN_LEX_FREQ:
                return True

            # Absence of corpus evidence -> default to Type A (NOT syncretic)
            return False

        return False

    def _check_contingent_imperfective_double_lock(self, norm_s: str) -> Optional[str]:
        if not norm_s or len(norm_s) < 2 or norm_s.endswith("ል") or norm_s.endswith("ቶ"):
            return None

        # 1. T-initial root imperatives (ትከል, ትረፍ, ትረክ from ተከለ, ተረፈ, ተረከ):
        # 'ት' is root radical C1, not a subject prefix. Form is 100% Type A imperative [5, 0, 5].
        if norm_s.startswith("ት") and tuple(self.phonology.get_vocalic_orders(norm_s)) == (5, 0, 5):
            return None

        MIN_LEX_FREQ = 3
        concord_prefixes = ("ይ", "ት", "እ", "እን", "ያ", "ታ", "ና")
        for pfx in concord_prefixes:
            if norm_s.startswith(pfx) and len(norm_s) > len(pfx) + 1:
                rem = norm_s[len(pfx):]
                if pfx in ("ያ", "ታ", "ና"):
                    rem = "አ" + rem

                if rem.endswith(("ኡ", "ኢ")) and len(rem) >= 3:
                    rem = rem[:-1]

                rem_orders = tuple(self.phonology.get_vocalic_orders(rem))

                # 2. Causative Jussives under 'ያ-' ending in Sadis (5) without auxiliary:
                # e.g. ያመልክት, ያመስግን, ያምበርክክ, ያርገብግብ, ያቁም, ያሙቅ, ያለስልስ
                # Main-clause contingent causatives require auxiliary -አል (ያመለክታል); bare forms are strictly Jussive.
                if pfx == "ያ" and len(rem) >= 3:
                    _, coda_ord = self.phonology.get_char_info(rem[-1])
                    if coda_ord == 5:
                        return None

                # 3. Hollow-A Jussives with stem [3, 5] (ይላክ, ይላስ, ይሳብ, ይሳም, ይማር, ይቻል):
                # Their contingent is [5, 5] (ይልክ, ይልስ, ይስብ). If the contingent exists, this form is Jussive!
                if pfx in ("ይ", "ት", "እን") and len(rem) == 2 and rem_orders == (3, 5):
                    cont_alt = (pfx or "ይ") + self.phonology.shift_order(rem[0], 5) + rem[1]
                    if self.vocab_freq.get(self._norm(cont_alt), 0) >= MIN_LEX_FREQ:
                        return None

                # 4. Defective Jussives with stem [5, 5] (ይስጥ, ይሽጥ, ይቅር, ይምሽ, ይሽሽ):
                # Their contingent is [0, 5] (ይሰጥ, ይሸጥ, ይቀር, ይመሽ, ይሸሽ). If contingent exists, this form is Jussive!
                if pfx in ("ይ", "ት", "እን") and len(rem) == 2 and rem_orders == (5, 5):
                    cont_alt = (pfx or "ይ") + self.phonology.shift_order(rem[0], 0) + rem[1]
                    if self.vocab_freq.get(self._norm(cont_alt), 0) >= MIN_LEX_FREQ:
                        return None

                # 5. Hollow-U Jussives with stem [1, 5] (ይሁን, ይሙት, ይሩጥ, ይቁም):
                # Their contingent is [6, 5] (ይሆን, ይሞት, ይሮጥ, ይቆም). Stem [1, 5] is strictly Jussive.
                if pfx in ("ይ", "ት", "እን") and len(rem) == 2 and rem_orders == (1, 5):
                    return None

                is_imp_shape = (
                    rem_orders in self.morphology.LESLAU_IMPERFECTIVE_PATTERNS
                    or self.morphology.is_valid_imperfective_stem(rem)
                    or rem in self.morphology.verb_stems
                )
                if is_imp_shape:
                    # Corpus-derived shape policy first (see
                    # tools/build_contingent_jussive_policy.py); the hand-written
                    # syncretism probe stays as the uncovered-shape fallback.
                    corpus_reading = self._corpus_contingent_jussive_reading(rem)
                    if corpus_reading == "BOTH":
                        return "IMPERFECTIVE:JUSSIVE"
                    if corpus_reading == "VB:JUSSIVE_IMPERATIVE":
                        return "JUSSIVE_IMPERATIVE"
                    if corpus_reading == "VB:IMPERFECTIVE":
                        return "IMPERFECTIVE"
                    if self._is_genuinely_syncretic(pfx, rem, rem_orders):
                        return "IMPERFECTIVE:JUSSIVE"
                    return "IMPERFECTIVE"

        return None

    def _evaluate_verbal_double_lock(self, norm_s: str) -> Optional[str]:
        if self._check_relative_imperfective_double_lock(norm_s):
            return "RELATIVE_IMPERFECTIVE"
        if self._check_relative_perfective_double_lock(norm_s):
            return "RELATIVE_PERFECTIVE"
        if self._check_compound_imperfect_double_lock(norm_s):
            return "COMPOUND_IMPERFECT"
        if self._check_compound_perfect_double_lock(norm_s):
            return "COMPOUND_PERFECT"
        if self._check_coverb_double_lock(norm_s):
            return "COVERB"
        
        contingent = self._check_contingent_imperfective_double_lock(norm_s)
        if contingent is not None:
            return contingent

        juss_imp = self._check_imperative_jussive_double_lock(norm_s)
        if juss_imp is not None:
            return juss_imp

        if self._check_perfective_double_lock(norm_s):
            return "PERFECTIVE"

        orders = tuple(self.phonology.get_vocalic_orders(norm_s))
        if orders in self.morphology.LESLAU_IMPERFECTIVE_PATTERNS or self.morphology.is_valid_imperfective_stem(norm_s):
            # Same nominal-evidence yield-out as the jussive/perfective locks:
            # a bare biconsonantal noun (ቤት, እጅ) is not an imperfective base.
            if len(norm_s) == 2 and self.evaluate_nominal_cohort(norm_s).robust_plural:
                return None
            return "IMPERFECTIVE"

        return None

    # ------------------------------------------------------------------
    # LAYER 1: MACRO-CATEGORY ARBITER (Structured Layer1Result Generator)
    # ------------------------------------------------------------------

    def identify_layer1(
        self, stem: str, candidate: Optional[Any] = None, exclude_word: Optional[str] = None
    ) -> Layer1Result:
        if stem == "ል":
            return Layer1Result(macro_tag=pos_tags.VERBAL, verbal_shape=VerbalShape.SHAPE_C_IMPERFECTIVE, verbal_subtag="IMPERFECTIVE")
        if not stem or len(stem) < 2:
            return Layer1Result(macro_tag=pos_tags.noun("CN"), verbal_shape=VerbalShape.UNKNOWN)

        norm_s = self._norm(stem)

        # STAGE 1: Closed Classes & Invariant Lexicon
        if norm_s in self.morphology.pronoun_stems:
            return Layer1Result(macro_tag=pos_tags.pronoun(), verbal_shape=VerbalShape.CLOSED_NOMINAL, closed_subtype="PRO")

        # Verbal Primitives (Auxiliaries, Copulas, Lexical Verb Whitelists)
        if (
            norm_s in self.morphology.auxiliary_verbs_standalone
            or norm_s.startswith(("አይደል", "አይደለ"))
        ):
            return Layer1Result(macro_tag=pos_tags.VERBAL, verbal_shape=VerbalShape.CLOSED_VERBAL, closed_subtype="AUX")

        # Lexical verbs and defective roots (guard against false overrides on count nouns like ቤት, ልጅ)
        n_ev_quick = self.evaluate_nominal_cohort(stem)
        is_protected_count_noun = (n_ev_quick.robust_plural or n_ev_quick.plural) and len(norm_s) == 2

        if not is_protected_count_noun and (
            (hasattr(self.morphology, "suppletive_stem_templates") and norm_s in self.morphology.suppletive_stem_templates)
            or norm_s in self.morphology.verb_stems
        ):
            return Layer1Result(macro_tag=pos_tags.VERBAL, verbal_shape=VerbalShape.UNKNOWN, verbal_subtag="")

        if self.morphology.is_licensed_defective_root(norm_s):
            # Biconsonantal root licensing is orthographic (final Sadis), so it
            # alone cannot resolve the category.  When the form also carries a
            # robust nominal paradigm (ቤት, ሰው, ልብ) the nominal reading must
            # survive; only forms without such evidence are pure VERBAL.
            if not (len(norm_s) == 2 and self.evaluate_nominal_cohort(norm_s).robust_plural):
                return Layer1Result(macro_tag=pos_tags.VERBAL, verbal_shape=VerbalShape.UNKNOWN, verbal_subtag="")

        # Closed Classes (adpositions, adverbs, conjunctions, interjections,
        # quantifiers, auxiliaries, ideophones).  These are orthogonal families
        # (or verbal-side for AUX/IDEO), not members of the nominal family.
        closed_pos = self.morphology.get_closed_class_pos(norm_s)
        if closed_pos is not None:
            closed_shape = (
                VerbalShape.CLOSED_VERBAL
                if closed_pos.startswith(pos_tags.VERBAL)
                else VerbalShape.CLOSED_NOMINAL
            )
            if closed_pos == pos_tags.auxiliary():
                return Layer1Result(macro_tag=pos_tags.auxiliary(), verbal_shape=closed_shape, closed_subtype="AUX")
            if not self._evaluate_verbal_double_lock(norm_s):
                return Layer1Result(
                    macro_tag=closed_pos,
                    verbal_shape=closed_shape,
                    closed_subtype=pos_tags.closed_subtype(closed_pos),
                )

        # Bare nominal/verbal homographs licensed by the shipped paradigm
        # matrix.  A form such as ራስ ("head" / "forget!") carries both a robust
        # nominal paradigm and a matrix paradigm cell (Imperative, Coverb, ...);
        # the cell names the verbal reading, so the tag is the disjunction
        # ``NOM:CN|VB:JUSSIVE_IMPERATIVE``.  Plain nouns (ቤት, ሰው) are not
        # matrix forms and deverbal nouns (መምህር, ጭንቀት) have no matrix cell.
        if len(norm_s) >= 2:
            bare_n_ev = self.evaluate_nominal_cohort(stem)
            closed_nom_early = self._nominal_side_closed_tag(closed_pos)
            has_nominal_evidence = (
                closed_nom_early is not None
                or bare_n_ev.robust_plural
                or bare_n_ev.true_pn_assoc
            )
            if has_nominal_evidence:
                readings = self.morphology.paradigm_readings_for(norm_s)
                verbal_readings = tuple(
                    tag for tag in readings if tag.startswith(pos_tags.VERBAL)
                )
                if verbal_readings:
                    nominal_side = closed_nom_early or (
                        pos_tags.noun("CN")
                        if bare_n_ev.robust_plural
                        else pos_tags.nominal()
                    )
                    core = self._core_verbal_lock(verbal_readings[0])
                    return Layer1Result(
                        macro_tag=pos_tags.disjoin(nominal_side, *verbal_readings),
                        verbal_shape=self.V_LOCK_TO_SHAPE.get(core, VerbalShape.UNKNOWN),
                        closed_subtype=closed_nom_early or "",
                        verbal_subtag=core,
                    )

        # Invariant Nominal Templates & Verbal Nouns
        if hasattr(self.morphology, "ambiguous_verbal_nouns") and norm_s in self.morphology.ambiguous_verbal_nouns:
            return Layer1Result(
                macro_tag=pos_tags.disjoin(pos_tags.noun(), "VN"),
                verbal_shape=VerbalShape.SHAPE_E_INFINITIVE,
            )

        if self.morphology.is_valid_verbal_noun_stem(norm_s):
            return Layer1Result(macro_tag="VN", verbal_shape=VerbalShape.SHAPE_E_INFINITIVE)

        if (
            norm_s in getattr(self.morphology, "geez_plurals", set())
            or norm_s in getattr(self.morphology, "confirmed_sparse_nouns", set())
            or self._has_overt_exclusive_nominal_affix(norm_s)
        ):
            return Layer1Result(macro_tag=pos_tags.noun("CN"), verbal_shape=VerbalShape.UNKNOWN)

        if self.morphology.is_abstract_noun_stem(norm_s):
            return Layer1Result(macro_tag=pos_tags.noun("MN"), verbal_shape=VerbalShape.UNKNOWN)

        # Relative Verbs Double-Lock Interception
        if self._check_relative_imperfective_double_lock(norm_s):
            return Layer1Result(macro_tag=pos_tags.subset("VB:IMPERFECTIVE", "RELATIVE"), verbal_shape=VerbalShape.SHAPE_C_IMPERFECTIVE, verbal_subtag="RELATIVE_IMPERFECTIVE")

        if self._check_relative_perfective_double_lock(norm_s):
            return Layer1Result(macro_tag=pos_tags.subset("VB:PERFECTIVE", "RELATIVE"), verbal_shape=VerbalShape.SHAPE_A_PERFECTIVE, verbal_subtag="RELATIVE_PERFECTIVE")

        # Compound Verbs Double-Lock Interception
        if self._check_compound_imperfect_double_lock(norm_s):
            return Layer1Result(macro_tag=pos_tags.subset("VB:IMPERFECTIVE", "COMPOUND"), verbal_shape=VerbalShape.SHAPE_C_IMPERFECTIVE, verbal_subtag="COMPOUND_IMPERFECT")

        if self._check_compound_perfect_double_lock(norm_s):
            return Layer1Result(macro_tag=pos_tags.subset("VB:COVERB", "COMPOUND"), verbal_shape=VerbalShape.SHAPE_B_COVERB, verbal_subtag="COMPOUND_PERFECT")

        # Converb System Double-Lock Interception
        if self._check_coverb_double_lock(norm_s):
            return Layer1Result(macro_tag="VB:COVERB", verbal_shape=VerbalShape.SHAPE_B_COVERB, verbal_subtag="COVERB")

        # Affix-Forced Candidate Context Constraints
        if candidate is not None:
            if any(s in ("ኦ", "ቼ", "ተው", "ቶ", "ታ", "ተን", "ተህ", "ተሽ") for s in candidate.suffixes):
                return Layer1Result(macro_tag=pos_tags.VERBAL, verbal_shape=VerbalShape.SHAPE_B_COVERB, verbal_subtag="COVERB")

        # ==================================================================
        # PARALLEL 2-AXIS EVIDENCE MATRIX (E_Nominal x E_Verbal Double-Lock)
        # ==================================================================
        v_lock = self._evaluate_verbal_double_lock(norm_s)
        v_shape = self.V_LOCK_TO_SHAPE.get(v_lock or "", VerbalShape.UNKNOWN)

        n_ev = self.evaluate_nominal_cohort(stem)
        stem_orders = tuple(self.phonology.get_vocalic_orders(stem)) if stem else ()
        has_authoritative_plural = (n_ev.robust_plural or n_ev.plural) and stem_orders != (5, 0, 5)

        has_nom_cohort = (
            has_authoritative_plural
            or (n_ev.total_slots >= 2)
        )
        if exclude_word and exclude_word != stem:
            n_ev_orig = self.evaluate_nominal_cohort(exclude_word)
            if n_ev_orig.plural or n_ev_orig.robust_plural:
                has_nom_cohort = True

        # A closed-class reading only counts as the nominal side of a dual when
        # it is a nominal-side family (ADP/ADV/CONJ/INT, quantifier MN).
        # Ideophones are verbal-side and never supply a nominal reading.
        closed_nom_tag = self._nominal_side_closed_tag(closed_pos)
        has_closed_nom = closed_nom_tag is not None

        # Bare surface homographs: a stem with a robust nominal paradigm that is
        # also an attested command form in the shipped paradigm matrix (ራስ
        # "head"/"forget!", መርዝ "poison"/"poison!") is a genuine dual.  This is
        # the licensor that separates them from deverbal nouns (መምህር, ጭንቀት)
        # and from plain nouns (ቤት, ሰው), which are not command forms.
        bare_homograph = self._is_bare_verbal_homograph(norm_s)

        # QUADRANT 3: DUAL AMBIGUITY (e.g. መልስ -> NOM:CN|VB:JUSSIVE_IMPERATIVE,
        # ቀድሞ -> ADV|VB:COVERB).  The superfamily is repeated on each member.
        if (has_nom_cohort or has_closed_nom) and v_lock:
            # Deverbal nouns (agent/instrument/result nouns such as መምህር and
            # ጭንቀት) match the jussive/perfective templates orthographically but
            # have no finite verbal concord: no ይ-/ል-/*-ኧው/... form of the root is
            # attested, while the nominal paradigm is robust.  Per the documented
            # Priority 6 policy, such stems are classified by their nominal cohort
            # evidence and must not be emitted as nominal/verbal duals.  Genuine
            # duals (ለውጥ, መልስ, ፈውስ, ድረስ, በላ) keep the tag because the corpus
            # attests their verbal concord; attested bare command homographs
            # (ራስ, መርዝ, ጋት) keep it because the paradigm matrix lists them.
            if (
                v_lock in ("JUSSIVE_IMPERATIVE", "PERFECTIVE", "IMPERFECTIVE")
                and n_ev.robust_plural
                and not has_closed_nom
                and not bare_homograph
                and not self.evaluate_verbal_cohort(stem, exclude_word=exclude_word).has_finite_concord
            ):
                return Layer1Result(macro_tag=pos_tags.noun("CN"), verbal_shape=VerbalShape.UNKNOWN)

            core_lock = self._core_verbal_lock(v_lock)

            # The double-lock above and ``subclassify_verbal_stem`` are two copies of
            # the same grammar, and they disagree: for መላ the double-lock says
            # IMPERFECTIVE while the subclassifier and the corpus-derived policy both
            # say PERFECTIVE, and for አገባ it says JUSSIVE_IMPERATIVE against a
            # PERFECTIVE gold.  Every other verbal decision in this module is made by
            # the subclassifier, so let it arbitrate the verbal half of the dual tag
            # too, and keep the double-lock as the fallback when it declines to commit.
            q3_pfx = tuple(candidate.prefixes) if candidate is not None else ()
            q3_sfx = tuple(candidate.suffixes) if candidate is not None else ()
            q3_sub = self.subclassify_verbal_stem(norm_s, q3_pfx, q3_sfx)
            if q3_sub == "VN":
                # Not a noun/verb dual at all - a verbal noun.  Tagging ማለት as
                # NOM:CN|VB:IMPERFECTIVE loses the one reading the corpus attests.
                return Layer1Result(
                    macro_tag="VN", verbal_shape=VerbalShape.SHAPE_E_INFINITIVE
                )
            if q3_sub and q3_sub != "VB":
                core_lock = (
                    q3_sub.split("VB:", 1)[-1] if q3_sub.startswith("VB:") else q3_sub
                )

            verbal_readings = pos_tags.disjuncts(
                pos_tags.canonical_tag(pos_tags.verb(core_lock))
            )
            return Layer1Result(
                macro_tag=pos_tags.disjoin(closed_nom_tag or pos_tags.noun("CN"), *verbal_readings),
                verbal_shape=v_shape,
                closed_subtype=closed_nom_tag or "",
                verbal_subtag=core_lock,
            )

        # QUADRANT 2: PURE VERBAL (e.g. ስበር -> VB:JUSSIVE_IMPERATIVE, ይሆናል -> VB:IMPERFECTIVE:COMPOUND)
        if v_lock and not has_nom_cohort:
            core_lock = self._core_verbal_lock(v_lock)
            return Layer1Result(
                macro_tag=pos_tags.canonical_tag(pos_tags.verb(core_lock)),
                verbal_shape=v_shape,
                verbal_subtag=core_lock,
            )

        # QUADRANT 1: PURE NOMINAL (e.g. ቤት -> NOM:CN, ሰባሪ -> NOM:CN, አስተማሪ -> NOM:CN)
        if has_nom_cohort and not v_lock:
            return Layer1Result(macro_tag=pos_tags.noun("CN"), verbal_shape=VerbalShape.UNKNOWN)

        # Fallback Concord Check for Inflected Stems with Candidate Context
        if candidate is not None:
            v_ev = self.evaluate_verbal_cohort(stem, exclude_word=exclude_word)
            if v_ev.has_finite_concord:
                return Layer1Result(macro_tag=pos_tags.VERBAL, verbal_shape=VerbalShape.UNKNOWN)

        # 2-Character Firewall
        if len(stem) == 2:
            if norm_s in getattr(self.morphology, "TWO_CHAR_CONTINGENTS", set()) or norm_s.startswith(("ይ", "ያ")):
                return Layer1Result(
                    macro_tag=pos_tags.disjoin("VB:IMPERFECTIVE", "VB:JUSSIVE_IMPERATIVE"),
                    verbal_shape=VerbalShape.SHAPE_C_IMPERFECTIVE,
                    verbal_subtag="IMPERFECTIVE|JUSSIVE_IMPERATIVE",
                )
            if self.morphology.is_valid_perfective_template(norm_s):
                return Layer1Result(macro_tag="VB:PERFECTIVE", verbal_shape=VerbalShape.SHAPE_A_PERFECTIVE, verbal_subtag="PERFECTIVE")
            if self.morphology.is_valid_imperfective_stem(norm_s):
                return Layer1Result(macro_tag="VB:IMPERFECTIVE", verbal_shape=VerbalShape.SHAPE_C_IMPERFECTIVE, verbal_subtag="IMPERFECTIVE")
            return Layer1Result(macro_tag=pos_tags.noun("CN"), verbal_shape=VerbalShape.UNKNOWN)

        return Layer1Result(macro_tag=pos_tags.UNKNOWN, verbal_shape=VerbalShape.UNKNOWN)

    # ------------------------------------------------------------------
    # CANONICAL TAG HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _core_verbal_lock(lock: str) -> str:
        """Strip legacy family heads from a lock name (``NV:PERFECTIVE`` → ``PERFECTIVE``)."""
        core = lock
        while core.startswith(("NV:", "CN/", "N:VB:", "NOM:N:", "VERBAL:", "VRB:", "VB:")):
            core = core.split(":", 1)[-1]
        return core

    @staticmethod
    def _nominal_side_closed_tag(closed_pos: Optional[str]) -> Optional[str]:
        """The nominal-side reading for a closed class, or None if verbal-side."""
        if not closed_pos:
            return None
        if closed_pos in ("ADP", "ADV", "CONJ", "INT", "NUM"):
            return closed_pos
        if closed_pos in (pos_tags.noun("CN"), pos_tags.noun("PN"), pos_tags.noun("MN")):
            return closed_pos
        return None

    def _has_exclusive_nominal_suffix_context(self, suffixes: Tuple[str, ...]) -> bool:
        """True when the affix context admits only a nominal reading.

        Suffixes that are nominal-class (plural, definiteness, possessive,
        derivational, abstract, instrumental) exclude the verbal reading of an
        otherwise ambiguous stem, unless a verbal-class suffix is also present.
        """
        if not suffixes:
            return False
        morphology = self.morphology
        # Only morphology that has no verbal homograph at all counts: the
        # possessive (እሽ, ችን, ዬ) and definiteness (ኡ, ው, ቱ) markers double as
        # verbal person/object suffixes (አገርሽ "your country" vs አገርሽ "be shy!"),
        # so they cannot settle the reading on their own.
        nominal_class = set().union(*(
            set(getattr(morphology, name, set())) for name in (
                "suffix_nominal_plural_exclusive",
                "suffix_nominal_exclusive_derivational",
                "suffix_nominal_exclusive_abstract",
                "suffix_nominal_exclusive_instrumental",
            )
        ))
        # Plural allomorphs peeled by literal rule (affix_peeler._peel_plural_*
        # and the atomic plural+accusative splits) are not in the inventories.
        nominal_class |= {"ዎች", "ኦች", "ሎች", "ቶች"}
        return any(s in nominal_class for s in suffixes)

    def _is_bare_verbal_homograph(self, stem: str) -> bool:
        """True when a bare stem is both a noun and an attested command form.

        The command-form side is licensed by the shipped paradigm matrix
        (``data/10_column_verbs.tsv``) through
        ``morphology.is_paradigm_matrix_form()``.  This is a general,
        data-backed rule: every reviewed homograph row (ራስ, መርዝ, ጋት, ጥላ,
        ጻፍ, ላም …) is a matrix surface form, while plain nouns (ቤት, ሰው,
        ልብ, ውሃ, እጅ, ቀን) and deverbal nouns (መምህር, ጭንቀት) are not.
        """
        if not stem:
            return False
        norm_s = self._norm(stem)
        return (
            self.morphology.is_paradigm_matrix_form(norm_s)
            or self.morphology.is_paradigm_matrix_form(stem)
        )

    def identify_macro_category(
        self, stem: str, candidate: Optional[Any] = None, exclude_word: Optional[str] = None
    ) -> str:
        return self.identify_layer1(stem, candidate=candidate, exclude_word=exclude_word).macro_tag

    # ------------------------------------------------------------------
    # LAYER 2: CATEGORY-INTERNAL SUBCLASSIFICATION
    # ------------------------------------------------------------------

    def subclassify_nominal_stem(
        self, stem: str, candidate: Optional[Any] = None
    ) -> str:
        if not stem or len(stem) < 2:
            return pos_tags.noun("CN")

        norm_s = self._norm(stem)
        
        if norm_s in self.morphology.pronoun_stems:
            return pos_tags.pronoun()
        closed_pos = self.morphology.get_closed_class_pos(norm_s)
        if closed_pos is not None and closed_pos != pos_tags.auxiliary():
            return closed_pos

        if norm_s in getattr(self.morphology, "geez_plurals", set()) or norm_s in getattr(self.morphology, "confirmed_sparse_nouns", set()):
            return pos_tags.noun("CN")
        if self.morphology.is_abstract_noun_stem(norm_s):
            return pos_tags.noun("MN")

        n_ev = self.evaluate_nominal_cohort(stem)

        has_subj_prefix = candidate is not None and any(
            p in self.morphology.all_subject_prefixes for p in candidate.prefixes
        )

        is_ambig_vn = (
            hasattr(self.morphology, "ambiguous_verbal_nouns")
            and norm_s in self.morphology.ambiguous_verbal_nouns
        )
        if (is_ambig_vn or self.morphology.is_valid_verbal_noun_stem(stem)) and not has_subj_prefix:
            if is_ambig_vn or n_ev.robust_plural or n_ev.plural:
                return pos_tags.disjoin(pos_tags.noun("CN"), "VN")
            return "VN"

        if candidate is not None:
            if any(
                p in self.morphology.prefix_nominal_exclusive_associatives
                for p in candidate.prefixes
            ):
                return pos_tags.noun("PN")
            if any(
                s in (
                    self.morphology.suffix_nominal_exclusive_possessive
                    | self.morphology.suffix_nominal_exclusive_plural
                    | self.morphology.suffix_nominal_exclusive_abstract
                )
                for s in candidate.suffixes
            ):
                return pos_tags.noun("CN")

        if (
            n_ev.robust_plural
            and n_ev.true_pn_assoc
            and stem not in self.morphology.pronoun_stems
        ):
            return pos_tags.disjoin(pos_tags.noun("CN"), pos_tags.noun("PN"))

        if (
            n_ev.true_pn_assoc
            and not n_ev.robust_plural
            and stem not in self.morphology.pronoun_stems
        ):
            return pos_tags.noun("PN")

        if n_ev.robust_plural or n_ev.plural:
            return pos_tags.noun("CN")

        mn_evidence = int(n_ev.def_acc) + int(n_ev.possessive) + int(n_ev.abstract) + int(n_ev.compound)
        if not n_ev.robust_plural and not n_ev.true_pn_assoc:
            if mn_evidence >= self.mn_triangulation_min:
                return pos_tags.noun("MN")

        has_verbal_context = candidate is not None and (
            any(p in self.morphology.all_verbal_prefixes or p in self.morphology.prefix_relative_pronoun for p in candidate.prefixes)
            or any(s in self.morphology.all_verbal_suffixes or s in ("ኡ", "ት", "ው", "ኝ", "አል", "ዋል") for s in candidate.suffixes)
        )
        if self.morphology.is_valid_ideophone_template(stem) and not has_verbal_context:
            if (
                not self.morphology.is_surface_finite_verb(stem)
                and mn_evidence < self.mn_triangulation_min
            ):
                return pos_tags.ideophone()

        return pos_tags.noun("CN")

    def _resolve_fused_inflectional_prefix(
        self, stem: str, prefixes: Tuple[str, ...], suffixes: Tuple[str, ...]
    ) -> Optional[Tuple[Tuple[str, ...], str]]:
        """Return the affix-layer split of a substrate that carries its prefix.

        ``ስብ`` + ``አል`` needs the subject prefix in ``prefixes`` for the
        affix-concord lock to see the imperfective pattern; when the generator
        left that prefix inside the stem (``ይስብ`` + ``አል``) the lock comes back
        UNKNOWN and the candidate is tagged bare ``VB``.  MaximalLemmaArbitrator
        supplies the prefix split; the split is adopted only when the shape is
        licensed as an imperfective (Shape C) without it.  Returns
        ``(prefixes, stem)`` or ``None`` to leave the analysis untouched.
        """
        if self.arbitrator is None:
            return None
        split = self.arbitrator.split_fused_inflectional_prefix(stem)
        if split is None:
            return None

        fused_pfx, remainder = split
        candidate_prefixes = tuple(prefixes) + (fused_pfx,)
        shape = self.morphology.classify_licensed_verbal_shape(
            candidate_prefixes, suffixes, stem=remainder
        )
        if shape == VerbalShape.SHAPE_C_IMPERFECTIVE:
            return candidate_prefixes, remainder
        return None

    def _experiencer_perfective(
        self, stem: str, prefixes: Tuple[str, ...], suffixes: Tuple[str, ...]
    ) -> bool:
        """True when ``stem`` + suffixes is an experiencer active carrying an object.

        Experiencer (psych) verbs take their object in the active (``ደነቀው`` "it
        amazed him"), express the same experience in the ``ተ-`` passive without one
        (``ተደነቀ``), and form their imperative on the ``ተ-`` stem (``ተቸገር``, rarely
        ``ቸግረው``).  So an active form bearing an object clitic is a perfective.

        The corpus cannot settle this by shape: of its object-suffixed forms 313
        are jussive/imperative against 7 perfective, and every shape bucket is
        majority-jussive.  So a lexical list decides.  The gate is deliberately
        narrow -- suffixes present, all of them object clitics, no prefix at all --
        so the ``ተ-`` imperative and the rare active imperative are untouched.
        """
        return bool(
            suffixes
            and not prefixes
            and set(suffixes) <= self.morphology.suffix_verb_object_clitics
            and self.morphology.is_experiencer_root(self._norm(stem))
        )

    def subclassify_verbal_stem(
        self, stem: str, prefixes: Tuple[str, ...] = (), suffixes: Tuple[str, ...] = ()
    ) -> str:
        norm_s = self._norm(stem)

        subord_capable_aux = {"ነበር", "ነበረ"}
        if norm_s in self.morphology.auxiliary_verbs_standalone or norm_s.startswith(("አይደል", "አይደለ")):
            if not prefixes or norm_s not in subord_capable_aux:
                return pos_tags.auxiliary()

        if not prefixes and not suffixes and hasattr(self.morphology, "suppletive_stem_templates") and norm_s in self.morphology.suppletive_stem_templates:
            pos = self.morphology.suppletive_stem_templates[norm_s]["pos"]
            return f"VB:{pos}" if not pos.startswith("VB:") else pos

        has_subject_pfx = any(p in ("ይ", "ት", "እ", "እን", "ያ", "ታ", "ና") for p in prefixes)

        if not prefixes and not suffixes and len(norm_s) == 2:
            _, c_coda_o = self.phonology.get_char_info(norm_s[-1])
            if c_coda_o == 6:  # Order 6 (Sabe: ይዞ, ሆኖ, ሄዶ, ቆሞ, ሮጦ, ሞቶ)
                return "VB:COVERB"
            if c_coda_o == 0:  # Order 0 (Geez: ያዘ, ዋለ, ዳነ, ጣለ, ሆነ, ሄደ)
                return "VB:PERFECTIVE"
            if c_coda_o in (5, 3) and norm_s.startswith(("ይ", "ያ")):  # Sadis/Rabe: ያይ, ያሽ, ያጭ, ይሻ, ይማ, ይጣ
                return pos_tags.disjoin("VB:IMPERFECTIVE", "VB:JUSSIVE_IMPERATIVE")
            if norm_s in getattr(self.morphology, "TWO_CHAR_CONTINGENTS", set()):
                return pos_tags.disjoin("VB:IMPERFECTIVE", "VB:JUSSIVE_IMPERATIVE")

        all_subord_prefixes = {
            "ከ", "እንደ", "ስለ", "በ", "ለ", "ል", "ሊ", "ስ", "ሲ",
            "ብ", "ቢ", "ክ", "ኪ", "እንድ", "እንዲ", "እስክ", "እስኪ", "እስከ", "የ"
        }
        has_subord_or_rel = any(p in all_subord_prefixes or p.startswith("የ") for p in prefixes)
        has_only_voice_or_no_pfx = not has_subord_or_rel and not has_subject_pfx

        if has_only_voice_or_no_pfx and (not suffixes or set(suffixes).issubset({"ኡ", "ኢ"})):
            # A 2MS imperative and a 3MS perfective are written identically, but the
            # two coincide for only ~3.6% of paradigms.  The corpus-derived shape
            # policy (tools/build_contingent_jussive_policy.py, validated on
            # held-out lexemes) therefore decides whether the dual tag is warranted;
            # single readings fall through to the shape branches below, which apply
            # the same table.  The hand-written probe stays as the uncovered-shape
            # fallback.
            pi_surface = "".join(prefixes) + norm_s + "".join(suffixes)
            if self.morphology.perfective_imperative_reading(pi_surface) == "BOTH":
                return pos_tags.disjoin("VB:PERFECTIVE", "VB:JUSSIVE_IMPERATIVE")
            if (
                self.morphology.perfective_imperative_reading(pi_surface) is None
                and self.morphology.is_perfective_jussive_syncretic(norm_s, prefixes=prefixes, suffixes=suffixes)
            ):
                return pos_tags.disjoin("VB:PERFECTIVE", "VB:JUSSIVE_IMPERATIVE")

        has_aux = any(s in self.morphology.suffix_verb_auxiliary for s in suffixes)
        has_coverb_vowel = any(s in self.morphology.suffix_verb_coverb for s in suffixes)

        # The substrate may still carry its inflectional prefix: candidate
        # generation offers ይስባል as stem 'ይስብ' + auxiliary 'አል'.  Buried in the
        # stem, the prefix hides the imperfective pattern from the affix-concord
        # lock, so the candidate degrades to a bare VB.  An auxiliary suffix is
        # the signal that the stem is an imperfective base rather than a lexical
        # item; MaximalLemmaArbitrator decides whether the leading morpheme
        # belongs to the affix layer, and the shape must then license the split.
        # A coverb-only vowel already licenses its own compound reading, so the
        # coverb analysis is never rewritten as an imperfective.  The ambiguous
        # members of that inventory (ን, ው, ህ, ሽ) are object clitics here and do
        # not decide the reading, which is why only the coverb-only morphemes
        # block the split.
        if (
            has_aux and not any(
                s in self.morphology.suffix_verb_coverb_only for s in suffixes
            )
        ) or (not prefixes and not suffixes and len(stem) >= 3):
            fused = self._resolve_fused_inflectional_prefix(stem, prefixes, suffixes)
            if fused is not None:
                prefixes, stem = fused
                norm_s = self._norm(stem)

        has_subject_pfx = any(p in self.morphology.all_subject_prefixes for p in prefixes)
        if has_aux and has_coverb_vowel and not has_subject_pfx:
            return "VB:COVERB:COMPOUND"

        # The substrate may itself be a fused coverb + auxiliary: ጭኗል is ጭኖ +
        # አል, so ተጭኗል is a passive compound perfect and አስፈትቷል a causative one.
        # The fused auxiliary leaves no suffix for has_aux to see, and the
        # affix-concord shape reads the substrate as a plain perfective
        # (ተጭኗል) or jussive (አስፈትቷል) instead.  The compound-perfect lock is the
        # evidence: it only returns true when the inverted coverb base is a
        # licensed verb form, so the fused reading is licensed verbal material
        # and outranks the concord shape.
        # The lock is a property of the word, not of one hypothesis about where
        # its morphemes split: ትቷል is the compound perfect ትቶ + አል whether the
        # segmentation leaves ትቷል intact or peels the concord ት, and the peel
        # candidate's short substrate (ቷል) is below the lock's reach.  So test
        # the surface the prefixes and stem reconstruct -- and only when there
        # are no suffixes, because a suffix here is a morpheme *label*: the
        # auxiliary of ይሞታል is አል, whose surface is the -ል of the word, so
        # concatenating it would ask the lock about a word that does not exist.
        # (Suffixes are also why ተንቀዋሎ -- the coverb ተንቀዋል + the 3MS ending ኦ --
        # keeps its own reading.)
        #
        # The compound imperfect takes precedence, as it does in
        # _evaluate_verbal_double_lock: ይሞታል is ይ- + ሞታ + አል, the present
        # imperfective, not a present perfect.
        if not suffixes:
            surface = "".join(prefixes) + stem
            if not self._check_compound_imperfect_double_lock(
                self._norm(surface)
            ) and self._check_compound_perfect_double_lock(self._norm(surface)):
                return "VB:COVERB:COMPOUND"

        licensed_shape = self.morphology.classify_licensed_verbal_shape(prefixes, suffixes, stem=stem)

        # Bare finite surface (voice prefix at most, no object/subject suffix): a 2MS
        # imperative and a 3MS perfective are written identically.  They coincide for
        # only ~3.6% of paradigms, so the corpus-derived shape policy - validated on
        # held-out lexemes - decides whether the dual tag is warranted.
        bare_finite_surface = (
            has_only_voice_or_no_pfx
            and (not suffixes or set(suffixes).issubset({"ኡ", "ኢ"}))
            and licensed_shape in (VerbalShape.SHAPE_A_PERFECTIVE, VerbalShape.SHAPE_D_JUSSIVE_IMPERATIVE)
        )
        pi_reading = (
            self.morphology.perfective_imperative_reading(
                "".join(prefixes) + norm_s + "".join(suffixes)
            )
            if bare_finite_surface
            else None
        )

        if licensed_shape == VerbalShape.SHAPE_E_INFINITIVE:
            return "VN"

        if licensed_shape == VerbalShape.SHAPE_A_PERFECTIVE:
            if any(p == "ከ" for p in prefixes):
                return "VB:PERFECTIVE:ABLATIVE"
            if any(p == "እንደ" for p in prefixes):
                return "VB:PERFECTIVE:SIMILATIVE"
            if any(p == "ስለ" for p in prefixes):
                return "VB:PERFECTIVE:CAUSAL"
            if any(p == "በ" for p in prefixes):
                return "VB:PERFECTIVE:TEMPORAL_PAST"
            if any(p == "የ" or p in ("የተ", "ያልተ") for p in prefixes):
                return "VB:PERFECTIVE:RELATIVE"
            if pi_reading == "BOTH":
                return pos_tags.disjoin("VB:PERFECTIVE", "VB:JUSSIVE_IMPERATIVE")
            if pi_reading == "VB:JUSSIVE_IMPERATIVE":
                return "VB:JUSSIVE_IMPERATIVE"
            return "VB:PERFECTIVE"

        if licensed_shape == VerbalShape.SHAPE_B_COVERB:
            return "VB:COVERB:COMPOUND" if has_aux else "VB:COVERB"

        if licensed_shape == VerbalShape.SHAPE_C_IMPERFECTIVE:
            if any(p in ("እስከ", "እስክ", "እስኪ", "እስከሚ") for p in prefixes):
                return "VB:IMPERFECTIVE:TERMINATIVE"
            if any(p in ("እንድ", "እንዲ") for p in prefixes):
                return "VB:IMPERFECTIVE:PURPOSIVE"
            if any(p in ("ቢ", "ብ") for p in prefixes):
                return "VB:IMPERFECTIVE:CONDITIONAL"
            if any(p in ("ሲ", "ስ") for p in prefixes):
                return "VB:IMPERFECTIVE:TEMPORAL"
            if any(p in ("ሊ", "ል") for p in prefixes):
                return "VB:IMPERFECTIVE:PROSPECTIVE"
            if any(
                p in self.morphology.rel_imperfective_markers
                for p in prefixes
            ):
                return "VB:IMPERFECTIVE:RELATIVE"
            if has_aux:
                return "VB:IMPERFECTIVE:COMPOUND"

            # A causative stem is derived, and a derived stem normally has one
            # form for the contingent and one for the jussive, so the bare ይ- form
            # carries both readings: ያስገነዝር is "he makes realise" and "let him
            # make realise".  Measured over the corpus on this path, an overt አስ-
            # gives 22 BOTH against 7 imperfective-only (75.9%, against an 18.5%
            # base rate for plain actives) - a real lift, not unanimity, and the
            # seven counterexamples (አስኬደ, አስኼደ, አስጌጠ, አስመነዘረ, አስከለከለ and
            # friends) are causatives of already irregular bases.
            #
            # ተ- is deliberately excluded.  It is not visible in the imperfective
            # of most passives (ተመራ -> ይመራ), and where it does survive the
            # contingent and jussive differ more often than not: 2 BOTH against
            # 5 imperfective-only.  Keying on it would lose rows.
            if any(p == "አስ" for p in prefixes) and not suffixes:
                return pos_tags.disjoin("VB:IMPERFECTIVE", "VB:JUSSIVE_IMPERATIVE")

            pfx0 = prefixes[0] if prefixes else ""
            stem_orders = tuple(self.phonology.get_vocalic_orders(norm_s))
            MIN_LEX_FREQ = 3

            # The two-character paradigm registry records stems from the simple
            # contingent column (for example, ይ- + ልቅ -> ይልቅ). Honor that
            # direct evidence before a frequent unrelated homograph can make the
            # stem look like a jussive.
            if (
                prefixes == ("ይ",)
                and not suffixes
                and self.morphology.is_licensed_2char_contingent_stem(norm_s)
            ):
                return "VB:IMPERFECTIVE"

            # 1. Causative stems under 'ያ-' (peeled as 'ይ-' + 'አ...') ending in Sadis (5) without auxiliary:
            # Simple contingent and jussive are the EXACT SAME string (homographs) -> dual tag
            if pfx0 == "ይ" and norm_s.startswith("አ") and len(norm_s) >= 3:
                _, coda_ord = self.phonology.get_char_info(norm_s[-1])
                if coda_ord == 5:
                    return pos_tags.disjoin("VB:IMPERFECTIVE", "VB:JUSSIVE_IMPERATIVE")

            # 2. Passive & Reciprocal stems (e.g. ይቀበል [0, 0, 5], ይመረመር [0, 0, 0, 5], ይሰብሰብ [0, 5, 0, 5]):
            # Passive contingent and passive jussive are 100% homophonous in script!
            if stem_orders in getattr(self.morphology, "PASSIVE_JUSSIVE_ROOT_PATTERNS", set()):
                return pos_tags.disjoin("VB:IMPERFECTIVE", "VB:JUSSIVE_IMPERATIVE")

            # 2. T-initial root imperatives mis-peeled as prefix 'ት-' (ትከል, ትረፍ, ትረክ from ተከለ, ተረፈ, ተረከ):
            if prefixes == ("ት",) and tuple(self.phonology.get_vocalic_orders("ት" + norm_s)) == (5, 0, 5):
                return "VB:JUSSIVE_IMPERATIVE"

            # The defective/hollow/geminate jussive probes below name bare
            # subject-concord command forms (ይስጥ, ይሁን, ይበል …).  A visible ተ-/ታ-
            # voice prefix changes the analysis: ይተልም is the ተ- passive of ለመመ,
            # not a defective jussive, even though the accidental
            # contingent ይለም is frequent - so under those prefixes the probes
            # stay silent and the shape policy decides.  Subject/negation
            # prefixes (አን-, እን-, …) are not voice prefixes and keep the probes:
            # አንግት is the regular defective jussive of ገተተ.
            has_voice_pfx = any(p in ("ተ", "ታ", "ተን") for p in prefixes)

            # 3. Hollow-A Jussives with stem [3, 5] under ይ- (ይላክ, ይላስ, ይሳብ, ይሳም, ይማር, ይቻል):
            # Their contingent is [5, 5] (ይልክ, ይልስ, ይስብ). If contingent exists, this form is strictly Jussive.
            if not has_voice_pfx and len(norm_s) == 2 and stem_orders == (3, 5):
                cont_alt = (pfx0 or "ይ") + self.phonology.shift_order(norm_s[0], 5) + norm_s[1]
                if (
                    self.vocab_freq.get(self._norm(cont_alt), 0) >= MIN_LEX_FREQ
                    or norm_s[0] + self.phonology.shift_order(norm_s[1], 0) in self.morphology.biradical_type_a_medial_a_verbs
                ):
                    return "VB:JUSSIVE_IMPERATIVE"

            # 4. Defective Jussives with stem [5, 5] under ይ- (ይስጥ, ይሽጥ, ይቅር, ይምሽ, ይሽሽ):
            # Their contingent is [0, 5] (ይሰጥ, ይሸጥ, ይቀር, ይመሽ, ይሸሽ). If contingent exists, this form is strictly Jussive.
            if not has_voice_pfx and len(norm_s) == 2 and stem_orders == (5, 5):
                cont_alt = (pfx0 or "ይ") + self.phonology.shift_order(norm_s[0], 0) + norm_s[1]
                if (
                    self.vocab_freq.get(self._norm(cont_alt), 0) >= MIN_LEX_FREQ
                    or norm_s in ("ስጥ", "ሽጥ", "ቅር", "ምሽ", "ሽሽ", "ፍጅ")
                ):
                    return "VB:JUSSIVE_IMPERATIVE"

            # 5. Hollow-U/I Jussives ([1, 5] ይሁን/ይሙት/ይሩጥ/ይቁም and [2, 5] ይሂድ/ይሺጥ):
            # Their contingent is [6, 5] (ይሆን) or [4, 5] (ይሄድ). Stems [1, 5] and [2, 5] are strictly Jussive.
            if not has_voice_pfx and len(norm_s) == 2 and stem_orders in ((1, 5), (2, 5)):
                return "VB:JUSSIVE_IMPERATIVE"

            # 6. Geminate Contracted Jussives with stem [0, 5] without auxiliary:
            # e.g. ይበል (አለ), ይለቅ (ለቀቀ), ይመጥ (መጠጠ), ይመር (መረረ), ይረድ (ረደደ), ይቀፍ (ቀፈፈ)
            # In imperfective these geminates take auxiliary -አል (ይመርራል, ይመጥጣል); bare forms under ይ- are Jussive.
            if not has_voice_pfx and len(norm_s) == 2 and stem_orders == (0, 5) and norm_s in (
                "በል", "ለቅ", "ለብ", "ለፍ", "መም", "መር", "መን", "መጥ",
                "ረም", "ረስ", "ረድ", "ረፍ", "ሰር", "ቀፍ", "ንጭ", "ረክ"
            ):
                return "VB:JUSSIVE_IMPERATIVE"

            # Syncretism probe: attested syncretic shapes, passive patterns,
            # lexeme whitelists and lexical-alternant checks.  This is direct
            # lexical evidence for the stem, so it outranks the statistical
            # shape policy below: ይዘግይ/ይጎብኝ are genuinely syncretic (0,5,5)
            # even though the shape majority says imperfective-only.
            if self._is_genuinely_syncretic(pfx0, norm_s, stem_orders):
                if not prefixes:
                    return "VB:JUSSIVE_IMPERATIVE"
                if any(p == "ት" for p in prefixes) and any(s in ("ኢ", "ኡ", "አችሁ", "አቹ") for s in suffixes):
                    return "VB:IMPERFECTIVE"
                return pos_tags.disjoin("VB:IMPERFECTIVE", "VB:JUSSIVE_IMPERATIVE")

            # Corpus-derived shape policy (tools/build_contingent_jussive_policy.py).
            # Amharic does not write gemination, so Type A (contingent and jussive
            # spelled differently) versus Type B/C (identical) is lexical; the
            # stem's vocalic-order shape is the strongest available proxy and is
            # validated on held-out lexemes.  It runs after the lexical probes
            # above, which check each stem's attested contingent alternation and
            # are strictly more specific than the shape majority: the policy says
            # (5,5) is imperfective, but ይስጥ/ይሽጥ are the attested jussives of
            # ሰጠ/ሸጠ and rule 4 settles them before the majority gets a vote.
            # (ል- already returned VB:IMPERFECTIVE:PROSPECTIVE above.)
            if any(p == "እ" for p in prefixes):
                return "VB:IMPERFECTIVE"
            if any(p == "ት" for p in prefixes) and any(s in ("ኢ", "ኡ", "አችሁ", "አቹ") for s in suffixes):
                return "VB:IMPERFECTIVE"

            corpus_reading = self._corpus_contingent_jussive_reading(norm_s)
            if corpus_reading == "BOTH":
                return pos_tags.disjoin("VB:IMPERFECTIVE", "VB:JUSSIVE_IMPERATIVE")
            if corpus_reading == "VB:JUSSIVE_IMPERATIVE":
                return "VB:JUSSIVE_IMPERATIVE"
            if corpus_reading == "VB:IMPERFECTIVE":
                return "VB:IMPERFECTIVE"

            return "VB:IMPERFECTIVE"

        if licensed_shape == VerbalShape.SHAPE_D_JUSSIVE_IMPERATIVE:
            if pi_reading == "BOTH":
                return pos_tags.disjoin("VB:PERFECTIVE", "VB:JUSSIVE_IMPERATIVE")
            if pi_reading == "VB:PERFECTIVE":
                return "VB:PERFECTIVE"
            if self._experiencer_perfective(stem, prefixes, suffixes):
                return "VB:PERFECTIVE"
            return "VB:JUSSIVE_IMPERATIVE"

        fine_pos = self.morphology.inspect_full_radical_template(stem, prefixes, suffixes)
        if fine_pos == "COVERB":
            return "VB:COVERB:COMPOUND" if has_aux else "VB:COVERB"
        if fine_pos == "IMPERFECTIVE":
            return "VB:IMPERFECTIVE:COMPOUND" if has_aux else "VB:IMPERFECTIVE"
        if fine_pos == "JUSSIVE_IMPERATIVE":
            return "VB:JUSSIVE_IMPERATIVE"
        if fine_pos == "PERFECTIVE":
            return "VB:PERFECTIVE"
        # No licensed shape and no template match: a two-syllable experiencer stem
        # such as ጠማ in ጠማው lands here, because the object clitic is enough to push
        # the shape classifier off the SHAPE_D branch.  The same lexical evidence
        # applies as on that branch, so decide it here too rather than surrendering
        # a bare verbal tag.
        if self._experiencer_perfective(stem, prefixes, suffixes):
            return "VB:PERFECTIVE"
        return "VB:PERFECTIVE" if not prefixes and not suffixes else pos_tags.VERBAL

    # ------------------------------------------------------------------
    # Unified 2-Layer Classification API
    # ------------------------------------------------------------------

    def classify_stem(
        self,
        stem: str,
        prefixes: Tuple[str, ...] = (),
        suffixes: Tuple[str, ...] = (),
        candidate: Optional[Any] = None,
        exclude_word: Optional[str] = None,
    ) -> str:
        if not stem:
            return pos_tags.noun("CN")

        pfx_tuple = candidate.prefixes if candidate and candidate.prefixes else prefixes
        sfx_tuple = candidate.suffixes if candidate and candidate.suffixes else suffixes

        # An auxiliary suffix is the signal that the substrate is an
        # imperfective base rather than a lexical item.  If generation left the
        # inflectional prefix inside the stem (ያስጌጥ for ያስጌጣል = ያ + ስጌጥ + አል)
        # the verbal test below would see only a non-verbal-looking string and
        # hand the word to the nominal layer.  Probe the arbitration split here
        # read-only -- subclassify_verbal_stem adopts it once, for the shape.
        fused = None
        has_coverb_only = any(
            s in self.morphology.suffix_verb_coverb_only for s in sfx_tuple
        )
        has_auxiliary = any(
            s in self.morphology.suffix_verb_auxiliary for s in sfx_tuple
        )
        # A bare ይ-form (ይሞት, ይጾም) carries its subject prefix inside the stem and
        # has no affix for the gate below to key on, so it would never reach the
        # verbal classifier.  Probe the fused split there as well.
        if (has_auxiliary and not has_coverb_only) or (
            not pfx_tuple and not sfx_tuple and len(stem) >= 3
        ):
            fused = self._resolve_fused_inflectional_prefix(stem, pfx_tuple, sfx_tuple)

        all_subord_prefixes = {
            "ከ", "እንደ", "ስለ", "በ", "ል", "ሊ", "ስ", "ሲ",
            "ብ", "ቢ", "ክ", "ኪ", "እንድ", "እንዲ", "እስክ", "እስኪ", "እስከ"
        }
        has_v_pfx = any(
            p in getattr(self.morphology, "prefix_verb_iv", ("ይ", "ት", "እ", "እን"))
            or p in getattr(self.morphology, "prefix_verb_voice", ("ተ", "አስ", "አስተ"))
            or p in getattr(self.morphology, "rel_imperfective_prefixes", ())
            or p in all_subord_prefixes
            for p in pfx_tuple
        )
        has_v_sfx = any(
            s in ("አል", "ዋል", "ኦ", "ቼ", "ተው", "ቶ", "ታ") 
            or s in getattr(self.morphology, "suffix_verb_auxiliary", ())
            or s in getattr(self.morphology, "suffix_verb_exclusive_subject", ())
            for s in sfx_tuple
        )
        has_rel_pfx = any(p == "የ" or p == "የተ" or p.startswith("የ") for p in pfx_tuple)

        # ``የሚ-`` (realised here as the relativiser ``የ`` plus the imperfective
        # particle ``ሚ``/``ምት``/``ምን``/``ምታ``) is an *exclusive* relative-imperfective
        # marker: no nominal or infinitive reading can carry it.  Routing these
        # candidates through Layer 1 lets the verbal-noun heuristic win whenever the
        # residual stem happens to begin with ``መ-`` (የሚመለስ -> "VN"), or lets a
        # lemma-inversion artefact be read as a perfective relative.  Decide here.
        if self._has_relative_imperfective_prefix(pfx_tuple):
            if any(s in getattr(self.morphology, "suffix_verb_auxiliary", ()) for s in sfx_tuple):
                return "VB:IMPERFECTIVE:COMPOUND"
            return "VB:IMPERFECTIVE:RELATIVE"

        is_verbal = (
            self.is_known_verbal_stem(stem)
            or self.morphology.is_valid_perfective_template(stem)
            or self.morphology.is_valid_imperfective_stem(stem)
            # A licensed split means the affix-concord lock already accepted the
            # substrate as an imperfective base, so it is verbal by evidence.
            or fused is not None
        )

        if (has_v_pfx or has_v_sfx or has_rel_pfx or fused is not None) and is_verbal:
            return self.subclassify_verbal_stem(stem, prefixes=pfx_tuple, suffixes=sfx_tuple)

        # Structured Layer 1 Handoff
        l1 = self.identify_layer1(stem, candidate=candidate, exclude_word=exclude_word)

        if "VN" in pos_tags.disjuncts(l1.macro_tag):
            if l1.macro_tag == "VN" and (has_v_pfx or has_v_sfx):
                return self.subclassify_verbal_stem(stem, prefixes=pfx_tuple, suffixes=sfx_tuple)
            return l1.macro_tag

        # Only apply double-lock fast path / cross-family shield when Layer 1 has an explicit v_lock subtag.
        # A dual tag is returned whole: both readings must survive into the output.
        if l1.is_verbal and not l1.is_nv and l1.verbal_subtag:
            core_tag = l1.verbal_subtag
            if not pfx_tuple and not sfx_tuple and core_tag in (
                "COMPOUND_PERFECT", "COMPOUND_IMPERFECT", "RELATIVE_PERFECTIVE", "RELATIVE_IMPERFECTIVE", "COVERB"
            ):
                _MAP = {
                    "COVERB": "VB:COVERB",
                    "COMPOUND_PERFECT": "VB:COVERB:COMPOUND",
                    "COMPOUND_IMPERFECT": "VB:IMPERFECTIVE:COMPOUND",
                    "RELATIVE_PERFECTIVE": "VB:PERFECTIVE:RELATIVE",
                    "RELATIVE_IMPERFECTIVE": "VB:IMPERFECTIVE:RELATIVE"
                }
                return _MAP.get(core_tag, f"VB:{core_tag}")

            layer2_pos = self.subclassify_verbal_stem(stem, prefixes=pfx_tuple, suffixes=sfx_tuple)

            if "JUSSIVE" in layer2_pos and "PERFECTIVE" in layer2_pos:
                return layer2_pos
            if "JUSSIVE" in layer2_pos and "IMPERFECTIVE" in layer2_pos:
                return layer2_pos

            if {core_tag, layer2_pos} == {"PERFECTIVE", "VB:JUSSIVE_IMPERATIVE"} or {
                core_tag, layer2_pos
            } == {"JUSSIVE_IMPERATIVE", "VB:PERFECTIVE"}:
                # Layer 1 and Layer 2 disagree on the perfective/imperative
                # homograph.  The corpus-derived shape policy is the better-
                # validated arbiter (94.5% held-out over unseen lexemes versus a
                # coin flip between the two hand-written locks), so follow it when
                # it covers this surface and keep the Layer 1 lock otherwise.
                # The check is symmetric: ይዝማ locks jussive in Layer 1 while the
                # subclassifier says perfective, and its lock is the one to keep
                # when the policy does not cover the surface.
                tie_surface = "".join(pfx_tuple) + stem + "".join(sfx_tuple)
                tie_reading = self.morphology.perfective_imperative_reading(tie_surface)
                if tie_reading in ("VB:PERFECTIVE", "VB:JUSSIVE_IMPERATIVE"):
                    return tie_reading
                return "VB:PERFECTIVE" if core_tag == "PERFECTIVE" else "VB:JUSSIVE_IMPERATIVE"

            if layer2_pos == "VB":
                return f"VB:{core_tag}"

            return layer2_pos

        if l1.is_nv:
            # Affix context disambiguates a homograph: a plural, definite,
            # possessive or derivational suffix (ስራዎችን ስራ + ዎች + ን, መርዙ መርዝ + ኡ)
            # is nominal-only, so the verbal reading of the stem must not be
            # carried onto the inflected form.  Conversely a verbal suffix keeps
            # the dual, which the Layer-1 evidence already resolved.
            if self._has_exclusive_nominal_suffix_context(sfx_tuple):
                return self.subclassify_nominal_stem(stem, candidate=candidate)
            return l1.macro_tag

        if l1.is_unknown:
            return pos_tags.UNKNOWN

        if l1.is_nominal:
            return self.subclassify_nominal_stem(stem, candidate=candidate)

        if l1.is_verbal:
            return self.subclassify_verbal_stem(stem, prefixes=prefixes, suffixes=suffixes)

        # Remaining Layer 1 tags are the orthogonal families (ADP, ADV, CONJ,
        # INT, VN).  They are neither nominal nor verbal, so they pass through
        # unchanged instead of collapsing to a common noun.
        if pos_tags.disjuncts(l1.macro_tag):
            return l1.macro_tag

        return pos_tags.noun("CN")

    # ------------------------------------------------------------------
    # Search Engine Predicates
    # ------------------------------------------------------------------

    def is_known_nominal_stem(self, stem: str) -> bool:
        """True when the stem has a known non-verbal lexical reading.

        Despite the historical name this doubles as the attestation gate for
        the engineered orthogonal families (ADP/ADV/CONJ/INT), verbal nouns and
        ideophones, so those are accepted here; AUX is verbal-only.
        """
        l1 = self.identify_layer1(stem)
        return (
            l1.is_nominal
            or l1.is_nv
            or l1.macro_tag == pos_tags.ideophone()
            or "VN" in pos_tags.disjuncts(l1.macro_tag)
            or (
                pos_tags.is_closed(l1.macro_tag)
                and pos_tags.closed_subtype(l1.macro_tag) != "AUX"
            )
        )

    def is_known_verbal_stem(self, stem: str, exclude_word: Optional[str] = None) -> bool:
        l1 = self.identify_layer1(stem, exclude_word=exclude_word)
        return l1.is_verbal or l1.is_nv

    def is_known_stem(self, stem: str) -> bool:
        if not self.vocab_freq:
            return True
        return self.is_known_nominal_stem(stem) or self.is_known_verbal_stem(stem)