"""
affix_peeler.py - Centralized Morphological Deconstruction Engine.
Responsible strictly for peeling grammatical affixes (proclitics, subordinators,
relatives, voice prefixes, enclitics, auxiliaries, objects, agreement morphemes)
under strict morphotactic strata and attestation firewalls.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from morph_types import DecomposeContext, MorphSplit
from phonology import AmharicBoundaryPhonology


class AffixPeeler:
    STRONG = 3
    NORMAL = 2
    WEAK = 1

    # Concentric Derivational Voice Morphemes
    # "አን"/"አም" are the causative counterparts of the middle-voice "ተን"/"ተም"
    # prefixes used on reduplicated/intensive stems (e.g. ተንቀሳቀሰ "moved" ->
    # አንቀሳቀሰ "moved (something)"). They must stay in sync with
    # morphemes.json's "prefix_verb_voice_ambiguous" / "derived_voice_prefixes".
    CAUSATIVE_PREFIXES: Tuple[str, ...] = ("አስ", "አስተ", "አስታ", "አን", "አም")
    NON_ACTIVE_PREFIXES: Tuple[str, ...] = ("ተ", "ተን", "ታ", "ተስተ", "ተም")
    ALL_VOICE_TARGETS: Tuple[str, ...] = CAUSATIVE_PREFIXES + NON_ACTIVE_PREFIXES

    def __init__(
        self,
        morphology: Any,
        phonology: Optional[AmharicBoundaryPhonology] = None,
    ):
        self.morphology = morphology
        self.phonology = phonology or AmharicBoundaryPhonology()

    # ------------------------------------------------------------------
    # Concentric Voice & Root Extraction Helper
    # ------------------------------------------------------------------
    def extract_voice_and_root(
        self, prefixes: Tuple[str, ...], stem: str
    ) -> Tuple[Optional[str], str]:
        """
        Extracts the active voice prefix and remnant radical root.
        Handles both decomposed prefixes and intact surface stems.
        """
        # 1. Voice prefix was already decomposed into prefixes
        for p in prefixes:
            if p in self.ALL_VOICE_TARGETS:
                return p, stem

        # 2. Voice prefix is still attached to the stem
        for vp in self.ALL_VOICE_TARGETS:
            if stem.startswith(vp) and len(stem) > len(vp) + 1:
                return vp, stem[len(vp):]

        return None, stem

    # ------------------------------------------------------------------
    # Master Decomposition Dispatcher
    # ------------------------------------------------------------------
    def decompose(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        """Runs the concentric decomposition strata over the active search state."""
        if not word or len(word) < 2 or self.morphology.is_reduplicative_quadriradical(word):
            return []

        results: List[MorphSplit] = []

        # Strata 1: High-Strength Boundary Phonology & Agglutinative Clitics
        results.extend(self._peel_applicatives(word, ctx))
        results.extend(self._peel_prefix_coalescence(word, ctx))
        results.extend(self._peel_relative_imperfective_prefix(word, ctx))
        results.extend(self._peel_verb_plural_u(word, ctx))
        results.extend(self._peel_possessives(word, ctx))
        results.extend(self._peel_instrument_nouns(word, ctx))
        results.extend(self._peel_plurals(word, ctx))
        results.extend(self._peel_definiteness(word, ctx))
        results.extend(self._peel_adjectivizers(word, ctx))

        # Strata 2: Verbal Person Agreement & Modals
        results.extend(self._peel_perfective_subjects(word, ctx))
        results.extend(self._peel_imperfective_subjects(word, ctx))
        results.extend(self._peel_bare_palatal_imperatives(word, ctx))
        results.extend(self._peel_abstract_derivational(word, ctx))
        results.extend(self._peel_auxiliaries(word, ctx))
        results.extend(self._peel_negative_finiteness(word, ctx))
        results.extend(self._peel_verb_object_clitics(word, ctx))
        results.extend(self._peel_coverbs(word, ctx))

        # Strata 3: Proclitic & Enclitic Fallbacks (Protected by Firewalls)
        results.extend(self._peel_literal_clitics_and_prefixes(word, ctx))
        results.extend(self._peel_raw_suffix_fallback(word, ctx))

        return results

    # ------------------------------------------------------------------
    # Strata Handlers
    # ------------------------------------------------------------------
    def _peel_applicatives(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        out = []
        if self.morphology.is_valid_infinitive_stem(word):
            return out

        for cluster, bipartite_affixes in self.morphology.bipartite_applicative_mappings.items():
            if word.endswith(cluster) and len(word) > len(cluster) + 1:
                stem = word[:-len(cluster)]
                role = "benefactive" if cluster.startswith("ል") else "malfactive"
                out.append(
                    MorphSplit(
                        stem=stem,
                        prefixes=(),
                        suffixes=tuple(bipartite_affixes),
                        rule=f"applicative_bipartite_{role}",
                        strength=self.STRONG,
                    )
                )
        return out

    def _is_supported_verbal_stem(
        self, stem: str, checker: Optional[Callable[[str], bool]] = None
    ) -> bool:
        return (
            self.morphology.is_valid_imperfective_stem(stem)
            or self.morphology.is_valid_jussive_stem(stem)
            or self.morphology.is_attested_verbal_stem(stem, checker)
        )

    def _peel_relative_imperfective_prefix(
        self, word: str, ctx: DecomposeContext
    ) -> List[MorphSplit]:
        """Peel relative imperfective ``የሚ-`` atomically, not as bare ``የ-``.

        The subordinate licensors that fuse with the same ሚ- allomorph
        (``እንደሚ-``, ``ስለሚ-``, ``በሚ-``, ``ከሚ-``, ``ለሚ-``) are peeled here as well,
        so the verb stem is recovered instead of being left inside an opaque
        unsegmented word (እንደሚችል -> እንደ+እም+ኢ+ችል).
        """
        if not word or len(word) <= 2:
            return []

        checker = ctx.known_verbal_stem_checker or ctx.known_stem_checker
        out: List[MorphSplit] = []

        if word.startswith("የሚ"):
            stem = word[2:]
            if self._is_supported_verbal_stem(stem, checker):
                out.append(
                    MorphSplit(
                        stem=stem,
                        prefixes=("የ", "ሚ"),
                        suffixes=(),
                        rule="rel_imp_ye_mi",
                        strength=self.STRONG,
                    )
                )

        for surface in sorted(
            getattr(self.morphology, "prefix_verb_subordinate_exclusive", ()),
            key=len,
            reverse=True,
        ):
            if not surface.endswith("ሚ") or not word.startswith(surface):
                continue
            stem = word[len(surface):]
            if len(stem) < 2 or not self._is_supported_verbal_stem(stem, checker):
                continue
            out.append(
                MorphSplit(
                    stem=stem,
                    prefixes=(surface[:-1], "እም", "ኢ"),
                    suffixes=(),
                    rule=f"rel_imp_{surface}",
                    strength=self.STRONG,
                )
            )

        return out

    def _has_licensed_coverb_surface(
        self, word: str, checker: Optional[Callable[[str], bool]] = None
    ) -> bool:
        """Whether a word has a rule-derived, licensed coverb analysis."""
        if not word:
            return False
        if self.morphology.is_valid_coverb_stem(word):
            return True

        for coverb_stem, _suffix, _rule in self.phonology.invert_coverb_3ms_o(word):
            if (
                self.morphology.is_attested_verbal_stem(coverb_stem, checker)
                or self.morphology.is_valid_coverb_stem(coverb_stem)
            ):
                return True

        for coverb_stem, _suffix, _rule in self.phonology.invert_coverb_3pl_w(word):
            if (
                self.morphology.is_attested_verbal_stem(coverb_stem, checker)
                or self.morphology.is_valid_coverb_stem(coverb_stem)
            ):
                return True

        for coverb_stem, _affixes, _rule in self.phonology.invert_labialized_coverb_auxiliary(word):
            if (
                self.morphology.is_attested_verbal_stem(coverb_stem, checker)
                or self.morphology.is_valid_coverb_stem(coverb_stem)
            ):
                return True

        return False

    def _protected_coda_finals(self) -> Set[str]:
        """Final characters whose peel may be over-segmentation.

        Derived from the shipped morpheme inventories, not from a literal list,
        so the shield cannot drift out of sync with the peels it guards.  Two
        groups qualify:

        * the pragmatic enclitics (``suffix_universal_clitics_*``);
        * single-character verbal subject suffixes -- minus any morpheme that
          also plays a case or object role, because the accusative/possessive
          ``ን``/``ሽ``/``ኡ`` legitimately peel off nouns and objects (ልጁን) and
          must keep doing so.
        """
        morphology = self.morphology

        def _inventory(*names: str) -> Set[str]:
            found: Set[str] = set()
            for name in names:
                found |= set(getattr(morphology, name, set()) or ())
            return found

        case_or_object_roles = (
            "suffix_nominal_",
            "suffix_accusative",
            "suffix_verb_object",
            "suffix_verb_coverb",
            "suffix_verb_imperfective_relative",
        )
        nominal_markers = {
            suffix
            for name in dir(morphology)
            if name.startswith(case_or_object_roles)
            for suffix in (getattr(morphology, name, set()) or ())
        }
        subject_suffixes = _inventory(
            "suffix_verb_subject_exclusive", "suffix_verb_subject_ambiguous"
        )
        finals = {
            suffix
            for suffix in _inventory(
                "suffix_universal_clitics_ambiguous",
                "suffix_universal_clitics_exclusive",
            )
            if len(suffix) == 1
        }
        finals |= {
            suffix
            for suffix in subject_suffixes
            if len(suffix) == 1 and suffix not in nominal_markers
        }
        return finals

    def _has_protected_verbal_coda(
        self, word: str, checker: Optional[Callable[[str], bool]] = None
    ) -> bool:
        """Keep lexical final ም/ስ/ማ/ና from being mistaken for an enclitic.

        The clitic table keys on the *surface* final character, so the vocalised
        allomorphs (ማ for ም, ና for ን) peel exactly like the bare consonants.  The
        shield therefore tests the word's final character against that table: a
        word that is itself a supported verb form keeps its final segment
        (ይዝማ stays whole, ይዝ+ማ does not outrank it).

        A closed-class host is the exception: enclitics attach to function words
        as a matter of course (ነውና "is it?", አለም), so those keep peeling even
        though the whole string also passes a verbal template check.
        """
        if not word or word[-1] not in self._protected_coda_finals():
            return False

        for clitic in self.morphology.clitics_by_final.get(word[-1], []):
            if word.endswith(clitic) and len(word) > len(clitic):
                host_word = word[: -len(clitic)]
                if self.morphology.get_closed_class_pos(host_word) is not None:
                    return False

        host = word[2:] if word.startswith("የሚ") else word
        return self._is_supported_verbal_stem(host, checker)

    def _peel_prefix_coalescence(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        """
        Unified Rabe Onset Inverter.
        Deterministically handles all 4th-order (Rabe / -a/) surface onsets:
        1. Privative Infinitives (ላለ-መ..., ባለ-መ..., ካለ-መ..., ወዳለ-መ...)
        2. Verb & Pronoun Coalescence (ያ-, ባ-, ላ-, ካ-, ታ-, ና-, ሳ-, እንዳ-, ስላ-, ወዳ-, እስካ-)
        3. Deterministic shape-routed resolution of ያ- (Relative የ- vs. Concord ይ-)
        """
        if not word or len(word) < 2:
            return []

        out: List[MorphSplit] = []

        # ------------------------------------------------------------------
        # 1. Lexical Shield: Autonomous Type C Roots, Commands & Rabe Nouns
        # (ባረከ, ባረቀ, ባለገ, ባርክ, ባልግ, ማረከ, ዳነ, ዋለ, ላላ, ሣሣ, ዳቦ, ሻይ, ላም, ባት)
        # 2-character contingents under ያ- (ያይ, ያጭ, ያሽ, ያዥ, ያዝ) are coalesced (ይ+አ-), not unpeeled roots.
        is_coalesced_ya = (word.startswith("ያ") and len(word) == 2 and word != "ያዘ")
        if not is_coalesced_ya and (
            word in getattr(self.morphology, "type_c_shield_words", set())
            or word in getattr(self.morphology, "confirmed_sparse_verbs", set())
            or word in getattr(self.morphology, "confirmed_sparse_nouns", set())
            or word in self.morphology.verb_stems
            or self.morphology.is_licensed_defective_root(word)
        ):
            return []

        # ------------------------------------------------------------------
        # 2. Paradigm 1: Adposition + Privative Infinitive (Coalesced አለ-)
        # e.g. ላለ-መሆን (ለ+አለ+መሆን), ባለ-መስማት (በ+አለ+መስማት), ወዳለ-መሄድ
        # ------------------------------------------------------------------
        privative_vn_prefix_map: Tuple[Tuple[str, Tuple[str, ...]], ...] = tuple(
             getattr(self.morphology, "privative_vn_prefix_map", {}).items()
        )
        for surf_pfx, pfx_components in privative_vn_prefix_map:
            if word.startswith(surf_pfx) and len(word) > len(surf_pfx) + 2:
                rem_stem = word[len(surf_pfx):]
                # Privative prefix አለ- strictly attaches to an Infinitive (መ-/ማ-)
                if rem_stem.startswith(("መ", "ማ")):
                    out.append(MorphSplit(
                        stem=rem_stem,
                        prefixes=pfx_components,
                        suffixes=(),
                        rule=f"pfx_privative_vn_{surf_pfx}",
                        strength=self.STRONG,
                    ))
                    # High-confidence morphotactic match; no need to test weaker partial splits
                    return out

        # ------------------------------------------------------------------
        # 3. Glide Epenthesis Coalescence (እንደሚያ-, ስለሚያ-, የሚያ-, እንዲያ-)
        # ------------------------------------------------------------------
        for surf_pfx, mappings in self.phonology.GLIDE_COALESCED_PREFIXES:
            if word.startswith(surf_pfx) and len(word) > len(surf_pfx):
                restored_stem = "አ" + word[len(surf_pfx):]
                for pfx_tuple, rule_name in mappings:
                    out.append(MorphSplit(
                        stem=restored_stem,
                        prefixes=pfx_tuple,
                        suffixes=(),
                        rule=rule_name,
                        strength=self.STRONG,
                    ))

        # ------------------------------------------------------------------
        # 4. Multi-Character 4th-Order Prepositions & Negation
        # ------------------------------------------------------------------
        MULTI_RABE_PREFIXES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
            ("እስካ", ("እስከ",), "pfx_prep_eska_coalescence"),
            ("እንዳ", ("እንደ",), "pfx_prep_enda_coalescence"),
            ("ስላ",   ("ስለ",),   "pfx_prep_sla_coalescence"),
            ("ወዳ",   ("ወደ",),   "pfx_prep_weda_coalescence"),
            ("እያ",   ("እየ",),   "pfx_progressive_eyye_coalescence"),
            ("አላ",   ("አል",),   "pfx_neg_atomic_al"),
            ("አያ",   ("አይ",),   "pfx_neg_imp_ay_a"),
            ("ሳያ",   ("ሳይ",),   "pfx_neg_subord_saya"),
            ("ያልተ", ("የ", "አል", "ተ"), "pfx_rel_neg_passive"),
            ("ያል",  ("የ", "አል"), "pfx_rel_neg_ye_al"),
        )
        for surf_pfx, pfx_tuple, rule_name in MULTI_RABE_PREFIXES:
            if word.startswith(surf_pfx) and len(word) > len(surf_pfx) + 1:
                # Relative negative 'ያል-' attaches only to past verbs (ending in vowel 0/3).
                # Words ending in Sadis (5) (ያልቅም, ያልብስ, ያልምጥ) are causative jussives (ይ+አል-), NOT relative negative.
                _, w_coda = self.phonology.get_char_info(word[-1])
                if surf_pfx == "ያል" and w_coda == 5:
                    continue

                restored_stem = "አ" + word[len(surf_pfx):]
                out.append(MorphSplit(
                    stem=restored_stem,
                    prefixes=pfx_tuple,
                    suffixes=(),
                    rule=rule_name,
                    strength=self.STRONG,
                ))

        # ------------------------------------------------------------------
        # 5. Causative & Concord Combinations (ያስ-, ያሥ-, ያን-)
        # ------------------------------------------------------------------
        for surf_pfx, pfx_subj, pfx_rel, rule_base in (
            ("ያስ", ("ይ", "አስ"), ("የ", "አስ"), "pfx_coalescence_yas"),
            ("ያሥ", ("ይ", "አስ"), ("የ", "አስ"), "pfx_coalescence_yas"),
        ):
            if word.startswith(surf_pfx) and len(word) > len(surf_pfx) + 1:
                restored_root = word[len(surf_pfx):]
                # If remnant root is a command or imperfective stem (e.g. መስል, ደርግ) -> concord ይ-
                # If remnant root is past tense (e.g. መሰለ, ደረገ) -> relative የ-
                _, r_coda = self.phonology.get_char_info(restored_root[-1]) if restored_root else ("", -1)
                # Sadis (5) codas (መስል, ኬድ) and Rabe (3) final-weak codas (መጣ, በራ, ተኛ, ነሳ)
                # are authentic imperfective / jussive stems licensing subject concord ይ-
                if r_coda in (5, 3):
                    out.append(MorphSplit(stem=restored_root, prefixes=pfx_subj, suffixes=(), rule=f"{rule_base}_subj", strength=self.STRONG))
                else:
                    out.append(MorphSplit(stem=restored_root, prefixes=pfx_rel, suffixes=(), rule=f"{rule_base}_rel", strength=self.STRONG))

        # ------------------------------------------------------------------
        # 6. Single-Character 4th-Order Onsets (ያ-, ባ-, ላ-, ካ-, ታ-, ና-, ሳ-)
        # Deterministically routed by remnant stem vocalic tier
        # ------------------------------------------------------------------
        # If 'ያስ-' or 'ያሥ-' already peeled causative voice, do not create a competing unpeeled 'አስ-' stem
        if word.startswith(("ያስ", "ያሥ")) and len(word) >= 4:
            return out

        first_c = word[0]
        _, first_order = self.phonology.get_char_info(first_c)

        if first_order == 3 and len(word) >= 2:
            restored_stem = "አ" + word[1:]
            is_perf_stem = self.morphology.is_valid_perfective_template(restored_stem)
            is_imp_stem = self.morphology.is_valid_imperfective_stem(restored_stem)
            is_juss_stem = self.morphology.is_valid_jussive_stem(restored_stem)

            # === Prefix 'ያ-' (Relative የ- vs. Concord ይ-) ===
            if first_c == "ያ":
                orders_restored = tuple(self.phonology.get_vocalic_orders(restored_stem))
                _, last_order = self.phonology.get_char_info(word[-1])

                is_jussive_shape = (
                    is_juss_stem
                    or orders_restored in getattr(self.morphology, "LESLAU_JUSSIVE_PATTERNS", set())
                    or orders_restored in getattr(self.morphology, "BOUND_JUSSIVE_STEM_PATTERNS", set())
                )

                # 1. Coda Order 0 (Ge'ez): Past base (ያላላከ, ያራራቀ, ያሳሳተ, ያየ, ያጨ, ያሸ, ያዠ, ያደረገ) -> 100% strictly Relative የ-
                if last_order == 0 and is_perf_stem:
                    out.append(MorphSplit(stem=restored_stem, prefixes=("የ",), suffixes=(), rule="pfx_rel_ye_a_deterministic", strength=self.STRONG))

                # 2. Coda Order 5 (Sadis) or 3/7 (Weak/Causative) -> Subject Concord ይ- (ያደርግ, ያይ, ያጭ, ያምጣ, ያሰማ, ያጉላላ)
                elif last_order in (5, 3, 7) or is_jussive_shape or is_imp_stem:
                    out.append(MorphSplit(stem=restored_stem, prefixes=("ይ",), suffixes=(), rule="pfx_subj_yi_a_deterministic", strength=self.STRONG))

                elif is_perf_stem:
                    out.append(MorphSplit(stem=restored_stem, prefixes=("የ",), suffixes=(), rule="pfx_rel_ye_a_deterministic", strength=self.STRONG))
                else:
                    # Fallback for nouns / adjectives (e.g. ያገር, ያህያ, ያበበ) -> Genitive የ-
                    out.append(MorphSplit(stem=restored_stem, prefixes=("የ",), suffixes=(), rule="pfx_rel_ye_a_fallback", strength=self.NORMAL))

            # === Prefix 'ታ-' (Passive ተ- vs. Concord ት-) ===
            elif first_c == "ታ":
                _, last_order = self.phonology.get_char_info(word[-1])
                # When terminating in a labialized coverb auxiliary (-Cʷal e.g. ታስሯል, ታውቋል, ታጥቧል, ታይቷል),
                # the verb is strictly a Present Perfect where 'ታ-' is Passive 'ተ-', never subject concord 'ት-'.
                _, penult_ord = self.phonology.get_char_info(word[-2]) if len(word) >= 3 else ("", -1)
                is_compound_coverb_tail = (
                    (word.endswith("ል") and (penult_ord == 7 or word.endswith("ቷል") or word[-2] in self.phonology.LABIOVELAR_MAP))
                    or any(s in ("ቶ", "ኦ", "ተው", "ተን", "ታ", "ቼ") for s in ctx.suffixes)
                    or (any(s in ("አል", "ዋል") for s in ctx.suffixes) and not any(p in self.morphology.all_subject_prefixes for p in ctx.prefixes))
                )
                if is_compound_coverb_tail:
                    out.append(MorphSplit(stem=restored_stem, prefixes=("ተ",), suffixes=(), rule="pfx_voice_passive_te_deterministic", strength=self.STRONG))
                elif (last_order == 0 and is_perf_stem) or (len(word) == 2 and last_order == 3 and is_perf_stem):
                    # e.g. ታሰረ, ታወቀ (coda 0) or ታጣ, ታማ (2-char coda 3) -> strictly Passive ተ-
                    out.append(MorphSplit(stem=restored_stem, prefixes=("ተ",), suffixes=(), rule="pfx_voice_passive_te_deterministic", strength=self.STRONG))
                elif is_imp_stem or is_juss_stem:
                    # e.g. ታደርግ, ታድርግ, ታመጣ -> strictly Concord ት-
                    out.append(MorphSplit(stem=restored_stem, prefixes=("ት",), suffixes=(), rule="pfx_subj_t_a_deterministic", strength=self.STRONG))
                else:
                    out.append(MorphSplit(stem=restored_stem, prefixes=("ተ",), suffixes=(), rule="pfx_voice_passive_te_fallback", strength=self.NORMAL))

            # === Prefix 'ና-' (1PL Concord እን-) ===
            elif first_c == "ና":
                # Converbs ending in -Cʷal / -tʷal (e.g. ናፍቋል, ናኝቷል) retain radical 'ና', never subject concord 'እን-'.
                _, penult_ord = self.phonology.get_char_info(word[-2]) if len(word) >= 3 else ("", -1)
                is_compound_coverb_tail = (
                    word.endswith(("ቷል", "ቋል", "ኳል", "ጓል", "ኋል", "ሯል", "ሷል", "ሟል", "ኗል", "ፏል", "ጇል", "ጯል", "ጿል", "ዟል"))
                    or word.endswith(("ቷ", "ቋ", "ኳ", "ጓ", "ኋ", "ሯ", "ሷ", "ሟ", "ኗ", "ፏ", "ጇ", "ጯ", "ጿ", "ዟ"))
                    or any(s in ("አል", "ዋል") for s in ctx.suffixes)
                )
                if not is_compound_coverb_tail and (is_imp_stem or is_juss_stem or is_perf_stem):
                    # e.g. ናደርግ, ናድርግ -> Concord እን-
                    out.append(MorphSplit(stem=restored_stem, prefixes=("እን",), suffixes=(), rule="pfx_subj_en_a_deterministic", strength=self.STRONG))

            # === Prefix 'ባ-' (Preposition በ- vs. Subordinator ብ-) ===
            elif first_c == "ባ":
                word_orders = tuple(self.phonology.get_vocalic_orders(word))
                is_type_c_shape = word_orders in ((3, 0, 0), (3, 5, 5), (3, 0, 5))
                restored_is_known = (
                    restored_stem in self.morphology.verb_stems
                    or self.morphology.is_valid_perfective_template(restored_stem)
                    or self.morphology.is_valid_imperfective_stem(restored_stem)
                    or self.morphology.is_valid_jussive_stem(restored_stem)
                    or self.morphology.is_licensed_defective_root(restored_stem)
                    or (ctx.known_verbal_stem_checker and ctx.known_verbal_stem_checker(restored_stem))
                    or (ctx.known_stem_checker and ctx.known_stem_checker(restored_stem))
                )
                if not (is_type_c_shape and not restored_is_known):
                    out.append(MorphSplit(stem=restored_stem, prefixes=("በ",), suffixes=(), rule="pfx_prep_b_a_coalescence", strength=self.STRONG))
                    if is_imp_stem or is_perf_stem:
                        out.append(MorphSplit(stem=restored_stem, prefixes=("ብ",), suffixes=(), rule="pfx_subord_b_a_coalescence", strength=self.NORMAL))

            # === Prefix 'ላ-' (Preposition ለ- vs. Subordinator ል-) ===
            elif first_c == "ላ":
                checker = ctx.known_verbal_stem_checker or ctx.known_stem_checker
                if not self._has_licensed_coverb_surface(word, checker):
                    out.append(MorphSplit(stem=restored_stem, prefixes=("ለ",), suffixes=(), rule="pfx_prep_l_a_coalescence", strength=self.STRONG))
                    if is_imp_stem or is_juss_stem:
                        out.append(MorphSplit(stem=restored_stem, prefixes=("ል",), suffixes=(), rule="pfx_subord_l_a_coalescence", strength=self.NORMAL))

            # === Prefix 'ካ-' (Preposition ከ-) ===
            elif first_c == "ካ":
                out.append(MorphSplit(stem=restored_stem, prefixes=("ከ",), suffixes=(), rule="pfx_prep_k_a_coalescence", strength=self.STRONG))

            # === Prefix 'ሳ-' (Temporal Subordinator ስ-) ===
            elif first_c == "ሳ":
                out.append(MorphSplit(stem=restored_stem, prefixes=("ስ",), suffixes=(), rule="pfx_subord_s_a_coalescence", strength=self.STRONG))

        return out

    def _can_be_definite_or_possessive(
        self, stem: str, checker: Optional[Callable] = None
    ) -> bool:
        if not stem or len(stem) < 2:
            return False
        if self.morphology.is_abstract_noun_stem(stem):
            return False
        if checker and checker(stem):
            return False
        for ur in ("ው", "ቷ", "ዋ"):
            if stem.endswith(ur):
                inner_cand = stem[:-len(ur)]
                if checker and checker(inner_cand):
                    return True
        for inner_stem, _, _ in self.phonology.invert_via_49_cell_matrix(stem, "ኡ"):
            if inner_stem != stem and checker and checker(inner_stem):
                return True
        for ur_suffix, _ in self.morphology.possessive_urs:
            if ur_suffix in ("ኡ", "ዎ", "ዎት"):
                continue
            for inner_stem, _, _ in self.phonology.invert_via_49_cell_matrix(stem, ur_suffix):
                if inner_stem != stem and checker and checker(inner_stem):
                    return True
        return False

    def _peel_possessives(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        has_verbal_prefix_context = any(p in self.morphology.exclusive_verbal_prefixes or p == "የ" for p in ctx.prefixes)
        if ctx.has_exclusive_verb_sfx or ctx.has_poss_or_def_sfx or ctx.has_exclusive_verb_pfx or has_verbal_prefix_context:
            return []
        out = []
        checker = ctx.known_stem_checker
        if not word or len(word) < 3 or word in self.morphology.pronoun_stems:
            return out

        for compound_sfx, ur_list in self.morphology.compound_plural_possessives.items():
            if word.endswith(compound_sfx) and len(word) >= len(compound_sfx) + 2:
                stem = word[:-len(compound_sfx)]
                if checker and not checker(stem):
                    continue
                if self._can_be_definite_or_possessive(stem, checker):
                    continue
                out.append(MorphSplit(stem=stem, prefixes=(), suffixes=tuple(ur_list), rule="possessive_compound_plural", strength=self.STRONG))

        if word.endswith("ዎት") and len(word) >= 4:
            stem = word[:-2]
            _, last_ord = self.phonology.get_char_info(stem[-1])
            if last_ord in (0, 2, 3, 4, 6) and (checker is None or checker(stem)):
                if not self._can_be_definite_or_possessive(stem, checker):
                    out.append(MorphSplit(stem=stem, prefixes=(), suffixes=("ዎት",), rule="possessive_2POL_wot", strength=self.STRONG))

        elif word.endswith("ዎ") and len(word) >= 3:
            stem = word[:-1]
            if not (checker and checker(word)) and (checker is None or checker(stem)):
                if not self._can_be_definite_or_possessive(stem, checker):
                    out.append(MorphSplit(stem=stem, prefixes=(), suffixes=("ዎ",), rule="possessive_2POL_wo", strength=self.STRONG))

        if word.endswith("ዬ") and len(word) >= 3:
            stem = word[:-1]
            if not (checker and not checker(stem)) and not self._can_be_definite_or_possessive(stem, checker):
                out.append(MorphSplit(stem=stem, prefixes=(), suffixes=("ዬ",), rule="possessive_1S_vocalic", strength=self.STRONG))

        if word.endswith("ጇ") and len(word) >= 2:
            stem = word[:-1] + "ጅ"
            if not (checker and not checker(stem)) and not self._can_be_definite_or_possessive(stem, checker):
                out.append(MorphSplit(stem=stem, prefixes=(), suffixes=("ዋ",), rule="possessive_3FS_palatal", strength=self.STRONG))

        for ur_suffix, person in self.morphology.possessive_urs:
            if ur_suffix in ("ዎ", "ዎት"):
                continue
            if ur_suffix == "ኡ" and word in getattr(self.morphology, "inherent_terminal_u_stems", set()):
                continue

            for stem, ur, rule in self.phonology.invert_via_49_cell_matrix(word, ur_suffix):
                if ur_suffix in ("እህ", "እሽ"):
                    _, last_o = self.phonology.get_char_info(stem[-1])
                    if last_o == 1:
                        continue

                if (ur_suffix in ("እህ", "እሽ", "አችን", "አችሁ", "አቹ", "አቸው", "ኤ") and checker and not checker(stem)):
                    continue

                if self._can_be_definite_or_possessive(stem, checker):
                    continue

                if ur_suffix == "ኤ" and len(stem) >= 2:
                    _, s_ord = self.phonology.get_char_info(stem[-1])
                    if s_ord == 5 and stem[-1] in self.phonology.PALATAL_CONSONANTS:
                        if stem[-1] in self.phonology.PALATAL_TO_CORONAL:
                            if any(
                                self.morphology.is_valid_coverb_stem(stem[:-1] + cor)
                                for cor in self.phonology.PALATAL_TO_CORONAL[stem[-1]]
                            ):
                                continue

                out.append(MorphSplit(stem=stem, prefixes=(), suffixes=(ur,), rule=f"possessive_{person}", strength=self.STRONG))

        return out

    def _peel_instrument_nouns(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        out = []
        if ctx.known_stem_checker and ctx.known_stem_checker(word):
            return out

        for ur in self.morphology.instrument_noun_urs:
            for stem, ur_suffix, _ in self.phonology.invert_via_49_cell_matrix(word, ur):
                if stem and stem.startswith(("መ", "ማ")) and len(stem) >= 4:
                    out.append(MorphSplit(stem=stem, prefixes=(), suffixes=(ur_suffix,), rule="instrument_noun", strength=self.STRONG))

        for stem, sfx, rule in self.phonology.invert_palatalization(word):
            if rule == "palatal_instrumental_noun":
                out.append(MorphSplit(stem=stem, prefixes=(), suffixes=(sfx,), rule=rule, strength=self.STRONG))

        return out

    def _peel_verb_plural_u(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        out = []
        if word in getattr(self.morphology, "inherent_terminal_u_stems", set()):
            return out
        for stem, sfx, rule in self.phonology.invert_rule_5_2nd_order_plural_u(word):
            if stem and len(stem) >= 2:
                out.append(MorphSplit(stem=stem, prefixes=(), suffixes=(sfx,), rule=rule, strength=self.STRONG))
        return out

    def _peel_plurals(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        if ctx.has_exclusive_verb_sfx or ctx.has_exclusive_verb_pfx:
            return []
        out = []
        
        # Atomic plural + accusative (-plural-ን) splits for vowel-final stems,
        # where the plural is a plain suffix (ገበያዎችን -> ገበያ + [ዎች, ን]).  The
        # accusative -ን can only be peeled once the plural is gone, so the
        # combined surface must be recognized in one step.  Note that -ቶች/-ሎች
        # are NOT listed: those allomorphs replace the stem-final radical
        # (ቤት -> ቤቶች, ሐኪም -> ሐኪሞች) and are recovered by the 49-cell matrix
        # inversion instead.
        for plural, rule in (
            ("ዎች", "plural_accusative_woch"),
            ("ኦች", "plural_accusative_och"),
        ):
            if word.endswith(plural + "ን") and len(word) > len(plural) + 1:
                stem = word[: -(len(plural) + 1)]
                if len(stem) >= 2:
                    out.append(MorphSplit(stem=stem, prefixes=(), suffixes=(plural, "ን"), rule=rule, strength=self.STRONG))

        if word.endswith("ዎች") and len(word) > 2:
            out.append(MorphSplit(stem=word[:-2], prefixes=(), suffixes=("ዎች",), rule="plural_allomorph_direct", strength=self.STRONG))

        for ur_plural in ("ኦች", "ዎች", "ሎች", "ቶች"):
            for stem, ur, rule in self.phonology.invert_via_49_cell_matrix(word, ur_plural):
                out.append(MorphSplit(stem=stem, prefixes=(), suffixes=(ur,), rule=rule, strength=self.STRONG))

        if word and len(word) >= 3 and word in self.morphology.geez_plurals:
            if word.endswith("ት") and len(word) >= 3:
                penult_c = word[-2]
                _, penult_ord = self.phonology.get_char_info(penult_c)
                if penult_ord == 3:
                    sades_c = self.phonology.shift_order(penult_c, 5)
                    singular_stem = word[:-2] + sades_c
                    out.append(MorphSplit(stem=singular_stem, prefixes=(), suffixes=("አት",), rule="geez_plural_at", strength=self.STRONG))
            elif word.endswith("ያት") and len(word) >= 4:
                out.append(MorphSplit(stem=word[:-2], prefixes=(), suffixes=("አት",), rule="geez_plural_at", strength=self.STRONG))

        return out

    def _peel_definiteness(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        if ctx.has_poss_or_def_sfx or ctx.has_exclusive_verb_sfx or ctx.has_exclusive_verb_pfx:
            return []
        out = []
        
        for ur in ("ው", "ዋ"):
            if word.endswith(ur) and len(word) > len(ur) + 1:
                if ur == "ው" and word.endswith("ተው"):
                    continue
                out.append(MorphSplit(stem=word[:-len(ur)], prefixes=(), suffixes=(ur,), rule=f"definite_{ur}_direct", strength=self.STRONG))

        if word not in getattr(self.morphology, "inherent_terminal_u_stems", set()):
            for stem, ur_sfx, rule in self.phonology.invert_via_49_cell_matrix(word, "ኡ"):
                out.append(MorphSplit(stem=stem, prefixes=(), suffixes=(ur_sfx,), rule=rule, strength=self.STRONG))

        # Generalized Labialized 3FS Feminine Definite/Possessive (-wa) Inversion
        # Handles: ቷ->ት, ኗ->ን, ሯ->ር, ሟ->ም, ሷ->ስ, ጇ->ጅ, ሏ->ል, ፏ->ፍ, ጧ->ጥ, ጯ->ጭ, ጿ->ጽ
        # Accepts len >= 2 to support authentic 2-character bases (e.g. ቤቷ -> ቤት + ዋ, ልጇ -> ልጅ + ዋ)
        LABIALIZED_FEMININE_CODAS = ("ቷ", "ኗ", "ሯ", "ሟ", "ሷ", "ጇ", "ሏ", "ፏ", "ጧ", "ጯ", "ጿ")
        if len(word) >= 2 and word[-1] in LABIALIZED_FEMININE_CODAS:
            c_last = word[-1]
            restored_sadis = self.phonology.shift_order(c_last, 5)
            restored_stem = word[:-1] + restored_sadis
            if len(restored_stem) >= 2:
                out.append(MorphSplit(
                    stem=restored_stem,
                    prefixes=(),
                    suffixes=("ዋ",),
                    rule="definite_labialized_3FS_restored",
                    strength=self.STRONG
                ))

        if word.endswith("ቲቷ") and len(word) >= 4:
            restored_stem = word[:-2] + self.phonology.shift_order(word[-2], 5)
            out.append(MorphSplit(stem=restored_stem, prefixes=(), suffixes=("ዋ",), rule="definite_itwa_fusion_restored", strength=self.STRONG))

        if word.endswith("ቲቱ") and len(word) >= 4:
            restored_stem = word[:-2] + self.phonology.shift_order(word[-2], 5)
            out.append(MorphSplit(stem=restored_stem, prefixes=(), suffixes=("ኢቱ",), rule="definite_f_itu", strength=self.STRONG))

        return out

    def _peel_adjectivizers(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        out = []
        adjectivizer_urs = ["አዊ", "አዊያን", "አዊት"]

        for ur in adjectivizer_urs:
            for stem, ur_sfx, rule in self.phonology.invert_via_49_cell_matrix(word, ur):
                if ctx.known_stem_checker and not ctx.known_stem_checker(stem):
                    continue
                out.append(MorphSplit(stem=stem, prefixes=(), suffixes=(ur_sfx,), rule=f"adjectivizer_{ur}", strength=self.STRONG))
        return out

    def _peel_perfective_subjects(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        out = []
        if self.morphology.is_valid_infinitive_stem(word) or word.startswith(("ይ", "ት", "እ", "እን", "ያ", "ታ", "ና")):
            return out

        if word.endswith("ሁ") and len(word) >= 3:
            out.append(MorphSplit(stem=word[:-1], prefixes=(), suffixes=("ሁ",), rule="perfective_subject_1S_direct", strength=self.STRONG))

        if word.endswith("ኹ") and len(word) >= 3:
            out.append(MorphSplit(stem=word[:-1], prefixes=(), suffixes=("ኹ",), rule="perfective_subject_1S_direct", strength=self.STRONG))

        if word.endswith("ኽ") and len(word) >= 3:
            out.append(MorphSplit(stem=word[:-1], prefixes=(), suffixes=("ኽ",), rule="perfective_subject_2MS_direct", strength=self.STRONG))

        if word.endswith("ች") and len(word) >= 3:
            stem_vowel = word[:-1]
            _, v_order = self.phonology.get_char_info(stem_vowel[-1])
            if v_order in (0, 3, 4):
                out.append(MorphSplit(stem=stem_vowel, prefixes=(), suffixes=("ኧች",), rule="perfective_subject_3FS_vowel", strength=self.STRONG))

        for sfx_ur, person in self.morphology.perfective_subject_urs:
            if sfx_ur in ("ሁ", "ኹ", "ክ", "ኽ", "ሽ", "ን", "ነን"):
                if sfx_ur == "ክ" and self._has_protected_verbal_coda(
                    word, ctx.known_verbal_stem_checker or ctx.known_stem_checker
                ):
                    continue
                if word.endswith(sfx_ur) and len(word) > len(sfx_ur) + 1:
                    stem_direct = word[:-len(sfx_ur)]
                    if sfx_ur == "ን" and stem_direct.endswith(
                        ("ች", "ሽ", "አች", "ያች", "ዋች", "ው", "ቸው", "ት", "አት", "ል", "ብ")
                    ):
                        continue
                    out.append(MorphSplit(stem=stem_direct, prefixes=(), suffixes=(sfx_ur,), rule=f"perfective_subject_{person}_direct", strength=self.STRONG))
            elif sfx_ur != "ኧች":
                for stem, ur, _ in self.phonology.invert_via_49_cell_matrix(word, sfx_ur):
                    if stem and len(stem) >= 2:
                        out.append(MorphSplit(stem=stem, prefixes=(), suffixes=(ur,), rule=f"perfective_subject_{person}", strength=self.STRONG))

        return out

    def _peel_imperfective_subjects(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        out = []
        if not word or len(word) < 3 or self.morphology.is_valid_infinitive_stem(word):
            return out
        checker = ctx.known_verbal_stem_checker or ctx.known_stem_checker

        EN_SUBORDINATE_EXCLUSIONS = ("እንድ", "እንዲ", "እንደ", "እንኳ")
        if word.startswith("እን") and len(word) >= 4 and not any(word.startswith(p) for p in EN_SUBORDINATE_EXCLUSIONS):
            rem = word[2:]
            if self.morphology.is_attested_verbal_stem(rem, checker) or self.morphology.is_valid_imperfective_stem(rem):
                out.append(MorphSplit(stem=rem, prefixes=("እን",), suffixes=(), rule="imp_subject_1PL", strength=self.STRONG))

        if word.startswith("ይ") and (len(word) >= 3 or word in self.morphology.irregular_verbs_say or word in ("ይላ", "ይሎ", "ይለ")):
            rem = word[1:]
            if rem.endswith("ኡ") and len(rem) >= 3:
                inner = rem[:-1]
                if self.morphology.is_attested_verbal_stem(inner, checker) or self.morphology.is_valid_imperfective_stem(inner):
                    out.append(MorphSplit(stem=inner, prefixes=("ይ",), suffixes=("ኡ",), rule="imp_subject_3P", strength=self.STRONG))
            elif self.morphology.is_attested_verbal_stem(rem, checker) or self.morphology.is_valid_imperfective_stem(rem):
                out.append(MorphSplit(stem=rem, prefixes=("ይ",), suffixes=(), rule="imp_subject_3MS", strength=self.STRONG))

        if word.startswith("ት") and len(word) >= 3:
            rem = word[1:]
            if rem.endswith("ኡ") and len(rem) >= 3:
                inner = rem[:-1]
                if self.morphology.is_attested_verbal_stem(inner, checker) or self.morphology.is_valid_imperfective_stem(inner):
                    out.append(MorphSplit(stem=inner, prefixes=("ት",), suffixes=("ኡ",), rule="imp_subject_2P", strength=self.STRONG))
            elif rem.endswith("ኢ") and len(rem) >= 3:
                inner = rem[:-1]
                if self.morphology.is_attested_verbal_stem(inner, checker) or self.morphology.is_valid_imperfective_stem(inner):
                    out.append(MorphSplit(stem=inner, prefixes=("ት",), suffixes=("ኢ",), rule="imp_subject_2FS", strength=self.STRONG))
            else:
                _, last_order = self.phonology.get_char_info(rem[-1])
                if last_order == 2:
                    for stem_pal, sfx_pal, rule_pal in self.phonology.invert_palatalization(rem):
                        if rule_pal == "palatal_2fs_i":
                            if (
                                self.morphology.is_attested_verbal_stem(stem_pal, checker)
                                or self.morphology.is_valid_imperfective_stem(stem_pal)
                                or self.morphology.is_valid_jussive_stem(stem_pal)
                            ):
                                out.append(MorphSplit(stem=stem_pal, prefixes=("ት",), suffixes=("ኢ",), rule="imp_subject_2FS_palatal", strength=self.STRONG))

                if self.morphology.is_attested_verbal_stem(rem, checker) or self.morphology.is_valid_imperfective_stem(rem):
                    out.append(MorphSplit(stem=rem, prefixes=("ት",), suffixes=(), rule="imp_subject_2MS_3FS", strength=self.STRONG))

        if word.startswith("እ") and len(word) >= 3:
            rem = word[1:]
            _, c2_ord = self.phonology.get_char_info(word[1])
            orders = tuple(self.phonology.get_vocalic_orders(word))

            is_initial_weak_cmd = (len(word) == 3 and orders in ((5, 0, 5), (5, 5)))
            if c2_ord != 5 and not is_initial_weak_cmd and not any(word.startswith(p) for p in self.morphology.subject_e_exclusions):
                if (
                    self.morphology.is_attested_verbal_stem(rem, checker)
                    or self.morphology.is_valid_imperfective_stem(rem)
                    or self.morphology.is_valid_jussive_stem(rem)
                ):
                    out.append(MorphSplit(stem=rem, prefixes=("እ",), suffixes=(), rule="imp_subject_1S", strength=self.STRONG))

        return out

    def _peel_bare_palatal_imperatives(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        if ctx.prefixes or len(word) < 2:
            return []
        out = []
        verbal_checker = ctx.known_verbal_stem_checker or ctx.known_stem_checker
        _, last_ord = self.phonology.get_char_info(word[-1])
        if last_ord == 2:
            for stem_pal, sfx_pal, rule_pal in self.phonology.invert_palatalization(word):
                if rule_pal == "palatal_2fs_i":
                    if self.morphology.is_valid_jussive_stem(stem_pal) or self.morphology.is_attested_verbal_stem(stem_pal, verbal_checker):
                        out.append(MorphSplit(stem=stem_pal, prefixes=(), suffixes=(sfx_pal,), rule="imperative_2FS_palatal", strength=self.STRONG))
        return out

    def _peel_abstract_derivational(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        out = []
        if len(word) > 3:
            for stem, ur_sfx, rule in self.phonology.invert_rule_2_schwa_epenthesis(word):
                if stem and len(stem) >= 2:
                    out.append(MorphSplit(stem=stem, prefixes=(), suffixes=(ur_sfx,), rule=rule, strength=self.STRONG))
        return out

    def _peel_auxiliaries(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        out = []
        if len(word) < 3 or self.morphology.is_valid_infinitive_stem(word):
            return out

        for stem, affixes, rule in self.phonology.invert_labialized_coverb_auxiliary(word):
            if len(stem) >= 2:
                out.append(MorphSplit(stem=stem, prefixes=(), suffixes=affixes, rule=rule, strength=self.STRONG))

        imp_pfx_licensors = ("ይ", "ት", "እ", "እን", "ያ", "ታ", "ና", "የሚ", "የምት", "የምና", "የምታ")
        has_context_imp_pfx = any(p in ("ይ", "ት", "እ", "እን") for p in ctx.prefixes)

        for tail, ur_aux in self.morphology.auxiliary_coda_shifts.items():
            for stem, _ in self.phonology.invert_coda_shift_4th_to_6th(word, tail):
                has_subj_pfx = has_context_imp_pfx or any(stem.startswith(p) for p in imp_pfx_licensors)
                if has_subj_pfx or self.morphology.is_valid_coverb_stem(stem):
                    if len(stem) >= 2:
                        out.append(MorphSplit(stem=stem, prefixes=(), suffixes=(ur_aux,), rule=f"verb_auxiliary_{ur_aux}", strength=self.STRONG))

            # 2. Final-weak imperfectives ending natively in 4th-order Rabe
            # (e.g. ይመታል -> ይመታ, ያመጣል -> ያመጣ, ያማል -> ያማ, ያሻል -> ያሻ, ያጣል -> ያጣ, ይላል -> ይላ, ይሻል -> ይሻ)
            if word.endswith(tail) and len(word) >= len(tail) + 2:
                # Terminal -ቷል is strictly the fused coverb auxiliary (-to + -al); never an imperfective stem taking -al
                if tail == "ል" and word.endswith("ቷል"):
                    continue
                weak_stem = word[:-len(tail)]
                _, w_last_o = self.phonology.get_char_info(weak_stem[-1])
                has_weak_imp_pfx = has_context_imp_pfx or any(weak_stem.startswith(p) for p in imp_pfx_licensors)
                
                intact_is_type_a_jussive = (
                    tail == "ል"
                    and word.startswith("ይ")
                    and (word[1:] in getattr(self.morphology, "biradical_type_a_jussive_stems", set()))
                )

                is_weak_imp = (
                    len(weak_stem) >= 2
                    and (w_last_o in (3, 7) or weak_stem in self.morphology.irregular_verbs_say)
                    and has_weak_imp_pfx
                    and not intact_is_type_a_jussive
                )
                if is_weak_imp:
                    out.append(MorphSplit(stem=weak_stem, prefixes=(), suffixes=(ur_aux,), rule=f"verb_auxiliary_{ur_aux}_weak", strength=self.STRONG))
                    
        for sfx, meta in self.morphology.compound_auxiliary_suffixes.items():
            if word.endswith(sfx):
                stem = word[:-len(sfx)] + meta["stem_suffix"]
                if len(stem) >= 2:
                    out.append(MorphSplit(stem=stem, prefixes=(), suffixes=tuple(meta["affixes"]), rule=meta["rule"], strength=self.STRONG))

        return out

    def _peel_negative_finiteness(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        out = []
        if not ctx.has_negative_context:
            return out
        if word.endswith("ም") and len(word) >= 3:
            out.append(MorphSplit(stem=word[:-1], prefixes=(), suffixes=("እም",), rule="negative_finiteness_em", strength=self.STRONG))
        return out

    def _peel_verb_object_clitics(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        out = []
        if len(word) < 3 or self.morphology.is_valid_infinitive_stem(word):
            return out

        if word in getattr(self.morphology, "geez_plurals", set()) or word in getattr(self.morphology, "confirmed_sparse_nouns", set()):
            return out

        checker = ctx.known_verbal_stem_checker or ctx.known_stem_checker
        all_multi_sfx = self.morphology.suffix_nominal_abstract | self.morphology.suffix_nominal_derivational
        for multi_s in all_multi_sfx:
            if word.endswith(multi_s) and len(word) > len(multi_s) + 1:
                base_s = word[:-len(multi_s)]
                if checker and checker(base_s):
                    return out

        for tail, obj_ur in (("ት", "አት"), ("ቸው", "አቸው"), ("ችሁ", "አችሁ"), ("ቹ", "አቹ")):
            if word.endswith(tail) and len(word) >= len(tail) + 2:
                penult_c = word[-len(tail) - 1]
                _, penult_o = self.phonology.get_char_info(penult_c)
                if penult_o == 7 or penult_c in ("ቋ", "ኳ", "ጓ", "ኋ", "ሯ", "ቷ", "ሷ", "ኗ", "ሟ", "ሏ", "ፏ", "ጇ", "ጧ", "ጯ", "ጿ"):
                    bound_c5 = self.phonology.shift_order(penult_c, 5)
                    bound_stem = word[:-len(tail) - 1] + bound_c5
                    if word[:-len(tail) - 1].endswith("ት") and len(word) >= len(tail) + 3:
                        out.append(MorphSplit(stem=word[:-len(tail) - 1], prefixes=(), suffixes=("ቶ", obj_ur), rule=f"coverb_weak_obj_{obj_ur}", strength=self.STRONG))
                    elif len(bound_stem) >= 2:
                        out.append(MorphSplit(stem=bound_stem, prefixes=(), suffixes=("ኦ", obj_ur), rule=f"coverb_labialized_obj_{obj_ur}", strength=self.STRONG))

        for clitic_tail in ("ኝ", "ህ", "ሽ", "ት", "ን", "ዎት", "ዎ"):
            if word.endswith(clitic_tail) and len(word) >= len(clitic_tail) + 2:
                base_part = word[:-len(clitic_tail)]
                penult_c = base_part[-1]
                _, penult_o = self.phonology.get_char_info(penult_c)
                if penult_o == 6 and penult_c not in self.phonology.LABIOVELAR_MAP:
                    bound_c5 = self.phonology.shift_order(penult_c, 5)
                    bound_stem = base_part[:-1] + bound_c5
                    if len(bound_stem) >= 2:
                        out.append(MorphSplit(stem=bound_stem, prefixes=(), suffixes=("ኦ", clitic_tail), rule=f"coverb_3ms_o_obj_{clitic_tail}", strength=self.STRONG))
                elif base_part.endswith("ቶ") and len(base_part) >= 2:
                    out.append(MorphSplit(stem=base_part, prefixes=(), suffixes=("ቶ", clitic_tail), rule=f"coverb_3ms_to_obj_{clitic_tail}", strength=self.STRONG))

        if word.endswith("ት") and len(word) >= 4:
            penult_c = word[-2]
            _, at_order = self.phonology.get_char_info(penult_c)
            if at_order == 3 and penult_c not in ("ቋ", "ኳ", "ጓ", "ኋ", "ሯ", "ቷ"):
                stem_3ms = word[:-2] + self.phonology.shift_order(penult_c, 0)
                out.append(MorphSplit(stem=stem_3ms, prefixes=(), suffixes=("አት",), rule="verb_object_at_3ms_restored", strength=self.STRONG))
                stem_c5 = word[:-2] + self.phonology.shift_order(penult_c, 5)
                out.append(MorphSplit(stem=stem_c5, prefixes=(), suffixes=("አት",), rule="verb_object_at_c5_restored", strength=self.STRONG))

        for rev_achew_stem, _ in self.phonology.invert_coda_shift_4th_to_6th(word, "ቸው"):
            if len(rev_achew_stem) >= 2:
                out.append(MorphSplit(stem=rev_achew_stem, prefixes=(), suffixes=("አቸው",), rule="verb_object_achew_4th_coda", strength=self.STRONG))
                
        if word.endswith("ቸው") and len(word) >= 3:
            stem_part = word[:-2]
            _, last_ord = self.phonology.get_char_info(stem_part[-1])
            if last_ord == 3:
                stem_0 = stem_part[:-1] + self.phonology.shift_order(stem_part[-1], 0)
                if len(stem_0) >= 2:
                    out.append(MorphSplit(stem=stem_0, prefixes=(), suffixes=("አቸው",), rule="verb_object_achew_3ms_restored", strength=self.STRONG))
                if stem_part.endswith("ላ") and len(stem_part) >= 2:
                    stem_restored = stem_part[:-1] + self.phonology.shift_order(stem_part[-1], 0)
                    if stem_restored in ("አለ", "በለ"):
                        out.append(MorphSplit(stem=stem_restored, prefixes=(), suffixes=("አቸው",), rule="verb_object_achew_coalescence", strength=self.STRONG))

        if not word.endswith("ተው"):
            for rev_w_stem, _ in self.phonology.invert_coda_shift_1st_to_6th(word, "ው"):
                if len(rev_w_stem) >= 2:
                    out.append(MorphSplit(stem=rev_w_stem, prefixes=(), suffixes=("ው",), rule="verb_obj_w_coda_shift", strength=self.STRONG))

            if word.endswith("ው") and len(word) >= 3:
                stem_vowel = word[:-1]
                last_c = stem_vowel[-1]
                base_c, last_v_order = self.phonology.get_char_info(last_c)
                if (
                    last_v_order in (0, 3, 4, 7)
                    or last_c in self.phonology.PALATAL_CONSONANTS
                    or base_c in self.phonology.PALATAL_CONSONANTS
                ):
                    out.append(MorphSplit(stem=stem_vowel, prefixes=(), suffixes=("ው",), rule="verb_object_w_direct", strength=self.STRONG))

        for obj_cl in ("ኝ", "ህ", "ሽ", "ን", "ት", "ዎ", "ዎት"):
            if word.endswith(obj_cl) and len(word) >= len(obj_cl) + 2:
                rem = word[:-len(obj_cl)]
                if len(rem) >= 2:
                    out.append(MorphSplit(stem=rem, prefixes=(), suffixes=(obj_cl,), rule="verb_object_clitic", strength=self.NORMAL))

        return out

    def _peel_coverbs(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        out = []
        if any(p in self.morphology.all_subject_prefixes for p in ctx.prefixes):
            return out
        if word.startswith(("ይ", "ት", "እ", "እን")) and word not in ("ይዞ", "ይዛ", "ይዘው", "ይዘን", "ይዘህ", "ይዘሽ"):
            return out

        checker = ctx.known_verbal_stem_checker or ctx.known_stem_checker
        has_explicit_preposition = any(
            p in self.morphology.prefix_universal for p in ctx.prefixes
        )
        if has_explicit_preposition or len(word) < 2 or self.morphology.is_valid_infinitive_stem(word):
            return out

        for stem_o, sfx_o, rule_o in self.phonology.invert_coverb_3ms_o(word):
            if self.morphology.is_attested_verbal_stem(stem_o, checker) or self.morphology.is_valid_coverb_stem(stem_o):
                if len(stem_o) >= 2:
                    out.append(MorphSplit(stem=stem_o, prefixes=(), suffixes=(sfx_o,) if isinstance(sfx_o, str) else sfx_o, rule=rule_o, strength=self.STRONG))

        for stem_w, sfx_w, rule_w in self.phonology.invert_coverb_3pl_w(word):
            if self.morphology.is_attested_verbal_stem(stem_w, checker) or self.morphology.is_valid_coverb_stem(stem_w):
                if len(stem_w) >= 2:
                    out.append(MorphSplit(stem=stem_w, prefixes=(), suffixes=(sfx_w,), rule=rule_w, strength=self.STRONG))

        for stem_p, sfx_p, rule_p in self.phonology.invert_palatalization(word):
            if rule_p.startswith("palatal_coverb_1s") or rule_p.startswith("coverb_1s"):
                if len(stem_p) >= 2:
                    out.append(MorphSplit(stem=stem_p, prefixes=(), suffixes=(sfx_p,), rule=rule_p, strength=self.STRONG))

        _, last_order = self.phonology.get_char_info(word[-1])
        if last_order == 3 and len(word) >= 3:
            _, penult_order = self.phonology.get_char_info(word[-2])
            if penult_order == 5 and not word.startswith(("መ", "ማ")) and word not in ("ታድያ", "ታዲያ", "መርሀ"):
                word_orders = tuple(self.phonology.get_vocalic_orders(word))
                if word_orders not in ((0, 0, 3), (0, 3), (5, 3)):
                    stem_sades = word[:-1] + self.phonology.shift_order(word[-1], 5)
                    is_attested = (
                        self.morphology.is_attested_verbal_stem(stem_sades, checker)
                        or stem_sades in self.morphology.verb_stems
                    )
                    if is_attested and self.morphology.is_valid_coverb_stem(stem_sades):
                        if len(stem_sades) >= 2:
                            out.append(MorphSplit(stem=stem_sades, prefixes=(), suffixes=("አ",), rule="coverb_3fs_a", strength=self.STRONG))

        for sfx_weak in self.morphology.suffix_verb_coverb_weak:
            if word.endswith(sfx_weak) and len(word) >= len(sfx_weak) + 2:
                rem_weak = word[:-len(sfx_weak)]

                if len(rem_weak) >= 2:
                    _, p_ord = self.phonology.get_char_info(rem_weak[-1])
                    if p_ord not in (5, 3):
                        continue

                if self.morphology.is_valid_weak_coverb_stem(
                    rem_weak, checker=checker, suffix=sfx_weak
                ) or self.morphology.is_valid_coverb_stem(rem_weak):
                    if len(rem_weak) >= 2:
                        out.append(MorphSplit(stem=rem_weak, prefixes=(), suffixes=(sfx_weak,), rule="coverb_weak_suffix", strength=self.STRONG))

        return out

    def _peel_literal_clitics_and_prefixes(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        out = []
        verbal_checker = ctx.known_verbal_stem_checker or ctx.known_stem_checker

        # --- Peel Literal Clitics ---
        if word and len(word) >= 3:
            final_c = word[-1]
            is_jussive_verb = "አት" in ctx.prefixes

            for x in self.morphology.clitics_by_final.get(final_c, []):
                if word.endswith(x) and len(word) > len(x) + 1:
                    if (
                        x == "ን"
                        and hasattr(self.morphology, "possessive_1pl_endings")
                        and word.endswith(self.morphology.possessive_1pl_endings)
                    ):
                        continue

                    if x == "ን" and any(s in self.morphology.suffix_nominal_possessive for s in ctx.suffixes):
                        continue

                    if word[-1] in self._protected_coda_finals() and self._has_protected_verbal_coda(
                        word, verbal_checker
                    ):
                        continue

                    if x == "ስ" and (is_jussive_verb or (len(word) >= 5 and word.startswith("ይ") and word[-2] == "ን")):
                        continue

                    # 1. Phonologically robust 1-2-1-2 reduplicative shield
                    if x in ("ስ", "ም", "ው") and len(word) >= 5 and word.startswith(("ይ", "ት", "እ", "እን")):
                        b1, _ = self.phonology.get_char_info(word[1])
                        b2, _ = self.phonology.get_char_info(word[2])
                        b3, _ = self.phonology.get_char_info(word[3])
                        b4, _ = self.phonology.get_char_info(word[4])
                        if b1 == b3 and b2 == b4:
                            continue

                    has_subj_concord = (
                        any(p in self.morphology.all_subject_prefixes for p in ctx.prefixes)
                        or word.startswith(("ይ", "ት", "እ", "እን", "ያ", "ታ", "ና"))
                    )
                    if x in ("ም", "ስ", "ን") and has_subj_concord:
                        _, last_coda = self.phonology.get_char_info(word[-1])
                        if last_coda == 5:
                            continue

                    if x == "ማ" and (ctx.suffixes or (word.startswith("የ") and len(word) == 4)):
                        continue
                    if x == "ና" and (word.startswith("የ") and len(word) == 4):
                        continue
                    rem = word[:-len(x)]

                    if x in ("ም", "ን", "ክ", "ት"):
                        if self.morphology.is_valid_verbal_noun_stem(word) and not self.morphology.is_valid_verbal_noun_stem(rem):
                            continue
                        if x == "ን" and not any(p in ("የ", "የሚ") for p in ctx.prefixes):
                            if (
                                word.startswith(("ይ", "ት", "እ", "እን"))
                                or tuple(self.phonology.get_vocalic_orders(word)) in self.morphology.LESLAU_JUSSIVE_PATTERNS
                            ):
                                continue

                    if x in ("ም", "ስ") and len(word) in (3, 4) and not ctx.prefixes:
                        penult_c = word[-2]
                        _, penult_ord = self.phonology.get_char_info(penult_c)
                        if penult_ord == 1:
                            continue

                    if x in self.morphology.suffix_accusative:
                        rule_name = "case_accusative"
                    elif hasattr(self.morphology, "suffix_pragmatic_clitics") and x in self.morphology.suffix_pragmatic_clitics:
                        rule_name = f"clitic_pragmatic_{x}"
                    else:
                        rule_name = f"clitic_literal_{x}"

                    if rem and len(rem) >= 2:
                        out.append(MorphSplit(stem=rem, prefixes=(), suffixes=(x,), rule=rule_name, strength=self.WEAK))

        # --- Peel Literal Prefixes ---
        if word and len(word) >= 3 and not (any(p in self.morphology.prefix_verb_atomic_negative for p in ctx.prefixes) or len(ctx.prefixes) >= 4):
            for multi_p in self.morphology.prefix_nominal_compound:
                if word.startswith(multi_p) and len(word) > len(multi_p) + 1:
                    rem_compound = word[len(multi_p):]
                    if ctx.known_stem_checker and ctx.known_stem_checker(rem_compound):
                        is_valid_len = len(rem_compound) >= 2 or rem_compound in getattr(self.morphology, "confirmed_sparse_verbs", set())
                        if rem_compound and is_valid_len:
                            out.append(MorphSplit(stem=rem_compound, prefixes=(multi_p,), suffixes=(), rule=f"compound_prefix:{multi_p}", strength=self.WEAK))

            initial_c = word[0]
            prefixes_to_test = sorted(self.morphology.prefixes_by_initial.get(initial_c, []), key=len, reverse=True)

            fused_composite_prefixes = {
                lic + rel
                for lic in ("", "የ", "በ", "ለ", "ከ", "እንደ", "ስለ", "እስከ", "ወደ")
                for rel in ("ሚ", "ሚያ", "ምት", "ምን", "ምና", "ተ")
                if lic + rel not in ("", "ተ")
            }

            for x in prefixes_to_test:
                if not word.startswith(x):
                    continue
                if x == "የ" and word.startswith("የሚ"):
                    # The relative-imperfective marker is peeled as one licensed
                    # bundle above; do not leave a bare ሚ- inside the stem.
                    continue
                if x == "ለ" and self._has_licensed_coverb_surface(word, verbal_checker):
                    # Preserve lexical ለ-initial coverbs instead of treating ለ-
                    # as a preposition when the whole form has a licensed coverb analysis.
                    continue
                rem_stem = word[len(x):]
                if len(word) <= len(x) + 1 and rem_stem not in getattr(self.morphology, "confirmed_sparse_verbs", set()):
                    continue

                if x in fused_composite_prefixes:
                    continue

                if any(
                    word.startswith(p)
                    for p in self.morphology.atomic_multi_prefixes
                    if len(p) > len(x) and p.startswith(x)
                ):
                    continue

                if x in ("በ", "ለ", "ከ") and len(word) == 4:
                    w_orders = tuple(self.phonology.get_vocalic_orders(word))
                    if w_orders in ((0, 0, 0, 0), (0, 5, 5, 5)):
                        rem_candidate = word[len(x):]
                        rem_is_known = (
                            rem_candidate in self.morphology.verb_stems
                            or (verbal_checker and verbal_checker(rem_candidate))
                            or (ctx.known_stem_checker and ctx.known_stem_checker(rem_candidate))
                        )
                        if (verbal_checker or ctx.known_stem_checker) and not rem_is_known:
                            continue

                rem = word[len(x):]

                if x == "እ":
                    orders_w = tuple(self.phonology.get_vocalic_orders(word))
                    if (
                        (len(word) == 3 and orders_w in ((5, 5, 3), (5, 5, 5), (5, 0, 5), (5, 5)))
                        or (len(word) == 4 and orders_w == (5, 5, 0, 5))
                    ):
                        continue

                if x == "ያል":
                    _, last_coda = self.phonology.get_char_info(word[-1])
                    if last_coda == 5:
                        continue

                if x in ("አል", "አላ"):
                    has_outer_licensor = any(p in self.morphology.prefix_universal or p in self.morphology.prefix_relative_pronoun for p in ctx.prefixes)
                    if not word.endswith("ም") and not has_outer_licensor:
                        continue
                    if word.endswith("ም") and len(word) >= 4 and not has_outer_licensor:
                        _, penult_ord = self.phonology.get_char_info(word[-2])
                        if penult_ord == 5:
                            continue

                if x == "አለ":
                    if not (rem.startswith(("መ", "ማ")) and self.morphology.is_valid_verbal_noun_stem(rem)):
                        continue

                if x == "አ" and word.startswith(
                    ("አለብ", "አለል", "አለህ", "አለሽ", "አለች", "አለን", "አላችሁ")
                ):
                    continue

                if x in self.morphology.single_letter_proclitics and len(word) == 3:
                    stem_checker = verbal_checker or ctx.known_stem_checker
                    is_valid_2c = (
                        rem in self.morphology.valid_2char_stems
                        or self.morphology.is_licensed_2char_stem(rem)
                        or self.morphology.is_valid_perfective_template(rem)
                        or self.morphology.is_licensed_defective_root(rem)
                        or bool(ctx.known_stem_checker and ctx.known_stem_checker(rem))
                        # A remainder that is licensed verbal in its own right is
                        # the relative pronoun's complement, not a coincidence of
                        # the proclitic's shape: የሸና is የ + ሸና (perfective
                        # relative), so the whole-word template match is not
                        # decisive.  Same verbal evidence the attested-prefix
                        # gate below uses.
                        or self.morphology.is_valid_imperfective_stem(rem)
                        or self.morphology.is_valid_3ms_perfective_form(rem, stem_checker)
                        or self.morphology.is_attested_verbal_stem(rem, stem_checker)
                    )
                    if self.morphology.is_valid_perfective_template(word) and not is_valid_2c:
                        continue

                if x in ("ስ", "ብ") and not rem.startswith(("ት", "ን")):
                    continue

                if x in self.morphology.prefix_nominal_associatives:
                    if rem.startswith("ነበር") or self.morphology.is_valid_3ms_perfective_form(rem, ctx.known_stem_checker):
                        continue
                    if x.startswith("ከነ") and self.morphology.is_valid_perfective_template(word) and len(word) >= 4:
                        continue
                    if x == "እነ" and any(
                        word.endswith(aux)
                        for aux in ("ለሁ", "ለህ", "ለሽ", "ለች", "ለን", "ላችሁ", "ሉ", "ል")
                    ):
                        continue

                if x in self.morphology.verbal_prefixes_requiring_attestation:
                    is_valid_stem = (
                        self.morphology.is_valid_imperfective_stem(rem)
                        or self.morphology.is_attested_verbal_stem(rem, verbal_checker or ctx.known_stem_checker)
                        or self.morphology.is_valid_3ms_perfective_form(rem, verbal_checker or ctx.known_stem_checker)
                    )
                    if not is_valid_stem:
                        continue

                if x in self.morphology.prefix_verb_voice:
                    if any(
                        rem.endswith(sfx)
                        for sfx in (
                            self.morphology.suffix_nominal_exclusive_plural
                            | self.morphology.suffix_nominal_exclusive_abstract
                        )
                    ):
                        continue
                    if verbal_checker:
                        if not verbal_checker(rem):
                            continue
                    elif not self.morphology.is_attested_verbal_stem(rem):
                        continue

                is_valid_len = len(rem) >= 2 or rem in getattr(self.morphology, "confirmed_sparse_verbs", set())
                # Lexical shield for Ge'ez-order onsets.  በረገደ "be delayed" is one
                # 4-radical root, not በ- + ረገደ; without this the literal peel wins on
                # affix-analysis score even though በረገደ is the more frequent stem.
                # Data-driven and empty by default, so the blast radius is the list.
                if word in getattr(self.morphology, "proclitic_root_shield", set()):
                    is_valid_len = False
                if rem and is_valid_len:
                    pfx_tuple = (x,)
                    out.append(MorphSplit(stem=rem, prefixes=pfx_tuple, suffixes=(), rule=f"prefix_literal:{x}", strength=self.WEAK))

                e_restored_stem = "እ" + rem
                if (
                    x in self.morphology.prefix_universal
                    and e_restored_stem in self.morphology.pronoun_stems
                    and not rem.startswith(("ተ", "አ", "ያስ", "አስ"))
                ):
                    is_valid_len_e = len(e_restored_stem) >= 2 or e_restored_stem in getattr(self.morphology, "confirmed_sparse_verbs", set())
                    if e_restored_stem and is_valid_len_e:
                        out.append(MorphSplit(stem=e_restored_stem, prefixes=(x,), suffixes=(), rule="contracted_initial_e_restored", strength=self.WEAK))

                if x in self.morphology.prefix_universal and self.morphology.is_valid_3ms_perfective_form(rem, ctx.known_stem_checker):
                    if rem and is_valid_len:
                        out.append(MorphSplit(stem=rem, prefixes=(x, "የ"), suffixes=(), rule="prep_rel_perfective_elision", strength=self.WEAK))

        return out

    def _peel_raw_suffix_fallback(self, word: str, ctx: DecomposeContext) -> List[MorphSplit]:
        if not ctx.include_raw_suffix_fallback:
            return []
        out = []
        if not word or len(word) < 4:
            return out

        if self.morphology.is_valid_infinitive_stem(word):
            return out

        final_c = word[-1]
        is_jussive_verb = "አት" in ctx.prefixes

        for x in self.morphology.clitics_by_final.get(final_c, []):
            if word.endswith(x) and len(word) > len(x) + 1:
                if x == "ም" and self.morphology.is_valid_verbal_noun_stem(word):
                    continue

                if (
                    x == "ን"
                    and hasattr(self.morphology, "possessive_1pl_endings")
                    and word.endswith(self.morphology.possessive_1pl_endings)
                ):
                    continue

                rem = word[:-len(x)]

                if self.morphology.is_valid_infinitive_stem(rem):
                    continue

                if x in ("ች", "ቹ"):
                    _, order = self.phonology.get_char_info(rem[-1])
                    if order == 5:
                        continue

                if rem and len(rem) >= 2:
                    out.append(MorphSplit(stem=rem, prefixes=(), suffixes=(x,), rule="raw_suffix_literal", strength=self.WEAK))

        return out
