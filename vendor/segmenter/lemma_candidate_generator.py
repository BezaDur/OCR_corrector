"""Rule-based generation of Amharic verbal citation-form lemma candidates.

The generator accepts a verbal stem and optional shape metadata. Known Shapes
A–E select their own stem-family inversions before the generic fallback runs.
Candidate generation remains rule-driven: it does not attest or select a
winning lemma. Dictionary-backed maximal-lemma protection and arbitration
belong to :class:`MaximalLemmaArbitrator`, a separate component.

The engine injects only the phonology, a small morphology-helper interface,
and its bounded cache, so these rules can be inspected and developed apart
from the larger morphology engine.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Protocol, Set, Tuple


class PhonologySupport(Protocol):
    """Phonological operations required by the inverse rules."""

    def get_char_info(self, char: str) -> Tuple[str, int]: ...

    def shift_order(self, char: str, target_order: int) -> str: ...

    def invert_coverb_3ms_o(self, word: str) -> List[Tuple[str, Any, str]]: ...

    def invert_coverb_3pl_w(self, word: str) -> List[Tuple[str, str, str]]: ...

    def invert_labialized_coverb_auxiliary(
        self, word: str
    ) -> List[Tuple[str, Any, str]]: ...


class MorphologySupport(Protocol):
    """Small, engine-independent surface consumed from morphology."""

    suppletive_stem_templates: Mapping[str, Mapping[str, str]]
    VERBAL_RADICAL_PATTERNS: Mapping[Any, Set[Tuple[int, ...]]]
    LESLAU_PERFECTIVE_PATTERNS: Set[Tuple[int, ...]]
    LESLAU_IMPERFECTIVE_PATTERNS: Set[Tuple[int, ...]]
    LESLAU_JUSSIVE_PATTERNS: Set[Tuple[int, ...]]
    LESLAU_INFINITIVE_PATTERNS: Set[Tuple[int, ...]]

    def is_reduplicative_quadriradical(self, stem: str) -> bool: ...

    def is_frequentative(self, stem: str) -> bool: ...


class LemmaCache(Protocol):
    """Minimal cache contract; the engine's bounded LRU implements it."""

    def __contains__(self, key: str) -> bool: ...

    def __getitem__(self, key: str) -> Tuple[str, ...]: ...

    def __setitem__(self, key: str, value: Tuple[str, ...]) -> None: ...


class LemmaCandidateGenerator:
    """Generate ordered citation-form lemma candidates using morphology rules.

    ``morphology`` and ``phonology`` are intentionally injected; the
    generator consumes only their small support interfaces. When supplied,
    ``cache`` is shared with the engine so its bounds and config-reload clearing
    semantics are retained. Candidate generation is not dictionary attestation
    or final lexical disambiguation.
    """

    _SHAPE_INVERTER_NAMES = {
        "SHAPE_A_PERFECTIVE": "_lemmatize_shape_a",
        "SHAPE_C_IMPERFECTIVE": "_lemmatize_shape_c",
        "SHAPE_D_JUSSIVE_IMPERATIVE": "_lemmatize_shape_d",
        "SHAPE_E_INFINITIVE": "_lemmatize_shape_e",
    }

    def __init__(
        self,
        morphology: MorphologySupport,
        phonology: PhonologySupport,
        cache: Optional[LemmaCache] = None,
    ) -> None:
        self.morphology = morphology
        self.phonology = phonology
        self.cache: LemmaCache = cache if cache is not None else {}

    def generate_candidates(
        self, stem: str, shape: Optional[Any] = None
    ) -> Tuple[str, ...]:
        """Return ordered rule-derived citations; do not attest them lexically."""
        shape_value = getattr(shape, "value", shape) if shape else None
        shape_key = shape_value or "NONE"
        cache_key = f"{stem}:{shape_key}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Route each known vocalic shape through its own inverse family. Shape B
        # retains its established coverb rules below; the other four shapes now
        # use their canonical family templates before falling back to generic
        # stem rules.
        inverter_name = self._SHAPE_INVERTER_NAMES.get(shape_value)
        if inverter_name is not None:
            result = getattr(self, inverter_name)(stem)
            self.cache[cache_key] = result
            return result

        if shape_value == "SHAPE_B_COVERB":
            candidates: List[str] = []
            coverb_stems: List[str] = []

            # Compound converbs first yield the pre-auxiliary Shape-B stem.
            for coverb_stem, _affixes, _rule in self.phonology.invert_labialized_coverb_auxiliary(stem):
                coverb_stems.append(coverb_stem)

            # Bare 3MS converbs yield the corresponding bound imperfective grade.
            for coverb_stem, _suffix, _rule in self.phonology.invert_coverb_3ms_o(stem):
                coverb_stems.append(coverb_stem)
            for coverb_stem, _suffix, _rule in self.phonology.invert_coverb_3pl_w(stem):
                coverb_stems.append(coverb_stem)

            if coverb_stems:
                for coverb_stem in dict.fromkeys(coverb_stems):
                    candidates.extend(self._coverb_shape_alternatives(coverb_stem))
                    candidates.extend(self.generate_candidates(coverb_stem))
            else:
                # Already-inverted weak bases can enter directly from shape-aware
                # consumers; retain their weak citation grade without relabeling a
                # complete inflected surface as if it were a bare stem.
                candidates.extend(self._coverb_shape_alternatives(stem))
                candidates.extend(self.generate_candidates(stem))

            result = tuple(dict.fromkeys(candidates))
            self.cache[cache_key] = result
            return result

        # Suppletive Aspectual Theme Resolution (ይዝ -> ያዘ, ብል -> አለ, ሆን -> ሆነ)
        if hasattr(self.morphology, "suppletive_stem_templates") and stem in self.morphology.suppletive_stem_templates:
            return (self.morphology.suppletive_stem_templates[stem]["lemma"],)

        if stem in ("ል", "እል"):
            return ("አለ", "በለ")
        if not stem or len(stem) < 2:
            return ()

        # Canonical Modal Verb Lemma Inversion (ችል/ቻል -> ቻለ)
        if stem in ("ችል", "ቻል"):
            return ("ቻለ", "ቻል")

        c0, last_char = stem[0], stem[-1]
        _, o0 = self.phonology.get_char_info(c0)
        _, last_order = self.phonology.get_char_info(last_char)

        # If already a 3MS perfective citation lemma (ends in vowel 0/3: ቀለቀለ, ለሰለሰ, ሰባበረ)
        if (self.morphology.is_reduplicative_quadriradical(stem) or self.morphology.is_frequentative(stem)) and last_order in (0, 3):
            return (stem, "ተ" + stem)

        cands: List[str] = []
        _, o0 = self.phonology.get_char_info(c0)
        _, last_order = self.phonology.get_char_info(last_char)

        # -------------------------------------------------------------
        # 1. 2-Character Bi-Radical Stems
        # -------------------------------------------------------------
        if len(stem) == 2:
            if last_order in (0, 3):
                cands.append(stem)
                if o0 == 3 and last_order == 3:
                    # Full dictionary lemma derivations: አሻ -> አሸ, አካ -> አከለ, አዛ -> አዘዘ
                    c_geez = self.phonology.shift_order(last_char, 0)
                    cands.append(c0 + c_geez)              # e.g. አሸ
                    cands.append(c0 + c_geez + "ለ")        # e.g. አከለ
                    cands.append(c0 + c_geez + c_geez)     # e.g. አዘዘ
            elif last_order == 5:
                if o0 == 6:   # Hollow-U: ቆም -> ቆመ, ሆን -> ሆነ, ሮጥ -> ሮጠ
                    cands.append(c0 + self.phonology.shift_order(last_char, 0))
                elif o0 == 1: # Hollow-U Jussive: ቁም -> ቆመ, ሁን -> ሆነ, ሩጥ -> ሮጠ
                    cands.append(self.phonology.shift_order(c0, 6) + self.phonology.shift_order(last_char, 0))
                elif o0 == 4: # Hollow-I: ሄድ -> ሄደ, ሸጥ -> ሸጠ
                    cands.append(c0 + self.phonology.shift_order(last_char, 0))
                elif o0 == 2: # Hollow-I Jussive: ሂድ -> ሄደ, ሽጥ -> ሸጠ
                    cands.append(self.phonology.shift_order(c0, 4) + self.phonology.shift_order(last_char, 0))
                elif o0 == 3: # Hollow-Rabe & Guttural: ጻፍ -> ጻፈ, ቻል -> ቻለ, ላክ -> ላከ, አይ -> አየ, አሽ -> አሸ, አጭ -> አጨ, አዥ -> አዠ
                    cands.append(c0 + self.phonology.shift_order(last_char, 0))
                elif o0 == 5: # Hollow-Rabe Imperfective: ጽፍ -> ጻፈ, ችል -> ቻለ, ልክ -> ላከ
                    cands.append(self.phonology.shift_order(c0, 3) + self.phonology.shift_order(last_char, 0))
                    cands.append(self.phonology.shift_order(c0, 0) + self.phonology.shift_order(last_char, 0))
                elif o0 == 0: # Defective: ቀር -> ቀረ, ተው -> ተወ, መሽ -> መሸ
                    cands.append(c0 + self.phonology.shift_order(last_char, 0))
                    cands.append(c0 + self.phonology.shift_order(last_char, 3))
                elif o0 == 7: # Dikala labiovelar: ጓዝ -> ጓዘ/ተጓዘ, ኳል -> ኳለ, ሟጭ -> ሟጨ
                    base_geez = c0 + self.phonology.shift_order(last_char, 0)
                    cands.append(base_geez)
                    cands.append("ተ" + base_geez)
            elif o0 == 5 and last_order == 3: # Final-Weak Jussive: ስማ -> ሰማ, ምጣ -> መጣ, ግባ -> ገባ
                cands.append(self.phonology.shift_order(c0, 0) + last_char)
            elif o0 == 1 and last_order == 3: # Hollow Weak Jussive: ሙላ -> ሞላ, ጉዳ -> ጎዳ
                cands.append(self.phonology.shift_order(c0, 6) + last_char)

        # -------------------------------------------------------------
        # 2. 3-Character Triradical Stems
        # -------------------------------------------------------------
        elif len(stem) == 3:
            penult_c = stem[1]
            _, penult_o = self.phonology.get_char_info(penult_c)

            if last_order in (0, 3):
                cands.append(stem)
            elif last_order == 5:
                # Dikala reciprocal/frequentative (e.g. ጯጯህ -> ተጯጯሀ / ተጯጯኸ)
                if o0 == 7 and penult_o == 7:
                    cands.append("ተ" + c0 + penult_c + self.phonology.shift_order(last_char, 0))
                elif o0 == 0 and penult_o == 5:
                    # Type A/B Imperfective/Coverb: ሰብር -> ሰበረ, ደንቅ -> ደነቀ, ነገረ
                    cands.append(c0 + self.phonology.shift_order(penult_c, 0) + self.phonology.shift_order(last_char, 0))
                elif o0 == 5 and penult_o == 0:
                    # Type A Jussive: ስበር -> ሰበረ, ድነቅ -> ደነቀ, ምሰል -> መሰለ
                    cands.append(self.phonology.shift_order(c0, 0) + penult_c + self.phonology.shift_order(last_char, 0))
                elif o0 == 3 and penult_o == 5:
                    # Type C Imperfective/Jussive: ማርክ -> ማረከ, ጋብዝ -> ጋበዘ, ባርክ -> ባረከ
                    cands.append(c0 + self.phonology.shift_order(penult_c, 0) + self.phonology.shift_order(last_char, 0))
                elif o0 == 6 and penult_o == 5:
                    # Sabe Triradical: ሞክር -> ሞከረ, ቆልፍ -> ቆለፈ, ቆጥር -> ቆጠረ
                    cands.append(c0 + self.phonology.shift_order(penult_c, 0) + self.phonology.shift_order(last_char, 0))
                else:
                    cands.append(self.phonology.shift_order(c0, 0) + self.phonology.shift_order(penult_c, 0) + self.phonology.shift_order(last_char, 0))
                    # Final-weak citation (e.g. ተበድ -> ተበዳ, ተሰም -> ተሰማ, አመጥ -> አመጣ)
                    if c0 in ("ተ", "አ", "የ", "በ", "ከ", "ለ"):
                        cands.append(c0 + self.phonology.shift_order(penult_c, 0) + self.phonology.shift_order(last_char, 3))

        # -------------------------------------------------------------
        # 3. 4-Character Quadriradical & Reduplicative Stems
        # -------------------------------------------------------------
        elif len(stem) == 4:
            if last_order in (0, 3):
                cands.append(stem)
            elif last_order == 5:
                # Never treat inflectional subject concord prefixes (ይ, ት, እ) as radical C1 of a quadriradical
                if stem[0] in ("ይ", "ት", "እ") and o0 == 5:
                    return tuple(dict.fromkeys(cands))

                c1_val, c2_val, c3_val, c4_val = stem[0], stem[1], stem[2], stem[3]
                _, o1_val = self.phonology.get_char_info(c1_val)
                _, o2_val = self.phonology.get_char_info(c2_val)
                _, o3_val = self.phonology.get_char_info(c3_val)

                if o1_val == 0 and o2_val == 5 and (o3_val in (0, 5)):
                    # 1-2-1-2 Reduplicative Command/Coverb (0,5,0,5 / 0,5,5,5): ቀልቅል -> ቀለቀለ, ቀላቀለ | ለስልስ -> ለሰለሰ | ከስክስ -> ከሰከሰ | ለምልም -> ለመለመ
                    c2_geez = self.phonology.shift_order(c2_val, 0)
                    c3_geez = self.phonology.shift_order(c3_val, 0) if o3_val == 5 else c3_val
                    c4_geez = self.phonology.shift_order(c4_val, 0)
                    cands.append(c1_val + c2_geez + c3_geez + c4_geez)
                    cands.append(c1_val + self.phonology.shift_order(c2_val, 3) + c3_geez + c4_geez)  # e.g. ቀላቀለ
                elif o1_val == 0 and o2_val == 3 and o3_val == 0:
                    # Type C quadriradical with a retained Rabe medial radical:
                    # ጠቃቀም -> ጠቃቀመ (do not flatten ቃ to ቀ).
                    cands.append(c1_val + c2_val + c3_val + self.phonology.shift_order(c4_val, 0))
                elif o1_val == 0 and o2_val == 5 and o3_val == 5:
                    # Type 1 Quadriradical Jussive (0,5,5,5): መስርት -> መሰረተ, ከልክል -> ከለከለ, ደንግጥ -> ደነገጠ
                    cands.append(c1_val + self.phonology.shift_order(c2_val, 0) + self.phonology.shift_order(c3_val, 0) + self.phonology.shift_order(c4_val, 0))
                elif o1_val == 0 and o2_val == 0 and o3_val == 5:
                    # Type 1 Quadriradical Contingent (0,0,5,5): መሰርት -> መሰረተ, ከለክል -> ከለከለ, ደነግጥ -> ደነገጠ
                    cands.append(c1_val + c2_val + self.phonology.shift_order(c3_val, 0) + self.phonology.shift_order(c4_val, 0))
                else:
                    cands.append(c1_val + self.phonology.shift_order(c2_val, 0) + self.phonology.shift_order(c3_val, 0) + self.phonology.shift_order(c4_val, 0))

        # -------------------------------------------------------------
        # 4. 5- and 6-Character Derived, Frequentative & Sabe Stems
        # -------------------------------------------------------------
        elif len(stem) >= 5:
            if last_order in (0, 3, 7):
                cands.append(stem)
            elif last_order == 5:
                penult_c = stem[-2]
                _, penult_o = self.phonology.get_char_info(penult_c)

                # Family 4: Sabe reduplicatives (e.g. አሽቆልቁል -> አሽቆለቆለ, አንጎድጉድ -> አንጎደጎደ)
                if len(stem) == 6 and stem[2] == stem[4] and penult_o in (1, 6):
                    c_sabe = stem[2]
                    c_geez = self.phonology.shift_order(stem[3], 0)
                    base_cit = stem[:2] + c_sabe + c_geez + c_sabe + c_geez
                    cands.append(base_cit)

                # Family 2: 1-2-1-2 Inceptive reduplicatives (e.g. አንጠልጥል -> አንጠለጠለ, አንቀጥቅጥ -> አንቀጠቀጠ)
                elif len(stem) == 6 and stem[2] == stem[4] and stem[3] == stem[5]:
                    c1_g = self.phonology.shift_order(stem[2], 0)
                    c2_g = self.phonology.shift_order(stem[3], 0)
                    base_cit = stem[:2] + c1_g + c2_g + c1_g + c2_g
                    cands.append(base_cit)

                # Family 1 & 3: Frequentatives & Intensive Type C (e.g. አንዣብብ -> አንዣበበ, አሰባብር -> አሰባበረ, ተመላለስ -> ተመላለሰ)
                else:
                    c_final_geez = self.phonology.shift_order(last_char, 0)
                    if penult_o == 5:
                        c_penult_geez = self.phonology.shift_order(penult_c, 0)
                        base_cit = stem[:-2] + c_penult_geez + c_final_geez
                    else:
                        base_cit = stem[:-1] + c_final_geez
                    cands.append(base_cit)

            elif last_order == 1:
                # Plural Ka'eb (e.g. ተሰጣጡ -> ተሰጣጣ, ተቆያዩ -> ተቆያየ)
                cands.append(stem[:-1] + self.phonology.shift_order(last_char, 3))
                cands.append(stem[:-1] + self.phonology.shift_order(last_char, 0))

            # Voice parity (አ- <-> ተ-) for 5/6-radical stems
            for c in list(cands):
                if c.startswith("አ"):
                    cands.append("ተ" + c[1:])
                elif c.startswith("ተ"):
                    cands.append("አ" + c[1:])

        # Passive citation fallback for derived roots (ተመሰለ, ተቀበለ)
        if len(stem) >= 3 and not stem.startswith("ተ"):
            cands.append("ተ" + cands[0] if cands else "ተ" + stem)

        res = tuple(dict.fromkeys(cands))
        self.cache[cache_key] = res
        return res

    def _patterns_for_shape(self, shape_name: str) -> Set[Tuple[int, ...]]:
        """Return the canonical vocalic templates for one named stem family."""
        pattern_map = getattr(self.morphology, "VERBAL_RADICAL_PATTERNS", {})
        for family, patterns in pattern_map.items():
            if getattr(family, "value", family) == shape_name:
                result = set(patterns)
                if shape_name == "SHAPE_A_PERFECTIVE":
                    # Shape A also contains perfective/jussive syncretic forms
                    # such as ለካ and ሰማ, partitioned outside its exclusive
                    # template set in the morphology inventory.
                    result.update(
                        getattr(
                            self.morphology,
                            "SYNCRETIC_PERFECTIVE_JUSSIVE_PATTERNS",
                            (),
                        )
                    )
                    # This biradical template is listed in the project’s
                    # perfective inventory but remains in the C/D syncretic
                    # partitions of the JSON template file.
                    result.add((0, 3))
                return result

        # Keep injected morphology implementations compatible if they expose
        # the family sets but not the dispatcher mapping.
        fallback_attributes = {
            "SHAPE_A_PERFECTIVE": "LESLAU_PERFECTIVE_PATTERNS",
            "SHAPE_B_COVERB": "LESLAU_COVERB_PATTERNS",
            "SHAPE_C_IMPERFECTIVE": "LESLAU_IMPERFECTIVE_PATTERNS",
            "SHAPE_D_JUSSIVE_IMPERATIVE": "LESLAU_JUSSIVE_PATTERNS",
            "SHAPE_E_INFINITIVE": "LESLAU_INFINITIVE_PATTERNS",
        }
        result = set(
            getattr(self.morphology, fallback_attributes.get(shape_name, ""), ())
        )
        if shape_name == "SHAPE_A_PERFECTIVE":
            result.update(
                getattr(
                    self.morphology,
                    "SYNCRETIC_PERFECTIVE_JUSSIVE_PATTERNS",
                    (),
                )
            )
            result.add((0, 3))
        return result

    def _orders(self, stem: str) -> Tuple[int, ...]:
        """Get the vocalic order tuple used by the canonical shape templates."""
        return tuple(self.phonology.get_char_info(char)[1] for char in stem)

    def _matches_shape_template(self, stem: str, shape_name: str) -> bool:
        """Test a stem against the radical-pattern family for its known shape."""
        return bool(stem) and self._orders(stem) in self._patterns_for_shape(shape_name)

    def _lemmatize_shape_a(self, stem: str) -> Tuple[str, ...]:
        """Invert Shape A by retaining a recognized perfective citation grade."""
        perfective_name = "SHAPE_A_PERFECTIVE"
        if stem.startswith("የ") and self._matches_shape_template(stem[1:], perfective_name):
            stem = stem[1:]

        if self._matches_shape_template(stem, perfective_name):
            # Preserve generic voice alternatives; the shape determines the
            # citation grade, not whether the arbitrator may later retain a passive lemma.
            return self._citation_grade_candidates(stem)
        return self.generate_candidates(stem)

    def _lemmatize_shape_c(self, stem: str) -> Tuple[str, ...]:
        """Invert Shape C from the imperfective-family stem pattern."""
        if not self._matches_shape_template(stem, "SHAPE_C_IMPERFECTIVE"):
            return self.generate_candidates(stem)

        # Final-weak imperfectives that share a canonical A-grade pattern are
        # already citation-shaped; otherwise the generic coda/medial rules
        # convert the imperfective grade to its perfective citation.
        orders = self._orders(stem)
        if (
            orders
            and orders[-1] in (0, 3)
            and self._matches_shape_template(stem, "SHAPE_A_PERFECTIVE")
        ):
            return self._citation_grade_candidates(stem)
        return self._prioritize_perfective_family(self.generate_candidates(stem))

    def _prioritize_perfective_family(
        self, candidates: Tuple[str, ...]
    ) -> Tuple[str, ...]:
        """Put citations licensed by the Shape-A family before generic fallbacks."""
        perfective: List[str] = []
        other: List[str] = []
        for candidate in candidates:
            base = candidate
            while base.startswith("ተ"):
                base = base[1:]
            destination = (
                perfective
                if self._matches_shape_template(base, "SHAPE_A_PERFECTIVE")
                else other
            )
            destination.append(candidate)
        return tuple(dict.fromkeys((*perfective, *other)))

    def _lemmatize_shape_d(self, stem: str) -> Tuple[str, ...]:
        """Invert Shape D's jussive/imperative grade to a citation family."""
        if not self._matches_shape_template(stem, "SHAPE_D_JUSSIVE_IMPERATIVE"):
            return self.generate_candidates(stem)

        orders = self._orders(stem)
        if (
            orders
            and orders[-1] in (0, 3)
            and self._matches_shape_template(stem, "SHAPE_A_PERFECTIVE")
        ):
            return self._citation_grade_candidates(stem)

        # Defective final-weak commands have a dedicated first-radical grade
        # inversion (ስማ→ሰማ, ሙላ→ሞላ), including cases the generic branch sees as
        # already vocalic because its final order is 3.
        if len(stem) == 2 and orders[-1] == 3:
            if orders[0] == 5:
                citation = self.phonology.shift_order(stem[0], 0) + stem[1]
                return self._citation_grade_candidates(citation)
            if orders[0] == 1:
                citation = self.phonology.shift_order(stem[0], 6) + stem[1]
                return self._citation_grade_candidates(citation)

        # Causative commands preserve their አ- derivation but change its first
        # radical from the command's C5 grade to the perfective C0 grade
        # (አውጣ→አወጣ, አስጣ→አሰጣ).
        if (
            stem.startswith("አ")
            and len(stem) >= 3
            and orders[1] == 5
            and orders[-1] in (0, 3)
        ):
            citation = stem[0] + self.phonology.shift_order(stem[1], 0) + stem[2:]
            if (
                self._matches_shape_template(citation, "SHAPE_A_PERFECTIVE")
                or self._matches_shape_template(citation, "SHAPE_D_JUSSIVE_IMPERATIVE")
            ):
                return self._citation_grade_candidates(citation)

        return self._prioritize_perfective_family(self.generate_candidates(stem))

    def _citation_grade_candidates(self, citation: str) -> Tuple[str, ...]:
        """Keep the inverted citation first, then its rule-based voice variants."""
        candidates = [citation]
        candidates.extend(self.generate_candidates(citation))
        if not citation.startswith(("ተ", "አ")) and len(citation) >= 2:
            candidates.append("ተ" + citation)
        return tuple(dict.fromkeys(candidates))

    def _lemmatize_shape_e(self, stem: str) -> Tuple[str, ...]:
        """Invert Shape E infinitives by removing their መ-/ማ- family markers."""
        if len(stem) < 3 or not stem.startswith(("መ", "ማ")):
            return self.generate_candidates(stem)

        candidates: List[str] = []
        orders = self._orders(stem)
        is_known_infinitive_pattern = self._matches_shape_template(
            stem, "SHAPE_E_INFINITIVE"
        )

        if stem.startswith("መ"):
            core = stem[1:]
            candidates.extend(self.generate_candidates(core))

            # Keep the vowel grade of the exposed root family while changing
            # only its final C5 citation coda. This captures families such as
            # መማለድ→ማለደ and preserves medial Sabe in መለጐም→ለጐመ.
            if len(core) >= 2:
                coda_candidate = (
                    core[:-1] + self.phonology.shift_order(core[-1], 0)
                )
                candidates.extend(self._citation_grade_candidates(coda_candidate))

                # Hollow-U roots retain their order-6 onset in the perfective
                # citation (መቁረጥ→ቈረጠ, መቁሰል→ቈሰለ).
                if orders[1] == 1 and len(core) >= 3:
                    hollow_u = (
                        self.phonology.shift_order(core[0], 6)
                        + core[1:-1]
                        + self.phonology.shift_order(core[-1], 0)
                    )
                    if self._matches_shape_template(
                        hollow_u, "SHAPE_A_PERFECTIVE"
                    ):
                        candidates.extend(self._citation_grade_candidates(hollow_u))

            # A final -ት is a verbal-noun ending only when deleting it exposes
            # a known perfective or jussive stem family. Keep the Shape-D
            # inversion for command-grade exposed stems (መስማት→ሰማ).
            if core.endswith("ት"):
                exposed = core[:-1]
                exposed_orders = self._orders(exposed)
                if self._matches_shape_template(exposed, "SHAPE_A_PERFECTIVE"):
                    candidates.extend(self._citation_grade_candidates(exposed))
                elif self._matches_shape_template(
                    exposed, "SHAPE_D_JUSSIVE_IMPERATIVE"
                ):
                    candidates.extend(self._lemmatize_shape_d(exposed))
                elif len(exposed) >= 2 and exposed_orders[0] == 5:
                    # Some መ- infinitives expose a Shape-A stem with the first
                    # radical still in its C5 grade (መስጠት→ሰጠ,
                    # መቅረት→ቀረ). Shift only that initial radical.
                    citation = (
                        self.phonology.shift_order(exposed[0], 0) + exposed[1:]
                    )
                    if self._matches_shape_template(
                        citation, "SHAPE_A_PERFECTIVE"
                    ):
                        candidates.extend(self._citation_grade_candidates(citation))
                elif len(exposed) >= 2 and exposed_orders[0] == 1:
                    # Hollow-U infinitives preserve the order-6 root onset
                    # (መቍረጥ→ቈረጠ, መቍሰል→ቈሰለ).
                    citation = (
                        self.phonology.shift_order(exposed[0], 6) + exposed[1:]
                    )
                    if self._matches_shape_template(
                        citation, "SHAPE_A_PERFECTIVE"
                    ):
                        candidates.extend(self._citation_grade_candidates(citation))
                elif len(exposed) == 1:
                    # A few biradical infinitives expose an initial derived
                    # vowel that is absent from the nominalized surface.
                    citation = "አ" + exposed
                    candidates.extend(self._citation_grade_candidates(citation))

            # Hollow-Sabe infinitives retain their order-6 first radical
            # (መሞከር→ሞከረ); the generic all-C0 fallback would flatten it.
            if (
                is_known_infinitive_pattern
                and orders == (0, 6, 0, 5)
                and len(core) == 3
            ):
                citation = (
                    core[0]
                    + self.phonology.shift_order(core[1], 0)
                    + self.phonology.shift_order(core[2], 0)
                )
                candidates.extend(self._citation_grade_candidates(citation))

        else:
            # ማ- is the coalesced realization of መ- + አ-. Replacing it with
            # the derived perfective onset exposes the full family to the
            # existing inversion rules (ማሰብ→አሰበ, ማድረግ→አደረገ).
            derived_stem = "አ" + stem[1:]
            candidates.extend(self.generate_candidates(derived_stem))

            # Map non-prefix C5 grades in the full ማ- family to the C0
            # perfective grade, but admit the alternative only when its result
            # is licensed by the Shape-A template inventory. This handles
            # internal C5 alternations in forms such as ማመስገን and
            # ማመልከት without disturbing retained ማ- derivation.
            derived_orders = self._orders(derived_stem)
            perfective_grade = "".join(
                char
                if index == 0 or order != 5
                else self.phonology.shift_order(char, 0)
                for index, (char, order) in enumerate(
                    zip(derived_stem, derived_orders)
                )
            )
            if self._matches_shape_template(
                perfective_grade, "SHAPE_A_PERFECTIVE"
            ):
                candidates.extend(
                    self._citation_grade_candidates(perfective_grade)
                )
            if len(derived_stem) >= 3 and derived_orders[1] == 1:
                hollow_u = (
                    derived_stem[0]
                    + self.phonology.shift_order(derived_stem[1], 6)
                    + derived_stem[2:-1]
                    + self.phonology.shift_order(derived_stem[-1], 0)
                )
                if self._matches_shape_template(
                    hollow_u, "SHAPE_A_PERFECTIVE"
                ):
                    candidates.extend(self._citation_grade_candidates(hollow_u))
            if stem.endswith("ት"):
                exposed = "አ" + stem[1:-1]
                exposed_orders = self._orders(exposed)
                coda_candidate = (
                    exposed + self.phonology.shift_order(stem[-1], 0)
                )
                candidates.extend(self._citation_grade_candidates(coda_candidate))
                if self._matches_shape_template(exposed, "SHAPE_A_PERFECTIVE"):
                    candidates.extend(self._citation_grade_candidates(exposed))
                elif self._matches_shape_template(
                    exposed, "SHAPE_D_JUSSIVE_IMPERATIVE"
                ):
                    candidates.extend(self._lemmatize_shape_d(exposed))

                # Weak/defective derived infinitives expose an A-grade root
                # after changing the first non-prefix C5 radical to C0
                # (ማውጣት→አወጣ, ማግኘት→አገኘ). Require the result to be a
                # licensed perfective template before adding it.
                if (
                    len(exposed) == 3
                    and exposed_orders[0] == 3
                    and exposed_orders[1] == 5
                    and exposed_orders[-1] in (0, 3)
                ):
                    citation = (
                        exposed[0]
                        + self.phonology.shift_order(exposed[1], 0)
                        + exposed[2:]
                    )
                    if (
                        self._matches_shape_template(citation, "SHAPE_A_PERFECTIVE")
                        or self._matches_shape_template(
                            citation, "SHAPE_D_JUSSIVE_IMPERATIVE"
                        )
                    ):
                        candidates.extend(self._citation_grade_candidates(citation))
                elif (
                    len(exposed) >= 4
                    and exposed_orders[0] == 3
                    and exposed_orders[-1] == 3
                    and 5 in exposed_orders[1:-1]
                ):
                    citation = "".join(
                        char
                        if index == 0 or order != 5
                        else self.phonology.shift_order(char, 0)
                        for index, (char, order) in enumerate(
                            zip(exposed, exposed_orders)
                        )
                    )
                    if (
                        self._matches_shape_template(citation, "SHAPE_A_PERFECTIVE")
                        or self._matches_shape_template(
                            citation, "SHAPE_D_JUSSIVE_IMPERATIVE"
                        )
                    ):
                        candidates.extend(self._citation_grade_candidates(citation))
            elif len(stem) >= 3:
                # Boundary coalescence can hide the C5 citation coda in a
                # ማ- infinitive; retain the phonological inversion alongside
                # the full-shape generic candidates.
                coda = self.phonology.shift_order(stem[-1], 0)
                reconstructed = "አ" + stem[1:-1] + coda
                candidates.extend(self.generate_candidates(reconstructed))

        return tuple(dict.fromkeys(candidates))

    def _coverb_shape_alternatives(self, stem: str) -> Tuple[str, ...]:
        """Return rule-based weak bi-radical citation grades for Shape B.

        Some coverbs invert to a two-character C5 base whose citation keeps a
        Rabe vowel or repeats the final consonant (e.g. ላል -> ላላ,
        ለቅ -> ለቀቀ). These are phonological alternatives, not lexicon lookups.
        """
        if len(stem) != 2:
            return ()

        _, first_order = self.phonology.get_char_info(stem[0])
        _, final_order = self.phonology.get_char_info(stem[-1])
        if final_order != 5:
            return ()

        if first_order == 3:
            return (stem[0] + self.phonology.shift_order(stem[-1], 3),)
        if first_order == 0:
            geez_final = self.phonology.shift_order(stem[-1], 0)
            return (stem[0] + geez_final + geez_final,)
        return ()

    def lemmatize(
        self, stem: str, shape: Optional[Any] = None
    ) -> Tuple[str, ...]:
        """Compatibility alias; prefer :meth:`generate_candidates`."""
        return self.generate_candidates(stem, shape=shape)

    def reconstruct_all_verbal_lemmas(
        self, stem: str, shape: Optional[Any] = None
    ) -> Tuple[str, ...]:
        """Compatibility alias for the former reconstruction API."""
        return self.generate_candidates(stem, shape=shape)


# Backward-compatible class names for callers of the former APIs.
Lemmatizer = LemmaCandidateGenerator
LemmaReconstructor = LemmaCandidateGenerator

__all__ = [
    "LemmaCandidateGenerator",
    "Lemmatizer",
    "LemmaReconstructor",
]
