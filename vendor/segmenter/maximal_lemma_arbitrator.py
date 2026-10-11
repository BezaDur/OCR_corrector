"""Corpus-backed arbitration and protection of maximal lemma candidates.

:class:`MaximalLemmaArbitrator` does not derive or assign a canonical lemma.
Its primary role is to protect a valid, larger/unpeeled candidate from
unsupported prefix or suffix peeling, using dictionary attestation and
paradigmatic sister evidence. For verbal checks it may consult the rule-based
candidate generator, then use corpus evidence to assess those candidates.

``LemmaIdentifier`` remains as a backwards-compatible class alias.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from morph_types import VerbalShape
from seg_normalizer import AmharicNormalizer  # VENDORED: renamed to avoid colliding with tess's normalizer.py
from phonology import AmharicBoundaryPhonology


@dataclass(frozen=True)
class CommutationResult:
    is_authentic: bool
    anchor_hits: int
    flexible_hits: int
    total_hits: int
    attested_forms: Tuple[str, ...] = ()
    tested_sisters: Tuple[str, ...] = ()


# Decisions for one unpeeled-parent -> peeled-child edge
PARENT = "PARENT"        # parent is the maximal lemma; the child's peel is rejected
CHILD = "CHILD"          # the peel is proven; the unpeeled parent is rejected
UNDECIDED = "UNDECIDED"  # evidence does not separate them; ranking decides


@dataclass(frozen=True)
class EdgeVerdict:
    """Evidence-backed verdict for one unpeeled parent vs. one peeled child."""
    parent_stem: str
    child_stem: str
    peeled: str
    side: str                 # "prefix" or "suffix"
    family: str               # slot family used for sister commutation
    decision: str             # PARENT / CHILD / UNDECIDED
    reason: str
    parent_verified: bool
    parent_evidence: Tuple[str, ...] = ()  # attested same-slot sisters on the parent
    child_evidence: Tuple[str, ...] = ()   # attested competing sisters on the child


@dataclass
class PoolArbitration:
    """Result of listwise arbitration over a whole candidate pool."""
    survivors: List[Any] = field(default_factory=list)
    eliminated: List[Any] = field(default_factory=list)
    verdicts: List[Tuple[Any, Any, EdgeVerdict]] = field(default_factory=list)
    evidence_rank: Dict[int, int] = field(default_factory=dict)
    root_verified: bool = False


@dataclass(frozen=True)
class VerbalLemmaEvidence:
    """Lexical evidence that a template-matching form is a real verb."""
    word: str
    shapes_tested: Tuple[str, ...]
    citations: Tuple[str, ...]           # rule-generated citation candidates (word excluded)
    attested_citations: Tuple[str, ...]  # citations in verb_stems or the corpus
    cohort_forms: Tuple[str, ...]        # attested verbal cohort forms of the word
    verified: bool
    reason: str


class MaximalLemmaArbitrator:
    """Protect maximal lexical candidates; do not generate canonical lemmas.

    The arbitrator compares unpeeled parents with peeled alternatives and uses
    dictionary frequency plus paradigmatic evidence to decide whether a split is
    licensed. The verbal citation bridge can use rule-generated candidates, but
    arbitration is evidence-based—not lemmatization. If a citation is not
    generated, independent attestation/paradigmatic evidence may still protect
    the maximal candidate; the arbitrator does not infer that citation.
    """

    # Anchor Sisters: Multi-character prefixes with near-zero lexical homograph risk
    PROCLITIC_ANCHORS: Tuple[str, ...] = ("እንደ", "ስለ", "ወደ", "እስከ")

    # Flexible Sisters: Single-character proclitics prone to root homograph collisions
    PROCLITIC_FLEXIBLE: Tuple[str, ...] = ("የ", "በ", "ለ", "ከ", "ያ", "ባ", "ላ", "ካ")

    ALL_PROCLITICS: Tuple[str, ...] = PROCLITIC_ANCHORS + PROCLITIC_FLEXIBLE

    # Declarative Subject Concord Sister Paradigms (reserved for paradigmatic subject probes)
    IMP_SUBJECT_SISTERS: Tuple[str, ...] = ("ይ", "ት", "እ", "እን")
    PERF_SUBJECT_SISTERS: Tuple[str, ...] = ("ሁ", "ክ", "ሽ", "ች", "ን", "አችሁ", "ኡ")

    # Slot families arbitrated by sister commutation
    FAMILY_PRAGMATIC = "pragmatic"          # -ም, -ስ, -ና, -ማ
    FAMILY_OBJECT = "object_clitic"         # -ኝ, -ው, -ን, -አት, -ቸው ...
    FAMILY_DETERMINER = "determiner"        # -ኡ, -ዋ, -ቷ, possessives
    FAMILY_PROCLITIC = "proclitic"          # የ-, በ-, ለ-, ከ-
    FAMILY_DERIVATIONAL = "derivational"    # -ነት, -ኛ, -አዊ
    FAMILY_ROOT_SHIELD = "root_shield"      # coverb / accusative slices off a verified root
    FAMILY_NONE = "none"

    PRAGMATIC_SISTERS: Tuple[str, ...] = ("ም", "ስ", "ና", "ማ")
    DERIVATIONAL_SUFFIXES: Tuple[str, ...] = ("ነት", "ኛ", "አዊ")
    OBJECT_ONLY_SUFFIXES: Tuple[str, ...] = ("ኝ", "አት", "ቸው", "አቸው", "አችሁ")
    DETERMINER_ONLY_SUFFIXES: Tuple[str, ...] = ("ኡ", "ዋ", "ቷ", "ዬ", "ችን", "አችን")
    AMBIGUOUS_OBJ_DET_SUFFIXES: Tuple[str, ...] = ("ው", "ት", "ን", "ህ", "ሽ")
    PROCLITIC_PEEL_TARGETS: Tuple[str, ...] = ("የ", "በ", "ለ", "ከ", "ባ", "ላ", "ካ")
    ROOT_SHIELD_RULES: Tuple[str, ...] = (
        "coverb_3fs_a", "case_accusative", "compound_coverb_", "clitic_"
    )
    # verify_verbal_clitic_commutation has always used 3; both sides use the same bar
    OBJECT_CLITIC_MIN_FREQ: int = 3

    # Missing parent sisters prove a peel only if the parent is frequent enough for
    # sisters to be expected: parent_freq >= ABSENCE_POWER_FACTOR x family threshold.
    ABSENCE_POWER_FACTOR: int = 10

    # Coalesced proclitics restore an initial 'አ' on the host (ባለፈ = በ + አለፈ)
    COALESCED_PROCLITICS: Dict[str, str] = {"ባ": "በ", "ላ": "ለ", "ካ": "ከ", "ያ": "የ"}

    def __init__(
        self,
        vocab_freq: Optional[Dict[str, int]] = None,
        min_freq: int = 10,
        phonology: Optional[AmharicBoundaryPhonology] = None,
        verbal_noun_checker: Optional[Callable[[str], bool]] = None,
        morphology: Optional[Any] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        citation_min_freq: Optional[int] = None,
    ):
        self.vocab_freq: Dict[str, int] = vocab_freq or {}
        self.min_freq: int = min_freq
        self.phonology: AmharicBoundaryPhonology = phonology or AmharicBoundaryPhonology()
        self.normalizer: Callable[[str], str] = (
            normalizer or AmharicNormalizer.normalize_word_strict
        )

        # Defensive attribute retention: always define both attributes
        self.morphology: Optional[Any] = morphology
        self.verbal_noun_checker: Optional[Callable[[str], bool]] = (
            verbal_noun_checker
            or (getattr(morphology, "is_valid_verbal_noun_stem", None) if morphology else None)
        )
        self._verify_memo: Dict[str, bool] = {}
        # Attestation threshold for generated citations (defaults to min_freq)
        self.citation_min_freq: int = (
            citation_min_freq if citation_min_freq is not None else min_freq
        )
        self._verbal_lemma_memo: Dict[Tuple[str, bool, bool], VerbalLemmaEvidence] = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_attested(self, form: str) -> bool:
        if not form:
            return False
        norm_form = self.normalizer(form) if self.normalizer else form
        return self.vocab_freq.get(norm_form, 0) >= self.min_freq

    def _synthesize_prefix_attachment(self, prefix: str, stem: str) -> List[str]:
        """
        Synthesizes surface forms when prefix attaches to stem,
        accounting for vowel coalescence on initial 'አ'.
        """
        if not prefix or not stem:
            return []

        forms = [prefix + stem]

        # Vowel coalescence on initial 'አ' (e.g. ስለ + አለፈ -> ስላለፈ, በ + አለፈ -> ባለፈ)
        if stem.startswith("አ") and len(stem) >= 2:
            last_pfx_char = prefix[-1]
            _, order = self.phonology.get_char_info(last_pfx_char)
            if order == 0:
                coalesced_char = self.phonology.shift_order(last_pfx_char, 3)
                coalesced_prefix = prefix[:-1] + coalesced_char
                forms.append(coalesced_prefix + stem[1:])

        return forms

    # ------------------------------------------------------------------
    # PROBE A: Remnant Commutability Test
    # ------------------------------------------------------------------

    def verify_proclitic_peeling(
        self, prefix: str, remnant: str, min_total_hits: int = 2
    ) -> CommutationResult:
        """
        PROBE A: Remnant Commutability Test.
        Tests if candidate 'remnant' is a genuine base when 'prefix' is peeled.
        Replaces 'prefix' with its sisters.
        Rule: A split is authentic if:
          - At least 1 multi-character anchor sister hits (e.g. እንደ+remnant), OR
          - At least min_total_hits (default 2) total sisters hit.
        """
        if prefix not in self.ALL_PROCLITICS:
            # Not governed by proclitic paradigm, do not block
            return CommutationResult(
                is_authentic=True, anchor_hits=0, flexible_hits=0, total_hits=0
            )

        anchor_hits = 0
        flexible_hits = 0
        attested: List[str] = []
        tested: List[str] = []

        # 1. Test Anchor Sisters
        for sister in self.PROCLITIC_ANCHORS:
            if sister == prefix:
                continue
            tested.append(sister)
            cands = self._synthesize_prefix_attachment(sister, remnant)
            matched = [w for w in cands if self._is_attested(w)]
            if matched:
                anchor_hits += 1
                attested.extend(matched)

        # 2. Test Flexible Sisters
        for sister in self.PROCLITIC_FLEXIBLE:
            if sister == prefix:
                continue
            tested.append(sister)
            cands = self._synthesize_prefix_attachment(sister, remnant)
            matched = [w for w in cands if self._is_attested(w)]
            if matched:
                flexible_hits += 1
                attested.extend(matched)

        total_hits = anchor_hits + flexible_hits
        is_authentic = (anchor_hits >= 1) or (total_hits >= min_total_hits)

        return CommutationResult(
            is_authentic=is_authentic,
            anchor_hits=anchor_hits,
            flexible_hits=flexible_hits,
            total_hits=total_hits,
            attested_forms=tuple(attested),
            tested_sisters=tuple(tested),
        )

    # ------------------------------------------------------------------
    # PROBE B: External Stacking Test (Law of Mutual Exclusion)
    # ------------------------------------------------------------------

    def is_uninflected_base(
        self, candidate_stem: str, min_anchor_hits: int = 1
    ) -> bool:
        """
        PROBE B: External Stacking Test (Law of Mutual Exclusion).
        Tests whether 'candidate_stem' is a clean uninflected base (like 'በላይ')
        or a dirty stem harboring an unpeeled proclitic (like 'የበለጸገ' or 'በሰው').
        """
        if not candidate_stem or len(candidate_stem) < 2:
            return False

        # Only check stems that begin with an ambiguous proclitic
        has_initial_proclitic = any(
            candidate_stem.startswith(p) for p in self.PROCLITIC_FLEXIBLE
        )
        if not has_initial_proclitic:
            return True

        anchor_hits = 0
        for sister in self.PROCLITIC_ANCHORS:
            cands = self._synthesize_prefix_attachment(sister, candidate_stem)
            if any(self._is_attested(w) for w in cands):
                anchor_hits += 1

        return anchor_hits >= min_anchor_hits

    # ------------------------------------------------------------------
    # PROBE C: Suffix Sister Commutation Test
    # ------------------------------------------------------------------

    def verify_suffix_commutation(
        self, suffix: str, remnant_stem: str, min_hits: int = 1
    ) -> bool:
        """
        PROBE C: Suffix Sister Commutation Test.
        Tests if a proposed remnant stem (e.g. '*ሁል' when peeling '-ኡ') is a genuine base
        capable of taking sister suffixes from the nominal paradigm.
        If 0 sister forms (plural, feminine, possessive) exist in the vocabulary,
        the proposed stem is rejected as a mutilated fragment.
        """
        if not remnant_stem or len(remnant_stem) < 2:
            return False

        last_c = remnant_stem[-1]
        _, last_order = self.phonology.get_char_info(last_c)

        test_forms: List[str] = []

        if last_order == 5:
            shifted_o = self.phonology.shift_order(last_c, 6)
            shifted_a = self.phonology.shift_order(last_c, 3)
            test_forms.append(remnant_stem[:-1] + shifted_o + "ች")   # e.g. ቤቶች, ልጆች
            test_forms.append(remnant_stem[:-1] + shifted_a + "ችን")  # e.g. ቤታችን, ልጃችን
            test_forms.append(remnant_stem[:-1] + "ቷ")               # e.g. ቤቷ (fused 7th-order)
            # NOTE: 'remnant_stem + "ዋ"' removed — phonologically invalid for 5th-order stems.
            # The correct feminine definite is the fused 7th-order form (ቷ), handled above.
        else:
            test_forms.append(remnant_stem + "ዎች")
            test_forms.append(remnant_stem + "ችን")
            test_forms.append(remnant_stem + "ዋ")

        hits = sum(1 for form in test_forms if self._is_attested(form))
        return hits >= min_hits

    # ------------------------------------------------------------------
    # SLOT 2: Pragmatic Enclitic Sister Commutation
    # ------------------------------------------------------------------

    def verify_pragmatic_commutation(
        self, remnant_stem: str, peeled_suffix: str = "", min_hits: int = 1
    ) -> bool:
        """
        SLOT 2: Non-Circular Pragmatic Enclitic Sister Commutation Test (-ም, -ስ, -ና, -ማ).
        Sisters occupying this slot never stack (*ታህሳምስ, *ዮሀንስም).
        Remnant R must commute with COMPETING sisters (strictly excluding peeled_suffix).
        e.g., when peeling -ስ from ታህሳስ, tests whether ታህሳ takes -ም, -ና, -ማ (0 hits).
        """
        if not remnant_stem or len(remnant_stem) < 2:
            return False

        competing_sisters = [c for c in ("ም", "ስ", "ና", "ማ") if c != peeled_suffix]
        test_forms = [remnant_stem + c for c in competing_sisters]
        hits = sum(1 for form in test_forms if self._is_attested(form))
        return hits >= min_hits

    # ------------------------------------------------------------------
    # SLOT 3: Verbal Object Clitic Sister Commutation
    # ------------------------------------------------------------------

    def verify_verbal_clitic_commutation(
        self, remnant_stem: str, min_hits: int = 1, min_freq: int = 3
    ) -> bool:
        """
        SLOT 3: Verbal Object Clitic Sister Commutation Test (-ኝ, -ው, -አት, -ቸው, -ን).
        Sisters occupying this slot never stack (*ሰበረውት).
        If a verbal object clitic is peeled from W -> R + Clitic (e.g. መጋቢት -> መጋቢ + ት),
        remnant R must prove it is an active finite verb by commuting with sister object clitics.
        If 0 sisters exist (*መጋቢኝ, *መጋቢአት), the split is rejected as an accidental homograph match.
        """
        if not remnant_stem or len(remnant_stem) < 2:
            return False

        last_c = remnant_stem[-1]
        _, last_order = self.phonology.get_char_info(last_c)

        test_forms: List[str] = [
            remnant_stem + "ኝ",
            remnant_stem + "ው",
            remnant_stem + "ን",
            remnant_stem + "አት",
            remnant_stem + "ቸው",
        ]

        if last_order == 0 and len(remnant_stem) >= 3:
            shifted_a = self.phonology.shift_order(last_c, 3)
            test_forms.append(remnant_stem[:-1] + shifted_a + "ት")
            test_forms.append(remnant_stem[:-1] + shifted_a + "ቸው")
        elif last_order == 5 and len(remnant_stem) >= 2:
            # On 5th-order codas, object clitics take Geez liaison (-äw, -äññ, -än, -äh) or Rabe (-at, -accäw)
            shifted_geez = self.phonology.shift_order(last_c, 0)
            shifted_rabe = self.phonology.shift_order(last_c, 3)
            test_forms.append(remnant_stem[:-1] + shifted_geez + "ው")
            test_forms.append(remnant_stem[:-1] + shifted_geez + "ኝ")
            test_forms.append(remnant_stem[:-1] + shifted_geez + "ን")
            test_forms.append(remnant_stem[:-1] + shifted_geez + "ህ")
            test_forms.append(remnant_stem[:-1] + shifted_rabe + "ት")
            test_forms.append(remnant_stem[:-1] + shifted_rabe + "ቸው")

        hits = sum(1 for form in test_forms if self.vocab_freq.get(self.normalizer(form) if self.normalizer else form, 0) >= min_freq)
        return hits >= min_hits

    # ------------------------------------------------------------------
    # SLOT 1: Determiner / Possessive Sister Commutation
    # ------------------------------------------------------------------

    def verify_determiner_commutation(
        self, remnant_stem: str, min_hits: int = 1
    ) -> bool:
        """
        SLOT 1: Determiner / Possessive Sister Commutation Test.
        Delegates to verify_suffix_commutation to ensure nominal base R
        commutes with sister possessives (-ችን, -ዋ, -ቷ) and plurals (-ኦች).
        """
        return self.verify_suffix_commutation(
            suffix="ኡ", remnant_stem=remnant_stem, min_hits=min_hits
        )

    # ------------------------------------------------------------------
    # Anchor hit counting
    # ------------------------------------------------------------------

    def count_anchor_hits(self, text: str) -> int:
        """
        Tests 'text' against the multi-character anchor sisters (እንደ-, ስለ-, ወደ-, እስከ-).
        Returns the count of attested forms in vocab_freq.
        """
        if not text or len(text) < 2:
            return 0

        hits = 0
        for sister in self.PROCLITIC_ANCHORS:
            cands = self._synthesize_prefix_attachment(sister, text)
            if any(self._is_attested(w) for w in cands):
                hits += 1
        return hits

    # ------------------------------------------------------------------
    # Maximal-Candidate Lexical Evidence Gate
    # ------------------------------------------------------------------

    @property
    def fused_subject_prefixes(self) -> Tuple[str, ...]:
        """Subject-concord markers that a candidate stem can have absorbed.

        Only the imperfective subject markers sit immediately before the
        radical, so they are the only affixes candidate generation can leave
        fused inside a stem (ይስብ for ይስባል = ይ + ስብ + አል).  Outer markers --
        the relative ሚ/ም, the subordinates ስ/ቢ/ሊ/ሲ, the prepositions -- are
        never absorbed, so offering them as a "fused" split would peel a
        root's own first radical (ምበረከክ, ስጌጥ).

        The inventory comes from the shipped morpheme tables; እ/ኢ are left out
        because the prosthetic and initial-weak readings are decided by their
        own branch in :meth:`is_autonomous_lemma`.
        """
        morphology = self.morphology
        if morphology is None:
            return ()

        prefixes: Set[str] = set(getattr(morphology, "prefix_verb_iv", ()) or ())
        prefixes.update(
            getattr(morphology, "prefix_rel_imperfective_agreement", ()) or ()
        )
        prefixes.difference_update({"እ", "ኢ"})
        return tuple(sorted(prefixes, key=len, reverse=True))

    def split_fused_inflectional_prefix(
        self, stem: str
    ) -> Optional[Tuple[str, str]]:
        """Split a leading subject-concord prefix off a candidate substrate.

        Candidate generation can leave the inflectional prefix inside the
        stem: ``ይስባል`` is offered as stem ``ይስብ`` + auxiliary ``አል``.  The
        buried prefix hides the imperfective pattern from the affix-concord
        lock, so the candidate degrades to a bare ``VB``.  A stem that cannot
        stand as a maximal lemma but begins with a subject-concord marker is
        exactly that case; return ``(prefix, remainder)``, longest prefix
        first, or ``None`` when the stem is a maximal lemma or carries no such
        marker.

        This is the arbitration hook that separates "አል is the auxiliary" from
        "አል is part of the lexical item": the caller keeps the split only when
        the remainder licenses an imperfective shape (Shape C), which is the
        evidence that the substrate really is an imperfective base.
        """
        if not stem or len(stem) < 2 or self.is_autonomous_lemma(stem):
            return None
        for prefix in self.fused_subject_prefixes:
            if not stem.startswith(prefix):
                continue
            remainder = stem[len(prefix):]
            # A one-character remainder is legitimate for the defective
            # radicals (ይላል = ይ + ላ + አል); the caller's shape test is what
            # decides, so the floor here only stops the prefix swallowing
            # the whole stem.
            if remainder:
                return prefix, remainder
        return None

    def is_autonomous_lemma(self, word: str) -> bool:
        """Check whether ``word`` is safe to retain as a maximal base candidate.

        The historical method name ``is_autonomous_lemma`` should not be read
        as a lemmatizer API: this is the arbitrator's lexical evidence gate, not
        a rule that derives the canonical lemma. Verbal checks may ask the
        candidate generator for citation forms and verify them against corpus
        evidence.
        """
        if not word or len(word) < 2:
            return False

        # 1. Canonical Verbal Noun / Infinitive Citation Template
        if self.verbal_noun_checker and self.verbal_noun_checker(word):
            return True
        if self.morphology and hasattr(self.morphology, "is_valid_verbal_noun_stem") and self.morphology.is_valid_verbal_noun_stem(word):
            return True

         # 2. Canonical Verbal Citation Lemma (Perfective, Imperative, Jussive, Reduplicative)
        # An autonomous lemma CANNOT harbor inflectional subject prefixes or subordinators
        inflectional_prefixes = (
            "ይ", "ት", "እን", "ሲ", "ቢ", "ሊ", "እንድ", "እንዲ",
            "እስከ", "እስክ", "እስኪ", "ስለ", "እንደ", "የ"
        )
        if any(word.startswith(pfx) for pfx in inflectional_prefixes):
            return False

        # Prefix 'እ' blocks autonomous lemmahood UNLESS it is a prosthetic or initial-weak command
        if word.startswith("እ"):
            orders_w = tuple(self.phonology.get_vocalic_orders(word))
            is_initial_weak_cmd = (len(word) == 3 and orders_w in ((5, 0, 5), (5, 5)))
            is_prosthetic_cmd = (
                (len(word) == 3 and orders_w in ((5, 5, 3), (5, 5, 5)))
                or (len(word) == 4 and orders_w == (5, 5, 0, 5))
            )
            if not (is_initial_weak_cmd or is_prosthetic_cmd):
                return False

        if self.morphology:
            orders = tuple(self.phonology.get_vocalic_orders(word))
            is_juss_imp = (
                orders in getattr(self.morphology, "LESLAU_JUSSIVE_PATTERNS", set())
                or self.morphology.is_valid_jussive_stem(word)
                or word in getattr(self.morphology, "biradical_type_b_verbs", set())
            )
            is_perf = self.morphology.is_valid_perfective_template(word)
            is_redup = (
                self.morphology.is_reduplicative_quadriradical(word)
                or (len(word) == 4 and word[0] == word[2] and word[1] == word[3])
            )
            if word in getattr(self.morphology, "verb_stems", set()):
                return True
            if is_juss_imp or is_perf or is_redup:
                # A template match is necessary but not sufficient: without a corpus
                # there is no evidence to check, so keep the template behavior.
                if not self.vocab_freq:
                    return True
                # Citation bridge: the form must map to an attested verb.
                if self.verify_verbal_lemma(
                    word, jussive=is_juss_imp, perfective=is_perf
                ).verified:
                    return True
                # Template-only match: fall through to the nominal tests below.

        # Exclude overt inflectional/pragmatic clitics and plural endings
        if word.endswith(("ኦች", "ዎች", "ቶች", "ሎች", "ኝ")):
            return False

        # 3. Outer anchor prefix test on uninflected roots
        if self.count_anchor_hits(word) > 0:
            return True

        # 4. Orthogonal Nominal Sister Commutation Test
        has_plural = any(self._is_attested(word + pl) for pl in ("ዎች", "ኦች"))
        has_poss = any(self._is_attested(word + poss) for poss in ("ችን", "ዬ", "ዎት"))
        has_det_acc = self._is_attested(word + "ን") and self._is_attested(word + "ው")

        if has_plural or has_poss or has_det_acc:
            return True

        return False

    def verify_verbal_lemma(
        self, word: str, jussive: bool = False, perfective: bool = False
    ) -> VerbalLemmaEvidence:
        """Use lexical evidence to support verbal-candidate arbitration.

        A template-matching form (e.g. the command 'በጥስ') is supported as a
        verb only by evidence such as a rule-generated citation candidate that
        is in ``verb_stems``/the corpus, or attested verbal cohort forms of the
        word itself. The word itself never counts as evidence for itself. This
        bridge does not make the arbitrator a lemma generator or make it choose a canonical lemma.
        """
        key = (word, jussive, perfective)
        if key in self._verbal_lemma_memo:
            return self._verbal_lemma_memo[key]
        if len(self._verbal_lemma_memo) > 8192:
            self._verbal_lemma_memo.clear()

        m = self.morphology
        if not m or not word:
            ev = VerbalLemmaEvidence(word, (), (), (), (), False, "no morphology engine")
            self._verbal_lemma_memo[key] = ev
            return ev

        # Shape-directed reconstruction; the general path only when no shape is known
        shapes: List[Optional[VerbalShape]] = []
        if jussive:
            shapes.append(VerbalShape.SHAPE_D_JUSSIVE_IMPERATIVE)
        if perfective:
            shapes.append(VerbalShape.SHAPE_A_PERFECTIVE)
        if not shapes:
            shapes.append(None)

        generate_candidates = getattr(m, "generate_lemma_candidates", None)
        if not callable(generate_candidates):
            generator = getattr(m, "lemma_candidate_generator", None)
            generate_candidates = getattr(generator, "generate_candidates", None)
        if not callable(generate_candidates):
            # Compatibility with morphology providers exposing earlier APIs.
            generate_candidates = getattr(m, "lemmatize", None)
        if not callable(generate_candidates):
            generate_candidates = getattr(m, "reconstruct_all_verbal_lemmas", None)

        citations: List[str] = []
        if callable(generate_candidates):
            for shape in shapes:
                try:
                    found = generate_candidates(word, shape=shape) or []
                except (TypeError, ValueError, KeyError, IndexError):
                    found = []
                for c in found:
                    if c and c != word and c not in citations:
                        citations.append(c)

        verb_stems = getattr(m, "verb_stems", set())
        attested_citations = tuple(
            c for c in citations
            if c in verb_stems
            or self.vocab_freq.get(self._n(c), 0) >= self.citation_min_freq
        )

        cohort_forms: Tuple[str, ...] = ()
        gen = getattr(m, "generate_verbal_cohort_bundles", None)
        if gen:
            try:
                forms = [f for bundle in gen(word) for f in bundle]
            except (TypeError, ValueError, KeyError, IndexError):
                forms = []
            cohort_forms = self._attested_forms(forms, self.min_freq, exclude=(word,))

        if attested_citations:
            reason = f"generated citation candidate(s) attested: {list(attested_citations)}"
        elif cohort_forms:
            reason = f"{len(cohort_forms)} attested verbal cohort form(s)"
        elif citations:
            reason = f"generated citation candidates {citations} are not attested"
        else:
            reason = "no citation candidate generated and no verbal cohort attested"

        ev = VerbalLemmaEvidence(
            word=word,
            shapes_tested=tuple(
                getattr(s, "name", str(s)) if s is not None else "GENERAL" for s in shapes
            ),
            citations=tuple(citations),
            attested_citations=attested_citations,
            cohort_forms=cohort_forms,
            verified=bool(attested_citations or cohort_forms),
            reason=reason,
        )
        self._verbal_lemma_memo[key] = ev
        return ev

    def arbitrate_suffix_lemmas(
        self, unpeeled_candidate: str, peeled_lemma: str, peeled_suffix: str
    ) -> str:
        """
        Pairwise suffix arbitration (e.g. 'በጥስ' vs 'በጥ' + '-ስ', 'ነበረው' vs 'ነበረ' + '-ው').
        Returns the winning string. Use arbitrate_suffix_pair() for the full verdict.
        """
        verdict = self.arbitrate_suffix_pair(unpeeled_candidate, peeled_lemma, peeled_suffix)
        return self._verdict_to_winner(verdict, unpeeled_candidate, peeled_lemma)

    def arbitrate_suffix_pair(
        self, unpeeled_candidate: str, peeled_lemma: str, peeled_suffix: str
    ) -> EdgeVerdict:
        """Evidence verdict for an unpeeled form vs. its suffix-peeled host."""
        families = self._families_for_suffix(peeled_suffix)
        if not families:
            return self._undecided(
                unpeeled_candidate, peeled_lemma, peeled_suffix, "suffix",
                "no sister paradigm defined for this suffix",
            )
        verdicts = [
            self._arbitrate(unpeeled_candidate, peeled_lemma, peeled_suffix, "suffix", fam)
            for fam in families
        ]
        # Ambiguous suffixes (-ው, -ት, -ን ...) are tested in every slot they can fill:
        # a proven peel in any slot wins, then a maximality verdict, then undecided.
        child_wins = [v for v in verdicts if v.decision == CHILD]
        if child_wins:
            return max(child_wins, key=lambda v: len(v.child_evidence))
        parent_wins = [v for v in verdicts if v.decision == PARENT]
        if parent_wins:
            return parent_wins[0]
        return verdicts[0]

    def verify_suffix_lemma_autonomy(self, word: str) -> bool:
        """
        Tests if word W hosts outer nominal determiners (-ው, -ኡ), case (-ን),
        or pragmatic enclitics (-ም, -ና, -ስ).
        e.g., ነሀሴው proves ነሀሴ is an autonomous nominal base.
        """
        if not word or len(word) < 2:
            return False

        test_forms = [word + c for c in ("ው", "ኡ", "ን", "ም", "ና", "ስ")]
        return any(self._is_attested(f) for f in test_forms)

    # ------------------------------------------------------------------
    # Lemma Arbitration (evidence-based, maximality-first)
    # ------------------------------------------------------------------
    #
    # Every decision tests the SAME slot on both sides of a peel:
    #   parent evidence = attested forms where the unpeeled parent takes a sister in the
    #                     slot the peel claims (if that slot is really filled, it cannot)
    #   child evidence  = attested forms where the peeled host takes COMPETING sisters
    #                     in that slot (the parent itself never counts: no circularity)
    #
    #   child == 0 and parent verified   -> PARENT    (unproven peel of a verified lemma)
    #   child == 0 and parent unverified -> UNDECIDED (no evidence either way)
    #   child  > 0 and parent == 0       -> CHILD     (mutual exclusion: slot is filled)
    #   child  > parent                  -> CHILD
    #   otherwise                        -> PARENT    (maximality breaks ties)

    def arbitrate_competing_lemmas(
        self, unpeeled_lemma: str, peeled_lemma: str, peeled_prefix: str
    ) -> str:
        """
        Pairwise prefix arbitration (e.g. 'በጀት' vs 'በ' + 'ጀት', 'በዚህ' vs 'በ' + 'ዚህ').
        Returns the winning string. Use arbitrate_prefix_pair() for the full verdict.
        """
        verdict = self.arbitrate_prefix_pair(unpeeled_lemma, peeled_lemma, peeled_prefix)
        return self._verdict_to_winner(verdict, unpeeled_lemma, peeled_lemma)

    def arbitrate_prefix_pair(
        self, unpeeled_lemma: str, peeled_lemma: str, peeled_prefix: str
    ) -> EdgeVerdict:
        """Evidence verdict for an unpeeled form vs. its proclitic-peeled remnant."""
        if peeled_prefix not in self.PROCLITIC_PEEL_TARGETS:
            return self._undecided(
                unpeeled_lemma, peeled_lemma, peeled_prefix, "prefix",
                "prefix is not a proclitic; no sister paradigm to test",
            )
        if self._looks_verbal(peeled_lemma):
            return self._undecided(
                unpeeled_lemma, peeled_lemma, peeled_prefix, "prefix",
                "proclitic on a verbal stem is grammatical (relative/subordinate)",
            )
        return self._arbitrate(
            unpeeled_lemma, peeled_lemma, peeled_prefix, "prefix", self.FAMILY_PROCLITIC
        )

    def arbitrate_pool(
        self, pool: List[Any], eligible: Optional[List[Any]] = None
    ) -> PoolArbitration:
        """
        Listwise arbitration over the WHOLE candidate pool (maximality principle).

        Candidates are duck-typed (.stem, .prefixes, .suffixes, .rules), so this module
        does not depend on segmenter.Candidate.
          pool     -- every generated candidate; all of them serve as evidence
          eligible -- candidates that passed morphotactic filtering; only these survive

        Parent-child links are rebuilt from the derivation path: a candidate's parent is
        the pool entry whose rules are a prefix of its rules and whose affixes it extends.
        Each edge that peels exactly one loose morpheme is decided by _arbitrate():
          PARENT -> the child and all of its descendants are eliminated
          CHILD  -> the unpeeled parent is eliminated
        evidence_rank[id(c)] = proven peels on c's path + edges c won as the parent.
        """
        pool = list(pool)
        eligible = list(pool if eligible is None else eligible)
        eligible_ids = {id(c) for c in eligible}
        result = PoolArbitration()

        root = next(
            (c for c in pool if not c.prefixes and not c.suffixes and not c.rules), None
        )
        result.root_verified = bool(
            root is not None
            and self.is_autonomous_lemma(root.stem)
            and self.is_uninflected_base(root.stem)
        )

        # No corpus -> no evidence -> no arbitration
        if not self.vocab_freq:
            result.survivors = eligible
            return result

        by_rules: Dict[Tuple[str, ...], List[Any]] = {}
        for c in pool:
            by_rules.setdefault(tuple(c.rules), []).append(c)
        parent_of: Dict[int, Any] = {
            id(c): self._find_lineage_parent(c, by_rules) for c in pool
        }

        edge_decision: Dict[int, str] = {}  # keyed by id(child) of the edge
        lost_as_parent: Set[int] = set()
        won_as_parent: Dict[int, int] = {}
        for c in pool:
            p = parent_of[id(c)]
            if p is None:
                continue
            verdict = self._arbitrate_lineage_edge(p, c, result.root_verified)
            if verdict is None:
                continue
            result.verdicts.append((p, c, verdict))
            edge_decision[id(c)] = verdict.decision
            if verdict.decision == CHILD:
                lost_as_parent.add(id(p))
            elif verdict.decision == PARENT and id(c) in eligible_ids:
                won_as_parent[id(p)] = won_as_parent.get(id(p), 0) + 1

        for c in eligible:
            decisions = [edge_decision.get(id(n)) for n in self._lineage(c, parent_of)]
            if PARENT in decisions or id(c) in lost_as_parent:
                result.eliminated.append(c)
                continue
            result.survivors.append(c)
            result.evidence_rank[id(c)] = decisions.count(CHILD) + won_as_parent.get(id(c), 0)

        if not result.survivors:
            # Never return an empty pool: fall back to the filtered candidates unchanged
            result.survivors = eligible
            result.eliminated = []
            result.evidence_rank = {}
        return result

    # ------------------------------------------------------------------
    # Arbitration internals
    # ------------------------------------------------------------------

    def _arbitrate(
        self,
        parent_stem: str,
        child_stem: str,
        peeled: str,
        side: str,
        family: str,
        root_verified: bool = False,
        child_has_prefixes: bool = False,
    ) -> EdgeVerdict:
        parent_verified = self._is_verified_lemma(parent_stem)

        def verdict(decision: str, reason: str, p_ev=(), c_ev=()) -> EdgeVerdict:
            return EdgeVerdict(
                parent_stem=parent_stem, child_stem=child_stem, peeled=peeled,
                side=side, family=family, decision=decision, reason=reason,
                parent_verified=parent_verified,
                parent_evidence=tuple(p_ev), child_evidence=tuple(c_ev),
            )

        if family == self.FAMILY_DERIVATIONAL:
            if self.is_autonomous_lemma(parent_stem):
                return verdict(PARENT, "derivational shield: unpeeled form is an autonomous lemma")
            return verdict(UNDECIDED, "derivational peel: unpeeled form is not an autonomous lemma")

        if family == self.FAMILY_ROOT_SHIELD:
            if root_verified and not child_has_prefixes:
                return verdict(PARENT, "slice off a verified root lemma (template + anchor evidence)")
            return verdict(UNDECIDED, "root is not a verified lemma")

        p_ev, c_ev, discarded, threshold = self._edge_sister_evidence(
            family, parent_stem, child_stem, peeled
        )
        p, c = len(p_ev), len(c_ev)
        note = f" [ambiguous, not counted: {list(discarded)}]" if discarded else ""
        if c == 0:
            if parent_verified:
                return verdict(
                    PARENT,
                    f"'{child_stem}' takes no competing sisters of '{peeled}'; "
                    f"'{parent_stem}' is a verified lemma (maximality){note}",
                    p_ev, c_ev,
                )
            return verdict(UNDECIDED, f"neither side has corpus evidence{note}", p_ev, c_ev)
        if p == 0:
            parent_freq = self.vocab_freq.get(self._n(parent_stem), 0)
            if parent_freq >= self.ABSENCE_POWER_FACTOR * threshold:
                return verdict(
                    CHILD,
                    f"'{child_stem}' takes {c} competing sister(s); '{parent_stem}' "
                    f"(freq {parent_freq}) takes none in the same slot (mutual exclusion){note}",
                    p_ev, c_ev,
                )
            return verdict(
                UNDECIDED,
                f"'{parent_stem}' (freq {parent_freq}) is too rare for its missing sisters "
                f"to count (needs >= {self.ABSENCE_POWER_FACTOR * threshold}); "
                f"child has {c}{note}",
                p_ev, c_ev,
            )
        if c > p:
            return verdict(CHILD, f"child paradigm stronger ({c} vs {p})", p_ev, c_ev)
        return verdict(
            PARENT, f"parent paradigm at least as strong ({p} vs {c}); maximality", p_ev, c_ev
        )

    def _edge_sister_evidence(
        self, family: str, parent_stem: str, child_stem: str, peeled: str
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], int]:
        """Returns (parent_evidence, child_evidence, discarded_ambiguous, threshold)."""
        if family == self.FAMILY_PRAGMATIC:
            p_forms = [parent_stem + s for s in self.PRAGMATIC_SISTERS]
            c_forms = [child_stem + s for s in self.PRAGMATIC_SISTERS if s != peeled]
            threshold = self.min_freq
        elif family == self.FAMILY_OBJECT:
            p_forms = self._object_clitic_forms(parent_stem)
            c_forms = self._object_clitic_forms(child_stem)
            threshold = self.OBJECT_CLITIC_MIN_FREQ
        elif family == self.FAMILY_DETERMINER:
            p_forms = self._nominal_sister_forms(parent_stem)
            c_forms = self._nominal_sister_forms(child_stem)
            threshold = self.min_freq
        elif family == self.FAMILY_PROCLITIC:
            p_forms = self._proclitic_sister_forms(parent_stem)
            c_forms = self._proclitic_sister_forms(child_stem, skip=peeled)
            threshold = self.min_freq
        else:
            return (), (), (), self.min_freq
        # Circularity guard: the parent is never evidence for its own peel
        p_ev = self._attested_forms(p_forms, threshold)
        c_ev = self._attested_forms(c_forms, threshold, exclude=(parent_stem,))
        if family == self.FAMILY_PROCLITIC:
            # Proclitic sisters are proclitic + host by design; nothing to disambiguate
            return p_ev, c_ev, (), threshold
        # Evidence purity: a suffix sister that also reads as proclitic + attested
        # word (በጥም = በ + ጥም) proves nothing about this host. Same test on both sides.
        p_ev, p_bad = self._split_unambiguous(p_ev)
        c_ev, c_bad = self._split_unambiguous(c_ev)
        return p_ev, c_ev, p_bad + c_bad, threshold

    def _split_unambiguous(
        self, forms: Tuple[str, ...]
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        clean: List[str] = []
        ambiguous: List[str] = []
        for f in forms:
            (ambiguous if self._has_proclitic_reading(f) else clean).append(f)
        return tuple(clean), tuple(ambiguous)

    def _has_proclitic_reading(self, form: str) -> bool:
        """
        True if `form` also splits as proclitic + a word attested at the
        arbitrator's threshold (e.g. በጥም = በ + ጥም, ባለፈም = በ + አለፈም).
        The same `min_freq` threshold is used even for the object-clitic family
        so that only well-attested alternatives disqualify.
        """
        if not form:
            return False
        for p in self.ALL_PROCLITICS:
            if form.startswith(p) and len(form) - len(p) >= 2:
                if self._is_attested(form[len(p):]):
                    return True
        if form[0] in self.COALESCED_PROCLITICS and len(form) >= 3:
            if self._is_attested("አ" + form[1:]):
                return True
        return False

    def _arbitrate_lineage_edge(
        self, parent: Any, child: Any, root_verified: bool
    ) -> Optional[EdgeVerdict]:
        n_pfx, n_sfx = len(parent.prefixes), len(parent.suffixes)
        # Prefixes accumulate outer-first (appended); suffixes inner-first (prepended)
        extra_pfx = tuple(child.prefixes[n_pfx:])
        extra_sfx = tuple(child.suffixes[: len(child.suffixes) - n_sfx])
        if len(extra_pfx) + len(extra_sfx) != 1:
            return None  # only single-morpheme peels are arbitrated
        rule = child.rules[len(parent.rules)] if len(child.rules) > len(parent.rules) else ""
        family = self._classify_edge(rule, extra_pfx, extra_sfx, child)
        if family is None:
            return None
        if extra_pfx:
            return self._arbitrate(parent.stem, child.stem, extra_pfx[0], "prefix", family)
        return self._arbitrate(
            parent.stem, child.stem, extra_sfx[0], "suffix", family,
            root_verified=root_verified, child_has_prefixes=bool(child.prefixes),
        )

    def _classify_edge(
        self, rule: str, extra_pfx: Tuple[str, ...], extra_sfx: Tuple[str, ...], child: Any
    ) -> Optional[str]:
        """Map one peel to its sister-testing slot family, if applicable."""
        r = rule or ""
        if extra_pfx:
            p = extra_pfx[0]
            if (
                p in self.PROCLITIC_PEEL_TARGETS
                and not r.startswith("rel_")
                and not self._looks_verbal(child.stem)
            ):
                return self.FAMILY_PROCLITIC
            return None

        s = extra_sfx[0]
        if r.startswith("clitic") and s in self.PRAGMATIC_SISTERS:
            return self.FAMILY_PRAGMATIC
        if r.startswith(("verb_object", "verb_obj")):
            # -ው / -ት after a relative prefix is a determiner/concord, not an object
            if s in ("ው", "ት") and self._is_relative_host(child):
                return None
            return self.FAMILY_OBJECT
        if r.startswith(("possessive", "definite")):
            if self._in_verbal_context(child):
                return None
            return self.FAMILY_DETERMINER
        if s in self.DERIVATIONAL_SUFFIXES:
            return self.FAMILY_DERIVATIONAL
        if r.startswith(self.ROOT_SHIELD_RULES):
            return self.FAMILY_ROOT_SHIELD
        return None

    def _families_for_suffix(self, suffix: str) -> Tuple[str, ...]:
        if suffix in self.PRAGMATIC_SISTERS:
            return (self.FAMILY_PRAGMATIC,)
        if suffix in self.DERIVATIONAL_SUFFIXES:
            return (self.FAMILY_DERIVATIONAL,)
        if suffix in self.OBJECT_ONLY_SUFFIXES:
            return (self.FAMILY_OBJECT,)
        if suffix in self.DETERMINER_ONLY_SUFFIXES:
            return (self.FAMILY_DETERMINER,)
        if suffix in self.AMBIGUOUS_OBJ_DET_SUFFIXES:
            return (self.FAMILY_OBJECT, self.FAMILY_DETERMINER)
        return ()

    def _verdict_to_winner(self, verdict: EdgeVerdict, unpeeled: str, peeled: str) -> str:
        if verdict.decision == CHILD:
            return peeled
        if verdict.decision == PARENT:
            return unpeeled
        # Undecided: keep the unpeeled form only if it is itself a verified lemma
        return unpeeled if verdict.parent_verified else peeled

    def _undecided(
        self, parent_stem: str, child_stem: str, peeled: str, side: str, reason: str
    ) -> EdgeVerdict:
        return EdgeVerdict(
            parent_stem=parent_stem, child_stem=child_stem, peeled=peeled, side=side,
            family=self.FAMILY_NONE, decision=UNDECIDED, reason=reason,
            parent_verified=self._is_verified_lemma(parent_stem),
        )

    # ------------------------------------------------------------------
    # Lineage reconstruction
    # ------------------------------------------------------------------

    def _find_lineage_parent(
        self, c: Any, by_rules: Dict[Tuple[str, ...], List[Any]]
    ) -> Optional[Any]:
        rules = tuple(c.rules)
        # Nearest recorded ancestor (an intermediate state may not have been recorded)
        for k in range(len(rules) - 1, -1, -1):
            for p in by_rules.get(rules[:k], ()):
                if p is not c and self._is_lineage_edge(p, c):
                    return p
        return None

    @staticmethod
    def _is_lineage_edge(parent: Any, child: Any) -> bool:
        n_pfx, n_sfx = len(parent.prefixes), len(parent.suffixes)
        if len(child.prefixes) < n_pfx or len(child.suffixes) < n_sfx:
            return False
        if tuple(child.prefixes[:n_pfx]) != tuple(parent.prefixes):
            return False
        if n_sfx and tuple(child.suffixes[len(child.suffixes) - n_sfx:]) != tuple(parent.suffixes):
            return False
        return True

    @staticmethod
    def _lineage(c: Any, parent_of: Dict[int, Any], max_depth: int = 64) -> List[Any]:
        chain: List[Any] = []
        node = c
        while node is not None and len(chain) < max_depth:
            chain.append(node)
            node = parent_of.get(id(node))
        return chain

    # ------------------------------------------------------------------
    # Evidence helpers
    # ------------------------------------------------------------------

    def _n(self, form: str) -> str:
        return self.normalizer(form) if self.normalizer else form

    def _attested_forms(
        self, forms: List[str], min_freq: int, exclude: Tuple[str, ...] = ()
    ) -> Tuple[str, ...]:
        excluded = {self._n(e) for e in exclude if e}
        seen: Set[str] = set()
        out: List[str] = []
        for f in forms:
            if not f:
                continue
            nf = self._n(f)
            if nf in seen or nf in excluded:
                continue
            seen.add(nf)
            if self.vocab_freq.get(nf, 0) >= min_freq:
                out.append(f)
        return tuple(out)

    def _is_verified_lemma(self, stem: str) -> bool:
        if not stem:
            return False
        if stem in self._verify_memo:
            return self._verify_memo[stem]
        if len(self._verify_memo) > 8192:
            self._verify_memo.clear()
        ok = self._is_attested(stem) or self.is_autonomous_lemma(stem)
        self._verify_memo[stem] = ok
        return ok

    def _looks_verbal(self, stem: str) -> bool:
        m = self.morphology
        if not m or not stem:
            return False
        if stem in getattr(m, "verb_stems", set()):
            return True
        for name in ("is_valid_perfective_template", "is_valid_imperfective_stem"):
            fn = getattr(m, name, None)
            if fn and fn(stem):
                return True
        return False

    def _is_relative_host(self, c: Any) -> bool:
        rel = getattr(self.morphology, "prefix_relative_pronoun", set()) if self.morphology else set()
        return any(p in rel or p.startswith("የ") for p in c.prefixes)

    def _in_verbal_context(self, c: Any) -> bool:
        m = self.morphology
        excl_v = getattr(m, "exclusive_verbal_prefixes", set()) if m else set()
        aux = getattr(m, "suffix_verb_auxiliary", set()) if m else set()
        return (
            any(p in excl_v or p == "የ" for p in c.prefixes)
            or any(s in ("አል", "ዋል") or s in aux for s in c.suffixes)
            or self._looks_verbal(c.stem)
        )

    # Sister-form generators (same shapes as verify_* probes above)

    def _object_clitic_forms(self, stem: str) -> List[str]:
        if not stem or len(stem) < 2:
            return []
        last_c = stem[-1]
        _, last_order = self.phonology.get_char_info(last_c)
        forms = [stem + s for s in ("ኝ", "ው", "ን", "አት", "ቸው")]
        if last_order == 0 and len(stem) >= 3:
            a = self.phonology.shift_order(last_c, 3)
            forms += [stem[:-1] + a + "ት", stem[:-1] + a + "ቸው"]
        elif last_order == 5:
            g = self.phonology.shift_order(last_c, 0)
            a = self.phonology.shift_order(last_c, 3)
            forms += [stem[:-1] + g + s for s in ("ው", "ኝ", "ን", "ህ")]
            forms += [stem[:-1] + a + "ት", stem[:-1] + a + "ቸው"]
        return forms

    def _nominal_sister_forms(self, stem: str) -> List[str]:
        if not stem or len(stem) < 2:
            return []
        last_c = stem[-1]
        _, last_order = self.phonology.get_char_info(last_c)
        if last_order == 5:
            o = self.phonology.shift_order(last_c, 6)
            a = self.phonology.shift_order(last_c, 3)
            return [stem[:-1] + o + "ች", stem[:-1] + a + "ችን", stem[:-1] + "ቷ"]
        return [stem + "ዎች", stem + "ችን", stem + "ዋ"]

    def _proclitic_sister_forms(self, stem: str, skip: str = "") -> List[str]:
        if not stem or len(stem) < 2:
            return []
        forms: List[str] = []
        for sister in self.ALL_PROCLITICS:
            if sister != skip:
                forms.extend(self._synthesize_prefix_attachment(sister, stem))
        return forms

# Backward-compatible class name for existing imports.
LemmaIdentifier = MaximalLemmaArbitrator

__all__ = [
    "MaximalLemmaArbitrator",
    "LemmaIdentifier",
    "CommutationResult",
    "EdgeVerdict",
    "PoolArbitration",
    "VerbalLemmaEvidence",
    "PARENT",
    "CHILD",
    "UNDECIDED",
]
