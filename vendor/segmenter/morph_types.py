"""
morph_types.py: Unified morphological type definitions.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Generic
from phonology import AmharicBoundaryPhonology
import pos_tags as tags


class VerbalShape(str, Enum):
    """Amharic verbal stem shapes and their finite surface realisations.

    There are two primary stem shapes (A and C) plus the infinitive (E).
    Shape B (coverb) is derived from Shape A.  All finite forms are built
    by adding agreement morphology on top of the non-finite base.

    SHAPE A — Perfective base (ቀዳሚ)
    ----------------------------------------
    The 3MS past-tense citation form.  Finite uses:
      - Simple perfect:       bare Shape A  (ሆነ, መሰለ)
      - Relative perfective:  የ- + Shape A  (የሆነ, የመሰለ)

    SHAPE B — Converb base (አዳጋሽ)
    ----------------------------------------
    Derived from Shape A by a coverb vowel suffix.  Finite uses:
      - Standalone coverb:   bare Shape B  (ሆኖ, መስሎ)
      - Compound perfect:     Shape B + auxiliary suffix  (ሆኗል, መስሎዋል)

    SHAPE C — Imperfective base (ትንቢት) — the foundational non-finite stem
    ----------------------------------------
    The core imperfective stem is non-finite by itself.  It becomes finite
    through four different agreement strategies:

      1. CONTINGENT (simple imperfective)
            Subject prefix only, no auxiliary, no relative prefix.
            e.g. ይሆን (ይ-), ትሆን (ት-), እሆን (እ-), እንሆን (እን-)
            Appears bare (without auxiliary) in subordinate clauses and
            as the base for the jussive/imperative.

      2. COMPOUND IMPERFECT (progressive / habitual)
            Subject prefix + Shape C + auxiliary suffix (አል/ዋል).
            e.g. ይሆናል (ይ- + ሆን + አል)
            The auxiliary is the universal finiteness marker here.
            Note: ይለምድ-አል is attested; ይልመድ-አል is not — this
            corpus asymmetry is a diagnostic for the imperfective vs jussive
            stem distinction.

      3. JUSSIVE / IMPERATIVE (modal / command)
            Same subject prefixes as contingent, but the stem vowel pattern
            shifts to the jussive grade (Leslau SHAPE_D patterns).
            Person is the only agreement difference between jussive and
            imperative:
              - 1st / 3rd person → JUSSIVE   (ል-/ይ-/ት- prefix)
              - 2nd person       → IMPERATIVE (bare or -ኢ/-ኡ suffix)
            They form a single paradigm family (JUSS_IMP).

      4. RELATIVE IMPERFECTIVE
            የሚ- prefix + Shape C stem (agreement prefix dropped).
            e.g. የሚሆን, የሚመስል

    SHAPE E — Infinitive / Verbal Noun (መካነ-ስም)
    ----------------------------------------
    Non-finite nominal use of the verb.  Prefix መ-/ማ-.
    e.g. መሆን, መመሰል, ማየት

    Closed-class sentinels
    ----------------------------------------
    CLOSED_NOMINAL and CLOSED_VERBAL are used when Layer 1 resolves a
    closed-class item so that the verbal_shape field on Layer1Result is
    always populated (never None) and Layer 2 can branch without a
    None-check.
    """
    # ---- 5 Canonical Semitic Prosodic Stem Grades ----
    SHAPE_A_PERFECTIVE         = "SHAPE_A_PERFECTIVE"         # Perfective base (ቀዳሚ: ሰበረ, ሆነ)
    SHAPE_B_COVERB            = "SHAPE_B_COVERB"            # Converb base (አዳጋሽ: ሰብሮ, ሆኖ)
    SHAPE_C_IMPERFECTIVE       = "SHAPE_C_IMPERFECTIVE"       # Imperfective base (ትንቢት: ሰብር, ሆን)
    SHAPE_D_JUSSIVE_IMPERATIVE = "SHAPE_D_JUSSIVE_IMPERATIVE" # Modal command/jussive base (ትእዛዝ: ስበር, ሁን)
    SHAPE_E_INFINITIVE         = "SHAPE_E_INFINITIVE"         # Infinitive base (መካነ-ስም: መስበር, መሆን)

    # ---- Closed-class sentinels ----
    CLOSED_NOMINAL = "CLOSED_NOMINAL"   # CONJ / ADP / ADV / INT / NUM / VB:IDEO
    CLOSED_VERBAL  = "CLOSED_VERBAL"    # VRB:AUX
    UNKNOWN        = "UNKNOWN"


# ---------------------------------------------------------------------------
# Aspect-family groupings (for family-level evaluation and Layer 2 branching)
# ---------------------------------------------------------------------------

# All Shape-C realisations belong to the imperfective stem family.
# Stem family groupings for prosodic shapes
IMPERF_STEM_SHAPES: frozenset = frozenset({
    VerbalShape.SHAPE_C_IMPERFECTIVE,
    VerbalShape.SHAPE_D_JUSSIVE_IMPERATIVE,
})

PERF_STEM_SHAPES: frozenset = frozenset({
    VerbalShape.SHAPE_A_PERFECTIVE,
    VerbalShape.SHAPE_B_COVERB,
})

# ---------------------------------------------------------------------------
# Layer 1 → Layer 2 handoff container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Layer1Result:
    """Immutable result produced by Layer 1 and consumed by Layer 2.

    Carrying this object from Layer 1 to Layer 2 avoids redundant re-processing:
    the macro_tag encodes the main category, verbal_shape encodes the
    canonical Semitic stem shape, closed_subtype carries the fine tag for
    closed items, and verbal_subtag carries the exact double-lock pattern.
    """
    macro_tag: str
    verbal_shape: VerbalShape = VerbalShape.UNKNOWN
    closed_subtype: str = ""
    verbal_subtag: str = ""

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    @property
    def canonical_tag(self) -> str:
        """The tag rewritten in canonical notation (``:`` subset, ``|`` disjunction)."""
        return tags.canonical_tag(self.macro_tag)

    @property
    def is_nominal(self) -> bool:
        return tags.is_nominal(self.macro_tag)

    @property
    def is_noun(self) -> bool:
        """True for ``NOM:CN`` / ``NOM:PN`` / ``NOM:MN`` — a noun proper, not an
        adjective, pronoun, or numeral."""
        return tags.is_noun(self.macro_tag)

    @property
    def is_verbal(self) -> bool:
        return tags.is_verbal(self.macro_tag)

    @property
    def is_closed(self) -> bool:
        return tags.is_closed(self.macro_tag)

    @property
    def is_nv(self) -> bool:
        """True when the tag offers both a nominal and a verbal reading."""
        return self.is_nominal and self.is_verbal

    @property
    def is_unknown(self) -> bool:
        return tags.canonical_tag(self.macro_tag) == tags.UNKNOWN


try:
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "morph_types requires pydantic. Install it with: pip install pydantic"
    ) from exc

try:
    # ConfigDict exists only in pydantic v2; its absence identifies v1.
    from pydantic import ConfigDict
    PYDANTIC_V2 = True
except ImportError:
    PYDANTIC_V2 = False
    ConfigDict = None  # type: ignore


@dataclass(frozen=True)
class MorphSplit:
    """Standardized result of a single morphological decomposition operation."""
    stem: str
    prefixes: Tuple[str, ...] = ()
    suffixes: Tuple[str, ...] = ()
    rule: str = ""
    strength: int = 1


@dataclass(frozen=True)
class NominalCohortEvidence:
    """Declarative paradigm evidence for nominal stems across 7 orthogonal dimensions."""
    compound: bool = False
    preposition: bool = False
    associative: bool = False
    def_acc: bool = False
    possessive: bool = False
    plural: bool = False
    abstract: bool = False
    robust_plural: bool = False
    true_pn_assoc: bool = False

    @property
    def total_slots(self) -> int:
        return (
            int(self.compound)
            + int(self.preposition)
            + int(self.associative)
            + int(self.def_acc)
            + int(self.possessive)
            + int(self.plural)
            + int(self.abstract)
        )


@dataclass(frozen=True)
class VerbalCohortEvidence:
    """Declarative paradigm evidence for verbal stems distinguishing direct vs reconstructed root concord."""
    exclusive_subject: bool = False
    exclusive_object: bool = False
    exclusive_auxiliary: bool = False
    negative: bool = False
    voice_aspect: bool = False
    coverb: bool = False
    applicative: bool = False
    is_direct_stem: bool = False

    @property
    def core_concord_hits(self) -> int:
        return int(self.exclusive_subject) + int(self.exclusive_object) + int(self.exclusive_auxiliary)

    @property
    def secondary_hits(self) -> int:
        return int(self.negative) + int(self.voice_aspect) + int(self.coverb) + int(self.applicative)

    @property
    def total_slots(self) -> int:
        return self.core_concord_hits + self.secondary_hits

    @property
    def has_finite_concord(self) -> bool:
        return self.core_concord_hits >= 1


@dataclass(frozen=True)
class DecomposeContext:
    """Carries search context and pre-computed morphotactic flags across decomposition handlers."""
    prefixes: Tuple[str, ...] = ()
    suffixes: Tuple[str, ...] = ()
    known_stem_checker: Optional[Callable[[str], bool]] = None
    known_verbal_stem_checker: Optional[Callable[[str], bool]] = None
    include_raw_suffix_fallback: bool = False
    has_exclusive_verb_pfx: bool = False
    has_exclusive_verb_sfx: bool = False
    has_poss_or_def_sfx: bool = False
    has_negative_context: bool = False
    has_prep: bool = False


class MorphemesConfig(BaseModel):
    """Pydantic validation contract for morphemes.json (100% human-readable schema)."""
    if PYDANTIC_V2:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"

    # Nominal Suffixes
    suffix_nominal_plural_exclusive: List[str] = Field(default_factory=list)
    suffix_nominal_plural_ambiguous: List[str] = Field(default_factory=list)
    suffix_nominal_definiteness_exclusive: List[str] = Field(default_factory=list)
    suffix_nominal_definiteness_ambiguous: List[str] = Field(default_factory=list)
    suffix_nominal_possessive_exclusive: List[str] = Field(default_factory=list)
    suffix_nominal_possessive_ambiguous: List[str] = Field(default_factory=list)
    suffix_nominal_abstract_exclusive: List[str] = Field(default_factory=list)
    suffix_nominal_abstract_ambiguous: List[str] = Field(default_factory=list)
    suffix_nominal_derivational_exclusive: List[str] = Field(default_factory=list)
    suffix_nominal_derivational_ambiguous: List[str] = Field(default_factory=list)
    suffix_nominal_instrumental_exclusive: List[str] = Field(default_factory=list)
    suffix_nominal_instrumental_ambiguous: List[str] = Field(default_factory=list)
    suffix_nominal_accusative_ambiguous: List[str] = Field(default_factory=list)

    # Verbal Suffixes
    suffix_verb_subject_exclusive: List[str] = Field(default_factory=list)
    suffix_verb_subject_ambiguous: List[str] = Field(default_factory=list)
    suffix_verb_object_exclusive: List[str] = Field(default_factory=list)
    suffix_verb_object_ambiguous: List[str] = Field(default_factory=list)
    suffix_verb_applicative_exclusive: List[str] = Field(default_factory=list)
    suffix_verb_applicative_ambiguous: List[str] = Field(default_factory=list)
    suffix_verb_coverb_exclusive: List[str] = Field(default_factory=list)
    suffix_verb_coverb_ambiguous: List[str] = Field(default_factory=list)
    suffix_verb_coverb_weak: List[str] = Field(default_factory=list)
    suffix_verb_auxiliary_exclusive: List[str] = Field(default_factory=list)
    suffix_verb_auxiliary_ambiguous: List[str] = Field(default_factory=list)
    suffix_verb_negative_finiteness_exclusive: List[str] = Field(default_factory=list)
    suffix_verb_negative_finiteness_ambiguous: List[str] = Field(default_factory=list)

    # Nominal Prefixes
    prefix_nominal_associatives_exclusive: List[str] = Field(default_factory=list)
    prefix_nominal_associatives_ambiguous: List[str] = Field(default_factory=list)
    prefix_nominal_compound_exclusive: List[str] = Field(default_factory=list)
    prefix_nominal_compound_ambiguous: List[str] = Field(default_factory=list)
    prefix_nominal_prepositional_exclusive: List[str] = Field(default_factory=list)
    prefix_nominal_prepositional_ambiguous: List[str] = Field(default_factory=list)
    prefix_nominal_locative: List[str] = Field(default_factory=list)
    prefix_nominal_privative: List[str] = Field(default_factory=list)

    # Verbal Prefixes
    voice_morphology: Dict[str, List[str]] = Field(default_factory=dict)
    prefix_verb_voice_exclusive: List[str] = Field(default_factory=list)
    prefix_verb_voice_ambiguous: List[str] = Field(default_factory=list)
    prefix_verb_aspectual_perfective_exclusive: List[str] = Field(default_factory=list)
    prefix_verb_aspectual_imperfective_ambiguous: List[str] = Field(default_factory=list)
    prefix_verb_negative_exclusive: List[str] = Field(default_factory=list)
    prefix_verb_negative_ambiguous: List[str] = Field(default_factory=list)
    prefix_verb_subordinate_exclusive: List[str] = Field(default_factory=list)
    prefix_verb_subordinate_ambiguous: List[str] = Field(default_factory=list)
    prefix_verb_progressive_exclusive: List[str] = Field(default_factory=list)
    prefix_verb_progressive_ambiguous: List[str] = Field(default_factory=list)
    

    # Universal Clitics & Prepositions
    prefix_universal_prepositional_exclusive: List[str] = Field(default_factory=list)
    prefix_universal_prepositional_ambiguous: List[str] = Field(default_factory=list)
    prefix_relative_pronoun: List[str] = Field(default_factory=list)
    prefix_rel_imperfective_particle: List[str] = Field(default_factory=list)
    prefix_rel_imperfective_particle_allomorphs: List[str] = Field(default_factory=list)
    prefix_rel_imperfective_agreement: List[str] = Field(default_factory=list)
    suffix_universal_clitics_exclusive: List[str] = Field(default_factory=list)
    suffix_universal_clitics_ambiguous: List[str] = Field(default_factory=list)
    prefix_verb_negative_vn: List[str] = Field(default_factory=list)
    prefix_infinitive_coalesced: List[str] = Field(default_factory=list)

    # Whitelists & Invariants
    pronoun_stems: List[str] = Field(default_factory=list)
    geez_plurals: List[str] = Field(default_factory=list)
    confirmed_sparse_nouns: List[str] = Field(default_factory=list)
    confirmed_sparse_verbs: List[str] = Field(default_factory=list)
    irregular_verbs_say: List[str] = Field(default_factory=list)
    biradical_type_b_verbs: List[str] = Field(default_factory=list, alias="Biradical_Type_B_verbs")
    biradical_type_a_medial_a_verbs: List[str] = Field(default_factory=list)
    verb_stems: List[str] = Field(default_factory=list)
    verbs_ending_in_gna: List[str] = Field(default_factory=list)
    weak_coverb_lemma_endings: List[str] = Field(default_factory=list)
    imperfective_negative_prefixes: List[str] = Field(default_factory=list)
    verbal_prefixes_requiring_attestation: List[str] = Field(default_factory=list)
    subject_e_exclusions: List[str] = Field(default_factory=list)
    derived_voice_prefixes: List[str] = Field(default_factory=list)
    unambiguous_assoc_prefixes: List[str] = Field(default_factory=list)
    possessive_1pl_endings: List[str] = Field(default_factory=list)
    closed_class_quantifiers: List[str] = Field(default_factory=list)
    ambiguous_verbal_nouns: List[str] = Field(default_factory=list)
    inherent_terminal_u_stems: List[str] = Field(default_factory=list)
    auxiliary_verbs_standalone: List[str] = Field(default_factory=list)
    closed_class_conjunctions: List[str] = Field(default_factory=list)
    closed_class_adpositions: List[str] = Field(default_factory=list)
    closed_class_adverbs: List[str] = Field(default_factory=list)
    closed_class_question_words: List[str] = Field(default_factory=list)
    closed_class_ideophones: List[str] = Field(default_factory=list)

    # Complex Mappings
    bipartite_applicative_mappings: Dict[str, List[str]] = Field(default_factory=dict)
    auxiliary_coda_shifts: Dict[str, str] = Field(default_factory=dict)
    compound_auxiliary_suffixes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    compound_plural_possessives: Dict[str, List[str]] = Field(default_factory=dict)
    defective_verbal_forms: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    suppletive_stem_templates: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Coalescence & Shield Data
    type_c_shield_words: List[str] = Field(default_factory=list)
    #: Words that look like ``proclitic + stem`` but are single roots, on onsets the
    #: Rabe shield in ``_peel_prefix_coalescence`` does not cover (Ge'ez-order በ-/ከ-/ለ-).
    #: በረገደ "be delayed" is a 4-radical root, not በ- + ረገደ.
    proclitic_root_shield: List[str] = Field(default_factory=list)
    privative_vn_prefix_map: Dict[str, List[str]] = Field(default_factory=dict)
    multi_rabe_prefixes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    syncretic_imp_juss_orders: List[List[int]] = Field(default_factory=list)
    syncretic_imp_juss_lexemes: List[str] = Field(default_factory=list)


K = TypeVar('K')
V = TypeVar('V')

class SimpleLRUCache(Generic[K, V]):
    """
    A fast, zero-dependency Least Recently Used (LRU) cache 
    built on Python's insertion-ordered dictionary.
    """
    def __init__(self, maxsize: int = 16384):
        self.cache: Dict[K, V] = {}
        self.maxsize = maxsize

    def __contains__(self, key: K) -> bool:
        return key in self.cache

    def __getitem__(self, key: K) -> V:
        # Move to end (mark as most recently used)
        val = self.cache.pop(key)
        self.cache[key] = val
        return val

    def get(self, key: K, default: Any = None) -> Any:
        if key in self.cache:
            val = self.cache.pop(key)
            self.cache[key] = val
            return val
        return default

    def __setitem__(self, key: K, value: V) -> None:
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.maxsize:
            # Pop the first item (least recently used)
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value

    def clear(self) -> None:
        self.cache.clear()

    def __len__(self) -> int:
        return len(self.cache)
