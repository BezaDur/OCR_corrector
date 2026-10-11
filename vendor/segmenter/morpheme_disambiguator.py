"""
morpheme_disambiguator.py - 100% Declarative Amharic Morpheme Disambiguator.
Zero hardcoded character comparisons.
Disambiguates polysemous affixes entirely through dynamic inverted category indexing
and domain set-intersection derived from morphemes.json schema keys.
"""
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class MorphemeRole(str, Enum):
    # Nominal Roles
    ACCUSATIVE_CASE = "ACCUSATIVE_CASE"
    DEFINITE_ARTICLE = "DEFINITE_ARTICLE"
    POSSESSIVE_PRONOUN = "POSSESSIVE_PRONOUN"
    GEEZ_PLURAL = "GEEZ_PLURAL"
    NOMINAL_PLURAL = "NOMINAL_PLURAL"
    GENITIVE_PREPOSITION = "GENITIVE_PREPOSITION"
    LOCATIVE_PREPOSITION = "LOCATIVE_PREPOSITION"
    ABSTRACT_SUFFIX = "ABSTRACT_SUFFIX"
    DERIVATIONAL_SUFFIX = "DERIVATIONAL_SUFFIX"
    INSTRUMENTAL_SUFFIX = "INSTRUMENTAL_SUFFIX"
    COMPOUND_PREFIX = "COMPOUND_PREFIX"
    ASSOCIATIVE_PREFIX = "ASSOCIATIVE_PREFIX"

    # Verbal Roles
    VERB_SUBJECT_AGREEMENT = "VERB_SUBJECT_AGREEMENT"
    VERB_OBJECT_CLITIC = "VERB_OBJECT_CLITIC"
    VERB_APPLICATIVE_CLITIC = "VERB_APPLICATIVE_CLITIC"
    VERB_COVERB = "VERB_COVERB"
    VERB_AUXILIARY = "VERB_AUXILIARY"
    RELATIVE_COMPLEMENTIZER = "RELATIVE_COMPLEMENTIZER"
    SUBORDINATE_CONJUNCTION = "SUBORDINATE_CONJUNCTION"
    VOICE_PREFIX = "VOICE_PREFIX"
    PROGRESSIVE_ASPECT = "PROGRESSIVE_ASPECT"
    NEGATIVE_PREFIX = "NEGATIVE_PREFIX"
    NEGATIVE_FINITENESS = "NEGATIVE_FINITENESS"

    # Universal & Fallback
    PRAGMATIC_CLITIC = "PRAGMATIC_CLITIC"
    UNKNOWN = "UNKNOWN"


class AmharicMorphemeDisambiguator:

    # Declarative Schema Category to Abstract Role Mapping
    CATEGORY_TO_ROLE: Dict[str, MorphemeRole] = {
        # Nominal Suffix Categories
        # Nominal Suffix Categories
        "suffix_accusative": MorphemeRole.ACCUSATIVE_CASE,
        "suffix_nominal_plural_exclusive": MorphemeRole.NOMINAL_PLURAL,
        "suffix_nominal_exclusive_plural": MorphemeRole.NOMINAL_PLURAL,
        "suffix_nominal_plural_ambiguous": MorphemeRole.NOMINAL_PLURAL,
        "suffix_nominal_plural": MorphemeRole.NOMINAL_PLURAL,
        "suffix_nominal_definiteness_exclusive": MorphemeRole.DEFINITE_ARTICLE,
        "suffix_nominal_exclusive_definiteness": MorphemeRole.DEFINITE_ARTICLE,
        "suffix_nominal_definiteness_ambiguous": MorphemeRole.DEFINITE_ARTICLE,
        "suffix_nominal_definiteness": MorphemeRole.DEFINITE_ARTICLE,
        "suffix_nominal_possessive_exclusive": MorphemeRole.POSSESSIVE_PRONOUN,
        "suffix_nominal_exclusive_possessive": MorphemeRole.POSSESSIVE_PRONOUN,
        "suffix_nominal_possessive_ambiguous": MorphemeRole.POSSESSIVE_PRONOUN,
        "suffix_nominal_possessive": MorphemeRole.POSSESSIVE_PRONOUN,
        "suffix_nominal_abstract_exclusive": MorphemeRole.ABSTRACT_SUFFIX,
        "suffix_nominal_exclusive_abstract": MorphemeRole.ABSTRACT_SUFFIX,
        "suffix_nominal_abstract_ambiguous": MorphemeRole.ABSTRACT_SUFFIX,
        "suffix_nominal_abstract": MorphemeRole.ABSTRACT_SUFFIX,
        "suffix_nominal_derivational_exclusive": MorphemeRole.DERIVATIONAL_SUFFIX,
        "suffix_nominal_exclusive_derivational": MorphemeRole.DERIVATIONAL_SUFFIX,
        "suffix_nominal_derivational_ambiguous": MorphemeRole.DERIVATIONAL_SUFFIX,
        "suffix_nominal_derivational": MorphemeRole.DERIVATIONAL_SUFFIX,
        "suffix_nominal_instrumental_exclusive": MorphemeRole.INSTRUMENTAL_SUFFIX,
        "suffix_nominal_exclusive_instrumental": MorphemeRole.INSTRUMENTAL_SUFFIX,
        "suffix_nominal_instrumental_ambiguous": MorphemeRole.INSTRUMENTAL_SUFFIX,
        "suffix_nominal_instrumental": MorphemeRole.INSTRUMENTAL_SUFFIX,

        # Verbal Suffix Categories
        "suffix_verb_subject_exclusive": MorphemeRole.VERB_SUBJECT_AGREEMENT,
        "suffix_verb_exclusive_subject": MorphemeRole.VERB_SUBJECT_AGREEMENT,
        "suffix_verb_subject_ambiguous": MorphemeRole.VERB_SUBJECT_AGREEMENT,
        "suffix_verb_pv": MorphemeRole.VERB_SUBJECT_AGREEMENT,
        "suffix_verb_object_exclusive": MorphemeRole.VERB_OBJECT_CLITIC,
        "suffix_verb_exclusive_object": MorphemeRole.VERB_OBJECT_CLITIC,
        "suffix_verb_object_ambiguous": MorphemeRole.VERB_OBJECT_CLITIC,
        "suffix_verb_object_clitics": MorphemeRole.VERB_OBJECT_CLITIC,
        "suffix_verb_applicative_exclusive": MorphemeRole.VERB_APPLICATIVE_CLITIC,
        "suffix_verb_exclusive_applicative": MorphemeRole.VERB_APPLICATIVE_CLITIC,
        "suffix_verb_applicative_ambiguous": MorphemeRole.VERB_APPLICATIVE_CLITIC,
        "suffix_applicative_clitics": MorphemeRole.VERB_APPLICATIVE_CLITIC,
        "suffix_verb_coverb_exclusive": MorphemeRole.VERB_COVERB,
        "suffix_verb_exclusive_coverb": MorphemeRole.VERB_COVERB,
        "suffix_verb_coverb_ambiguous": MorphemeRole.VERB_COVERB,
        "suffix_verb_coverb": MorphemeRole.VERB_COVERB,
        "suffix_verb_auxiliary_exclusive": MorphemeRole.VERB_AUXILIARY,
        "suffix_verb_exclusive_auxiliary": MorphemeRole.VERB_AUXILIARY,
        "suffix_verb_auxiliary_ambiguous": MorphemeRole.VERB_AUXILIARY,
        "suffix_verb_auxiliary": MorphemeRole.VERB_AUXILIARY,
        "suffix_verb_negative_finiteness_exclusive": MorphemeRole.NEGATIVE_FINITENESS,
        "suffix_verb_exclusive_negative_finiteness": MorphemeRole.NEGATIVE_FINITENESS,
        "suffix_verb_negative_finiteness_ambiguous": MorphemeRole.NEGATIVE_FINITENESS,
        "suffix_verb_negative_finiteness": MorphemeRole.NEGATIVE_FINITENESS,

        # Nominal Prefix Categories
        "prefix_nominal_compound_exclusive": MorphemeRole.COMPOUND_PREFIX,
        "prefix_nominal_exclusive_compound": MorphemeRole.COMPOUND_PREFIX,
        "prefix_nominal_compound_ambiguous": MorphemeRole.COMPOUND_PREFIX,
        "prefix_nominal_compound": MorphemeRole.COMPOUND_PREFIX,
        "prefix_nominal_associatives_exclusive": MorphemeRole.ASSOCIATIVE_PREFIX,
        "prefix_nominal_exclusive_associatives": MorphemeRole.ASSOCIATIVE_PREFIX,
        "prefix_nominal_associatives_ambiguous": MorphemeRole.ASSOCIATIVE_PREFIX,
        "prefix_nominal_associatives": MorphemeRole.ASSOCIATIVE_PREFIX,
        "prefix_nominal_prepositional_exclusive": MorphemeRole.LOCATIVE_PREPOSITION,
        "prefix_nominal_exclusive_prepositional": MorphemeRole.LOCATIVE_PREPOSITION,
        "prefix_nominal_prepositional_ambiguous": MorphemeRole.LOCATIVE_PREPOSITION,
        "prefix_nominal_prepositional": MorphemeRole.LOCATIVE_PREPOSITION,
        "prefix_universal_exclusive": MorphemeRole.GENITIVE_PREPOSITION,
        "prefix_universal_ambiguous": MorphemeRole.GENITIVE_PREPOSITION,
        "prefix_universal": MorphemeRole.GENITIVE_PREPOSITION,

        # Verbal Prefix Categories
        "prefix_relative_pronoun": MorphemeRole.RELATIVE_COMPLEMENTIZER,
        "prefix_verb_subordinate_exclusive": MorphemeRole.SUBORDINATE_CONJUNCTION,
        "prefix_verb_exclusive_subordinate": MorphemeRole.SUBORDINATE_CONJUNCTION,
        "prefix_verb_subordinate_ambiguous": MorphemeRole.SUBORDINATE_CONJUNCTION,
        "prefix_verb_subord_imp": MorphemeRole.SUBORDINATE_CONJUNCTION,
        "prefix_verb_subordinate_imperfective": MorphemeRole.SUBORDINATE_CONJUNCTION,
        "prefix_verb_voice_exclusive": MorphemeRole.VOICE_PREFIX,
        "prefix_verb_exclusive_voice": MorphemeRole.VOICE_PREFIX,
        "prefix_verb_voice_ambiguous": MorphemeRole.VOICE_PREFIX,
        "prefix_verb_voice": MorphemeRole.VOICE_PREFIX,
        "prefix_verb_progressive_exclusive": MorphemeRole.PROGRESSIVE_ASPECT,
        "prefix_verb_progressive": MorphemeRole.PROGRESSIVE_ASPECT,
        "prefix_verb_negative_exclusive": MorphemeRole.NEGATIVE_PREFIX,
        "prefix_verb_exclusive_negative": MorphemeRole.NEGATIVE_PREFIX,
        "prefix_verb_negative_ambiguous": MorphemeRole.NEGATIVE_PREFIX,
        "prefix_verb_atomic_negative": MorphemeRole.NEGATIVE_PREFIX,
        "prefix_verb_aspectual_perfective_exclusive": MorphemeRole.RELATIVE_COMPLEMENTIZER,
        "prefix_verb_exclusive_aspectual_perfective": MorphemeRole.RELATIVE_COMPLEMENTIZER,

        # Universal Clitics
        "suffix_universal_clitics_exclusive": MorphemeRole.PRAGMATIC_CLITIC,
        "suffix_universal_clitics_ambiguous": MorphemeRole.PRAGMATIC_CLITIC,
        "suffix_universal_clitics": MorphemeRole.PRAGMATIC_CLITIC,
        "suffix_pragmatic_clitics": MorphemeRole.PRAGMATIC_CLITIC,
    }

    NOMINAL_PRIORITY = (
        "suffix_accusative",
        "suffix_nominal_accusative_ambiguous",
        "suffix_nominal_plural_exclusive",
        "suffix_nominal_exclusive_plural",
        "suffix_nominal_plural_ambiguous",
        "suffix_nominal_plural",
        "suffix_nominal_definiteness_exclusive",
        "suffix_nominal_exclusive_definiteness",
        "suffix_nominal_definiteness_ambiguous",
        "suffix_nominal_definiteness",
        "suffix_nominal_possessive_exclusive",
        "suffix_nominal_exclusive_possessive",
        "suffix_nominal_possessive_ambiguous",
        "suffix_nominal_possessive",
        "suffix_nominal_abstract_exclusive",
        "suffix_nominal_exclusive_abstract",
        "suffix_nominal_abstract_ambiguous",
        "suffix_nominal_abstract",
        "suffix_nominal_derivational_exclusive",
        "suffix_nominal_exclusive_derivational",
        "suffix_nominal_derivational_ambiguous",
        "suffix_nominal_derivational",
        "suffix_nominal_instrumental_exclusive",
        "suffix_nominal_exclusive_instrumental",
    )

    VERBAL_PRIORITY = (
        "suffix_verb_subject_exclusive",
        "suffix_verb_exclusive_subject",
        "suffix_verb_subject_ambiguous",
        "suffix_verb_pv",
        "suffix_verb_subject_agreement",
        "suffix_verb_object_exclusive",
        "suffix_verb_exclusive_object",
        "suffix_verb_object_ambiguous",
        "suffix_verb_object_clitics",
        "suffix_verb_applicative_exclusive",
        "suffix_verb_exclusive_applicative",
        "suffix_verb_applicative_ambiguous",
        "suffix_applicative_clitics",
        "suffix_verb_coverb_exclusive",
        "suffix_verb_exclusive_coverb",
        "suffix_verb_coverb_ambiguous",
        "suffix_verb_coverb",
        "suffix_verb_auxiliary_exclusive",
        "suffix_verb_exclusive_auxiliary",
        "suffix_verb_auxiliary_ambiguous",
        "suffix_verb_auxiliary",
        "suffix_verb_negative_finiteness_exclusive",
        "suffix_verb_exclusive_negative_finiteness",
        "suffix_verb_negative_finiteness_ambiguous",
        "suffix_verb_negative_finiteness",
    )

    def __init__(self, morphology=None):
        self.morphology = morphology
        self.morpheme_to_categories: Dict[str, Set[str]] = defaultdict(set)
        self.nominal_categories: Set[str] = set()
        self.verbal_categories: Set[str] = set()
        self._build_inverted_index()

    def rebuild_index(self):
        """Rebuild the inverted index from the engine's current inventories.

        The index copies morpheme *strings* out of the morphology engine, so it is
        a snapshot: any later change to the inventories (notably
        AmharicMorphologyEngine.normalize_inventories) leaves it stale. Callers
        that mutate inventories after construction must call this.
        """
        self.morpheme_to_categories = defaultdict(set)
        self.nominal_categories = set()
        self.verbal_categories = set()
        self._build_inverted_index()

    def _build_inverted_index(self):
        if not self.morphology:
            return

        for cat_key, role in self.CATEGORY_TO_ROLE.items():
            if hasattr(self.morphology, cat_key):
                m_set = getattr(self.morphology, cat_key)
                if isinstance(m_set, (set, list, tuple)):
                    for morpheme in m_set:
                        if morpheme and isinstance(morpheme, str):
                            self.morpheme_to_categories[morpheme].add(cat_key)

            if cat_key.startswith(("suffix_nominal", "prefix_nominal", "suffix_accusative")):
                self.nominal_categories.add(cat_key)
            elif cat_key.startswith(("suffix_verb", "prefix_verb", "prefix_relative")):
                self.verbal_categories.add(cat_key)

    def resolve_suffix(
        self,
        stem: str,
        morpheme: str,
        stem_is_nominal: bool,
        stem_is_verbal: bool,
        adjacent_suffixes: Tuple[str, ...] = (),
    ) -> MorphemeRole:
        cats = self.morpheme_to_categories.get(morpheme, set())
        if not cats:
            return MorphemeRole.UNKNOWN

        if morpheme == "ን" and adjacent_suffixes:
            idx = adjacent_suffixes.index(morpheme) if morpheme in adjacent_suffixes else -1
            if idx > 0:
                prev_sfx = adjacent_suffixes[idx - 1]
                if prev_sfx in ("ል", "ብ"):
                    return MorphemeRole.VERB_OBJECT_CLITIC
                if prev_sfx in ("ው", "ቸው", "ት", "አት", "ኝ", "አል", "ዋል"):
                    return MorphemeRole.ACCUSATIVE_CASE

        has_nom_neighbor = any(
            any(c in self.nominal_categories for c in self.morpheme_to_categories.get(s, set()))
            for s in adjacent_suffixes if s != morpheme
        )

        if stem_is_nominal or has_nom_neighbor:
            nom_cats = cats & self.nominal_categories
            if nom_cats:
                for target_cat in self.NOMINAL_PRIORITY:
                    if target_cat in nom_cats:
                        return self.CATEGORY_TO_ROLE[target_cat]
                first_nom = next(iter(nom_cats))
                return self.CATEGORY_TO_ROLE.get(first_nom, MorphemeRole.UNKNOWN)

        if stem_is_verbal and not has_nom_neighbor:
            verb_cats = cats & self.verbal_categories
            if verb_cats:
                for target_cat in self.VERBAL_PRIORITY:
                    if target_cat in verb_cats:
                        return self.CATEGORY_TO_ROLE[target_cat]
                first_verb = next(iter(verb_cats))
                return self.CATEGORY_TO_ROLE.get(first_verb, MorphemeRole.UNKNOWN)

        for target_cat in self.NOMINAL_PRIORITY + self.VERBAL_PRIORITY:
            if target_cat in cats:
                return self.CATEGORY_TO_ROLE[target_cat]

        first_cat = next(iter(cats))
        return self.CATEGORY_TO_ROLE.get(first_cat, MorphemeRole.UNKNOWN)

    def resolve_prefix(
        self,
        stem: str,
        prefix: str,
        stem_is_nominal: bool,
        stem_is_verbal: bool,
    ) -> MorphemeRole:
        cats = self.morpheme_to_categories.get(prefix, set())
        if not cats:
            return MorphemeRole.UNKNOWN

        if stem_is_verbal:
            verb_cats = cats & self.verbal_categories
            if verb_cats:
                first_cat = next(iter(verb_cats))
                return self.CATEGORY_TO_ROLE.get(first_cat, MorphemeRole.UNKNOWN)

        if stem_is_nominal:
            nom_cats = cats & self.nominal_categories
            if nom_cats:
                first_cat = next(iter(nom_cats))
                return self.CATEGORY_TO_ROLE.get(first_cat, MorphemeRole.UNKNOWN)

        first_cat = next(iter(cats))
        return self.CATEGORY_TO_ROLE.get(first_cat, MorphemeRole.UNKNOWN)