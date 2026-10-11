"""Canonical normalization of POS tags to the four Layer 1 macro classes.

The runtime tagger emits canonical POS tags (see ``pos_tags.py``): ``:`` marks
a subset relation and ``|`` a disjunction between complete tags, with the
superfamily repeated on each member.

This module keeps the historical four-box *macro* view used by the evaluation
reports -- ``NOMINAL``, ``VERBAL``, ``NV`` (a tag offering both readings) and
``UN``.  The boxes are aggregates, not tags: the canonical tag is the source of
truth, and a nominal/verbal disjunction such as
``NOM:CN|VB:JUSSIVE_IMPERATIVE`` reports here as ``NV``.

Legacy spellings (``CN``, ``PRO``, ``NOM:ADP``, ``VRB:AUX``,
``VERBAL:PERFECTIVE``, ``N:VB:*``, ``NV:*``, ``CN/VB:*``) are still accepted so
older evaluation files and diagnostics keep working; they are canonicalized
first through :func:`pos_tags.canonical_tag`.
"""

from typing import Optional

import pos_tags


MACRO_CATEGORIES = ("NOMINAL", "VERBAL", "NV", "UN")

#: Orthogonal families are non-verbal for the four-box view: the nominal
#: umbrella covers nouns, adjectives, pronouns and numerals, and the closed
#: classes sat under it before the notation change (``NOM:ADP``, ...).
_NOMINAL_FAMILIES = frozenset({"NOM", "ADP", "ADV", "CONJ", "INT", "NUM", "VN"})

_VERBAL_FLAT_TAGS = frozenset(
    {
        "PERFECTIVE",
        "IMPERFECTIVE",
        "COVERB",
        "JUSSIVE_IMPERATIVE",
        "JUSSIVE",
        "IMPERATIVE",
        "COMPOUND_PERFECT",
        "COMPOUND_IMPERFECT",
        "CONTINGENT",
        "RELATIVE_PERFECTIVE",
        "RELATIVE_IMPERFECTIVE",
        "SUBORDINATE_IMPERFECTIVE",
        "SUBORDINATE_PERFECTIVE",
        "VB",
        "AUX",
        "VERBAL",
    }
)


def _family_macro(family: str) -> str:
    if family == pos_tags.VERBAL:
        return "VERBAL"
    if family in _NOMINAL_FAMILIES:
        return "NOMINAL"
    return "UN"


def macro_category_of_tag(tag: Optional[str]) -> str:
    """Normalize a POS tag to ``NOMINAL``, ``VERBAL``, ``NV``, or ``UN``.

    A disjunction maps to ``NV`` when it offers both a nominal and a verbal
    reading, to the single family when all readings agree, and to ``UN`` when
    any reading is unresolved.  ``VN`` counts as nominal here; the
    verb-paradigm evaluator applies its own corpus-specific convention.
    """
    if not tag:
        return "UN"

    canonical = pos_tags.canonical_tag(str(tag).strip())
    if not canonical:
        return "UN"

    if pos_tags.is_disjunction(canonical):
        readings = {
            macro_category_of_tag(part) for part in pos_tags.disjuncts(canonical)
        }
        if not readings or "UN" in readings:
            return "UN"
        if readings == {"NOMINAL", "VERBAL"} or "NV" in readings:
            return "NV"
        if len(readings) == 1:
            return next(iter(readings))
        return "UN"

    # Legacy spellings that canonical_tag() leaves untouched (flats).
    if canonical in _VERBAL_FLAT_TAGS:
        return "VERBAL"
    if canonical in ("NOMINAL", "NOM"):
        return "NOMINAL"
    if canonical in ("UN", "UNKNOWN"):
        return "UN"

    family = pos_tags.family_of(canonical)
    if family == pos_tags.VERBAL:
        return "VERBAL"
    if family in _NOMINAL_FAMILIES:
        return "NOMINAL"
    return "UN"


__all__ = ["MACRO_CATEGORIES", "macro_category_of_tag"]
