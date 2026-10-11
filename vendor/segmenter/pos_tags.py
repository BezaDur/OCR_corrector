"""Canonical POS tag notation for the segmenter.

Notation grammar
----------------
``:`` marks a **subset (set) relation**; ``|`` marks **disjunction**.

* The superfamily is spelled out on every member of a disjunction::

      NOM:CN|VB:JUSSIVE_IMPERATIVE
      VB:PERFECTIVE|VB:IMPERFECTIVE
      NOM:CN|NOM:PN

* A subset chain keeps a single parent; the head is not repeated::

      NOM:CN                    (NOM > CN)
      VB:PERFECTIVE:RELATIVE    (VB > PERFECTIVE > RELATIVE)

* Subset chains and disjunctions never mix inside one chain segment.

Families
--------
==============  ==========================================================
``NOM``         nominals: nouns ``NOM:CN`` / ``NOM:PN`` / ``NOM:MN``,
                ``NOM:ADJ``, ``NOM:PRO``, bare ``NOM``
``VB``          verbals: ``VB:PERFECTIVE``, ``VB:IMPERFECTIVE``,
                ``VB:JUSSIVE_IMPERATIVE``, ``VB:COVERB[:COMPOUND]``,
                ``VB:PERFECTIVE:RELATIVE``, ``VB:IMPERFECTIVE:RELATIVE``,
                ``VB:AUX``, ``VB:IDEO``, bare ``VB``
orthogonal      ``ADP``, ``ADV``, ``CONJ``, ``INT``, ``NUM``, ``VN``
                (and ``UN``)
==============  ==========================================================

``ADP``, ``ADV``, ``CONJ``, ``INT``, ``NUM`` and ``VN`` are *not* members of
the nominal family: they are orthogonal top-level families.  Only nouns
(``NOM:CN`` / ``NOM:PN`` / ``NOM:MN``), adjectives (``NOM:ADJ``) and pronouns
(``NOM:PRO``) are nominals.  Numerals and quantifiers are ``NUM``: they are a
family of their own, not a kind of noun.

Legacy spellings
----------------
The runtime and the gold data contain older spellings (``CN``, ``MN``,
``PRO``, ``CN/VN``, ``NOM:ADP``, ``VRB:AUX``, ``VERBAL:PERFECTIVE``,
``N:VB:*``, ``NV:*``, ``CN/VB:*``, the dropped noun node (``NOM:N:CN`` →
``NOM:CN``) and the colon disjunctions ``VB:IMPERFECTIVE:JUSSIVE`` /
``VB:PERFECTIVE:JUSSIVE``).
:func:`canonical_tag` rewrites all of them, so consumers can migrate
progressively and old evaluation files keep working.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Family heads
# ---------------------------------------------------------------------------
NOMINAL = "NOM"
VERBAL = "VB"
UNKNOWN = "UN"

#: Top-level families that are neither nominal nor verbal.
ORTHOGONAL_FAMILIES: Tuple[str, ...] = ("ADP", "ADV", "CONJ", "INT", "NUM", "VN")

#: Closed-class families (invariant membership, no productive morphology).
CLOSED_FAMILIES: Tuple[str, ...] = ("ADP", "ADV", "CONJ", "INT", "NUM")

#: Noun subtypes.  ``NUM`` is not one of them: numerals and quantifiers are
#: their own top-level family, not nouns.
NOUN_SUBTYPES: Tuple[str, ...] = ("CN", "PN", "MN")


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------
def noun(subtype: str = "CN", head: str = NOMINAL) -> str:
    """``NOM:CN`` (common noun), ``NOM:PN`` (proper), ``NOM:MN`` (mass)."""
    return f"{head}:{subtype}"


def nominal() -> str:
    """Bare nominal family tag."""
    return NOMINAL


def adjective() -> str:
    """``NOM:ADJ``."""
    return f"{NOMINAL}:ADJ"


def pronoun() -> str:
    """``NOM:PRO``."""
    return f"{NOMINAL}:PRO"


def verb(subtag: Optional[str] = None, head: str = VERBAL) -> str:
    """``VB`` or ``VB:<subtag>``."""
    return head if not subtag else f"{head}:{subtag}"


def auxiliary() -> str:
    """``VB:AUX``."""
    return verb("AUX")


def ideophone() -> str:
    """``VB:IDEO`` (ideophones sit on the verbal side)."""
    return verb("IDEO")


def numeral() -> str:
    """``NUM`` (numeral / quantifier; its own top-level family)."""
    return "NUM"


def subset(parent: str, child: str) -> str:
    """Append a subset level: ``subset("VB:PERFECTIVE", "RELATIVE")``."""
    return f"{parent}:{child}"


def disjoin(*members: str) -> str:
    """Join complete tags with ``|`` (superfamily repeated on each member)."""
    return "|".join(m for m in members if m)


def disjoin_all(members: Iterable[str]) -> str:
    return disjoin(*members)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def disjuncts(tag: Optional[str]) -> Tuple[str, ...]:
    """Split a tag into its disjuncts (no canonicalization)."""
    if not tag:
        return ()
    return tuple(part.strip() for part in str(tag).split("|") if part.strip())


def family_of(tag: str) -> str:
    """Return the head family of a single (non-disjoined) chain."""
    if not tag:
        return UNKNOWN
    return tag.split(":", 1)[0] if ":" in tag else tag


def tag_families(tag: Optional[str]) -> FrozenSet[str]:
    """Families named by a tag; one entry per disjunct."""
    return frozenset(family_of(part) for part in disjuncts(tag))


def is_disjunction(tag: Optional[str]) -> bool:
    return len(disjuncts(tag)) > 1


def is_nominal(tag: Optional[str]) -> bool:
    return NOMINAL in tag_families(tag)


def is_verbal(tag: Optional[str]) -> bool:
    return VERBAL in tag_families(tag)


def is_noun(tag: Optional[str]) -> bool:
    """True for ``NOM:CN`` / ``NOM:PN`` / ``NOM:MN`` (a noun proper, not an
    adjective, pronoun, or numeral)."""
    return any(
        part.split(":", 1)[1] in NOUN_SUBTYPES
        for part in disjuncts(tag)
        if part.startswith(f"{NOMINAL}:")
    )


def noun_subtype(tag: Optional[str]) -> Optional[str]:
    """``CN`` / ``PN`` / ``MN`` for a ``NOM:<subtype>`` tag."""
    for part in disjuncts(tag):
        if part.startswith(f"{NOMINAL}:"):
            subtype = part.split(":", 1)[1]
            if subtype in NOUN_SUBTYPES:
                return subtype
    return None


def is_closed(tag: Optional[str]) -> bool:
    """True for closed-class families (orthogonal heads and AUX)."""
    parts = disjuncts(tag)
    if any(part in CLOSED_FAMILIES for part in parts):
        return True
    return any(part == f"{VERBAL}:AUX" for part in parts)


def closed_subtype(tag: Optional[str]) -> str:
    """Bare closed-class subtype (``ADP``, ``ADV``, ``AUX``, ...)."""
    for part in disjuncts(tag):
        if part in CLOSED_FAMILIES:
            return part
        if part == f"{VERBAL}:AUX":
            return "AUX"
        if part.startswith(f"{VERBAL}:"):
            return part.split(":", 1)[1]
    return ""


# ---------------------------------------------------------------------------
# Legacy -> canonical
# ---------------------------------------------------------------------------
_LEGACY_EXACT = {
    # Nominal spellings
    "NOMINAL": NOMINAL,
    "CN": noun("CN"),
    "PN": noun("PN"),
    # Pre-notation "MN" was the closed-class *quantifier*, which is now the
    # NUM family; NOM:MN (bare) is the mass noun, so it is left canonical.
    "MN": numeral(),
    "NOM:CN": noun("CN"),
    "NOM:PN": noun("PN"),
    "PRO": pronoun(),
    "NOM:PRO": pronoun(),
    "ADJ": adjective(),
    "AN": adjective(),
    "CN/VN": disjoin(noun("CN"), "VN"),
    "CN/PN": disjoin(noun("CN"), noun("PN")),
    # Orthogonal families (were nominal-prefixed)
    "NOM:ADP": "ADP",
    "NOM:ADV": "ADV",
    "NOM:CONJ": "CONJ",
    "NOM:INT": "INT",
    "NOM:IDEO": "VB:IDEO",
    "IDEO": "VB:IDEO",
    # Verbal spellings
    "VERBAL": VERBAL,
    "VRB:AUX": auxiliary(),
    "AUX": auxiliary(),
    "VB:VN": "VN",
    "VB:RELATIVE_PERFECTIVE": "VB:PERFECTIVE:RELATIVE",
    "VB:RELATIVE_IMPERFECTIVE": "VB:IMPERFECTIVE:RELATIVE",
    "VB:COMPOUND_PERFECT": "VB:COVERB:COMPOUND",
    "VB:COMPOUND_IMPERFECT": "VB:IMPERFECTIVE:COMPOUND",
    "RELATIVE_PERFECTIVE": "VB:PERFECTIVE:RELATIVE",
    "RELATIVE_IMPERFECTIVE": "VB:IMPERFECTIVE:RELATIVE",
    # Legacy colon disjunctions
    "VB:IMPERFECTIVE:JUSSIVE": "VB:IMPERFECTIVE|VB:JUSSIVE_IMPERATIVE",
    "VB:PERFECTIVE:JUSSIVE": "VB:PERFECTIVE|VB:JUSSIVE_IMPERATIVE",
    "IMPERFECTIVE:JUSSIVE": "VB:IMPERFECTIVE|VB:JUSSIVE_IMPERATIVE",
    "PERFECTIVE:JUSSIVE": "VB:PERFECTIVE|VB:JUSSIVE_IMPERATIVE",
    # Dual spellings
    "N:VB": f"{noun('CN')}|{VERBAL}",
    "NV": f"{noun('CN')}|{VERBAL}",
    # Flats
    "PERFECTIVE": "VB:PERFECTIVE",
    "IMPERFECTIVE": "VB:IMPERFECTIVE",
    "COVERB": "VB:COVERB",
    "JUSSIVE": "VB:JUSSIVE_IMPERATIVE",
    "IMPERATIVE": "VB:JUSSIVE_IMPERATIVE",
    "JUSSIVE_IMPERATIVE": "VB:JUSSIVE_IMPERATIVE",
    "CONTINGENT": "VB:IMPERFECTIVE",
    "COMPOUND_PERFECT": "VB:COVERB:COMPOUND",
    "COMPOUND_IMPERFECT": "VB:IMPERFECTIVE:COMPOUND",
    "VERBAL:PERFECTIVE": "VB:PERFECTIVE",
    "UNKNOWN": UNKNOWN,
    "UN": UNKNOWN,
}

_LEGACY_PREFIXES = (
    ("VERBAL:", f"{VERBAL}:"),
    ("VRB:", f"{VERBAL}:"),
    # The noun node was dropped as redundant: NOM:N:CN -> NOM:CN.
    (f"{NOMINAL}:N:", f"{NOMINAL}:"),
)

_CANONICAL_HEADS = (NOMINAL, VERBAL) + ORTHOGONAL_FAMILIES + (UNKNOWN,)


def is_canonical(tag: Optional[str]) -> bool:
    """True when every disjunct already uses a canonical head and spelling."""
    parts = disjuncts(tag)
    if not parts:
        return False
    for part in parts:
        head = family_of(part)
        if head not in _CANONICAL_HEADS:
            return False
        if head == NOMINAL and part not in (NOMINAL, adjective(), pronoun()):
            subtype = part.split(":", 1)[1] if ":" in part else ""
            if subtype not in NOUN_SUBTYPES:
                return False
    return True


def canonical_tag(tag: Optional[str]) -> str:
    """Rewrite any legacy spelling into the canonical notation.

    Idempotent: canonical input is returned unchanged (modulo whitespace).
    """
    if not tag:
        return ""
    out = []
    for part in disjuncts(tag):
        out.append(_canonical_part(part))
    return "|".join(out)


#: Legacy dual heads: a nominal reading prefixed to a verbal chain.
_DUAL_PREFIXES = ("N:VB:", "CN/VB:", "NV:")


def _canonical_part(part: str) -> str:
    if part in _LEGACY_EXACT:
        return _LEGACY_EXACT[part]

    for dual_prefix in _DUAL_PREFIXES:
        if part.startswith(dual_prefix):
            rest = part[len(dual_prefix):]
            verbal_readings = disjuncts(canonical_tag(rest))
            if not verbal_readings:
                return disjoin(noun("CN"), VERBAL)
            return disjoin(noun("CN"), *verbal_readings)

    for legacy_prefix, canonical_prefix in _LEGACY_PREFIXES:
        if part.startswith(legacy_prefix):
            rewritten = canonical_prefix + part[len(legacy_prefix):]
            return _LEGACY_EXACT.get(rewritten, rewritten)
    return part


__all__ = [
    "NOMINAL",
    "VERBAL",
    "UNKNOWN",
    "ORTHOGONAL_FAMILIES",
    "CLOSED_FAMILIES",
    "NOUN_SUBTYPES",
    "noun",
    "nominal",
    "adjective",
    "pronoun",
    "verb",
    "auxiliary",
    "ideophone",
    "subset",
    "disjoin",
    "disjoin_all",
    "disjuncts",
    "family_of",
    "tag_families",
    "is_disjunction",
    "is_nominal",
    "is_verbal",
    "is_noun",
    "noun_subtype",
    "is_closed",
    "closed_subtype",
    "is_canonical",
    "canonical_tag",
]
