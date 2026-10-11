"""
AmharicMorphologyEngine: Declarative, Granular POS Morphological Engine.
Equipped with Robust Multi-Path Whitelist Loading, Universal Proclitic Disentanglement,
5-Class Paradigm-Aware Cohort Generation, and Reduplicative Quadriradical Integrity Gates.
"""

from __future__ import annotations
import csv
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from morph_types import DecomposeContext, MorphemesConfig, MorphSplit, VerbalShape, SimpleLRUCache
import pos_tags
from phonology import (
    VOICE_PREFIXES,
    AmharicBoundaryPhonology,
    consonantal_skeleton,
    split_voice_prefix,
)
from affix_peeler import AffixPeeler
from lemma_candidate_generator import LemmaCandidateGenerator
from project_paths import (
    CONTINGENT_JUSSIVE_POLICY_PATH,
    EXPERIENCER_VERBS_PATH,
    DATA_DIR,
    MORPHEMES_CLUSTERS_PATH,
    MORPHEMES_PATH,
    PARADIGM_MATRIX_PATH,
    PERFECTIVE_IMPERATIVE_POLICY_PATH,
)

logger = logging.getLogger(__name__)

DEFAULT_2CHAR_WORDS_TSV = "2_char_true_words.tsv"
#DEFAULT_2CHAR_WORDS_FILENAME = "2_char_true_words.txt"  # Legacy fallback
DEFAULT_MACRO_CLUSTERS_FILENAME = "morphemes_clusters.json"

# ============================================================================
# MODULE-LEVEL BI-RADICAL REGISTRY & CACHE
# ============================================================================
# Category-typed hints still read by the runtime.
TWO_CHAR_PERFECTIVE: Set[str] = set()
TWO_CHAR_CONTINGENT_STEMS: Set[str] = set()
TWO_CHAR_CONTINGENTS: Set[str] = set()

# Flat raw-text whitelist: every 2-character token appearing anywhere in the TSV.
ALL_LICENSED_2CHAR_TOKENS: Set[str] = set()

def load_biradical_registry(base_dir: Optional[Union[str, Path]] = None) -> None:
    """
    Loads the 2-radical TSV table into O(1) in-memory sets.

    Policy: the TSV is a RAW TEXT whitelist of genuine 2-character Amharic
    words.  Its column organization is NOT trusted -- the shipped file places
    most entries in the first column regardless of their grammatical category
    (ቤት, ሰው, ልብ are all listed there as if they were perfective verbs), so the
    file is scanned as flat text and every non-empty 2-character Ethiopic field
    licenses that word.  Category decisions are made later by the phonology and
    the nominal/verbal cohort evidence, never by the TSV layout.

    Only the category hints the runtime actually reads are retained:
      - Column 3: Contingent stem after stripping ይ-/ያ- (ሆን, ሄድ, ቆም...)
      - Column 0: shipped paradigm list (used for the 3MS-perfective check)
    """
    global TWO_CHAR_PERFECTIVE
    global TWO_CHAR_CONTINGENT_STEMS, TWO_CHAR_CONTINGENTS, ALL_LICENSED_2CHAR_TOKENS

    # Prefer the canonical shared-data location; retain explicit and legacy
    # search paths for callers that supply their own resource directory.
    search_dirs = []
    if base_dir:
        search_dirs.append(Path(base_dir))
    search_dirs.append(DATA_DIR)
    search_dirs.append(Path(os.path.dirname(__file__)))
    search_dirs.append(Path.cwd())

    tsv_path: Optional[Path] = None
    for d in search_dirs:
        candidate = d / DEFAULT_2CHAR_WORDS_TSV
        if candidate.is_file():
            tsv_path = candidate
            break

    if tsv_path:
        logger.info("Loading 2-radical registry from TSV: %s", tsv_path)
        TWO_CHAR_PERFECTIVE.clear()
        TWO_CHAR_CONTINGENT_STEMS.clear()
        TWO_CHAR_CONTINGENTS.clear()
        ALL_LICENSED_2CHAR_TOKENS.clear()
        with open(tsv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            try:
                next(reader)  # header row
            except StopIteration:
                return

            for row in reader:
                if not row:
                    continue

                # RAW TEXT SEARCH: every field of every row is a candidate word.
                for field in row:
                    val = field.strip()
                    if _is_ethiopic_2char(val):
                        ALL_LICENSED_2CHAR_TOKENS.add(val)

                # Col 0: shipped paradigm list (still read by the 3MS-perfective check)
                if row[0].strip():
                    val = row[0].strip()
                    if len(val) == 2:
                        TWO_CHAR_PERFECTIVE.add(val)

                # Col 3: Contingent stem (strip ይ-/ያ- for 3-char forms; keep 2-char coalesced forms)
                if len(row) > 3 and row[3].strip():
                    val = row[3].strip()
                    if len(val) == 3 and val.startswith(("ይ", "ያ")):
                        TWO_CHAR_CONTINGENT_STEMS.add(val[1:])
                    elif len(val) == 2:
                        TWO_CHAR_CONTINGENTS.add(val)

        logger.info(
            "2-Radical registry initialized: %d licensed tokens (raw text scan).",
            len(ALL_LICENSED_2CHAR_TOKENS),
        )
    else:
        logger.warning("Warning: %s could not be found across candidate directories.", DEFAULT_2CHAR_WORDS_TSV)


def _is_ethiopic_2char(stem: str) -> bool:
    if len(stem) != 2:
        return False
    return all(0x1200 <= ord(ch) <= 0x137F for ch in stem)


# ---------------------------------------------------------------------------
# Shipped verb paradigm matrix (surface-form licences)
# ---------------------------------------------------------------------------
_PARADIGM_MATRIX_FORMS: Set[str] = set()
_PARADIGM_MATRIX_LOADED = False


def paradigm_matrix_forms() -> Set[str]:
    """All surface forms listed in the shipped verb paradigm matrix.

    The matrix (``data/10_column_verbs.tsv``) is the repository's verb
    inventory: every column is a paradigm cell, so a surface form appearing
    anywhere in it is an attested verb form.  This is the licensor that
    separates genuine bare nominal/verbal homographs (ራስ "head" / "forget!",
    መርዝ "poison" / "poison!") from plain nouns (ቤት, ሰው) and deverbal nouns
    (መምህር, ጭንቀት), which are not paradigm forms.
    """
    global _PARADIGM_MATRIX_FORMS, _PARADIGM_MATRIX_LOADED
    if _PARADIGM_MATRIX_LOADED:
        return _PARADIGM_MATRIX_FORMS

    forms: Set[str] = set()
    try:
        with open(PARADIGM_MATRIX_PATH, mode="r", encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh, delimiter="\t"):
                for field in row:
                    value = field.strip()
                    if value and value != "-":
                        forms.add(value)
    except OSError:
        logger.warning("Paradigm matrix not found at %s", PARADIGM_MATRIX_PATH)

    _PARADIGM_MATRIX_FORMS = forms
    _PARADIGM_MATRIX_LOADED = True
    logger.info("Paradigm matrix initialized: %d surface forms.", len(forms))
    return _PARADIGM_MATRIX_FORMS


#: Paradigm-matrix column header -> canonical verbal reading.
_PARADIGM_COLUMN_TAGS: Dict[str, str] = {
    "Simple Perfect": "VB:PERFECTIVE",
    "Coverb": "VB:COVERB",
    "Compound Perfect": "VB:COVERB:COMPOUND",
    "Contingent (simple imperfect)": "VB:IMPERFECTIVE",
    "Compound Imperfect": "VB:IMPERFECTIVE:COMPOUND",
    "Jussive": "VB:JUSSIVE_IMPERATIVE",
    "Imperative": "VB:JUSSIVE_IMPERATIVE",
    "IMP_passive": "VB:JUSSIVE_IMPERATIVE",
    "VerbalNoun": "VN",
    "Perfective relative": "VB:PERFECTIVE:RELATIVE",
    "Imperfective relative": "VB:IMPERFECTIVE:RELATIVE",
    "prospective": "VB:IMPERFECTIVE:PROSPECTIVE",
    "temporal": "VB:IMPERFECTIVE:TEMPORAL",
    "purposive": "VB:IMPERFECTIVE:PURPOSIVE",
    "conditional": "VB:IMPERFECTIVE:CONDITIONAL",
    "terminative": "VB:IMPERFECTIVE:TERMINATIVE",
    "Ablative": "VB:PERFECTIVE:ABLATIVE",
    "Similative(temporal_immediate)": "VB:PERFECTIVE:SIMILATIVE",
    "causal": "VB:PERFECTIVE:CAUSAL",
    "temporal_past": "VB:PERFECTIVE:TEMPORAL_PAST",
}

_PARADIGM_MATRIX_READINGS: Dict[str, Tuple[str, ...]] = {}
_PARADIGM_MATRIX_READINGS_LOADED = False


def paradigm_matrix_readings(form: str) -> Tuple[str, ...]:
    """Canonical verbal readings licensed for a surface form by the matrix.

    The matrix headers name the paradigm cell, so a form listed in the
    ``Imperative`` column is licensed as ``VB:JUSSIVE_IMPERATIVE`` and one in
    ``Coverb`` as ``VB:COVERB``.  Verdicts are cached per form.
    """
    global _PARADIGM_MATRIX_READINGS, _PARADIGM_MATRIX_READINGS_LOADED
    if not _PARADIGM_MATRIX_READINGS_LOADED:
        readings: Dict[str, set] = {}
        try:
            with open(PARADIGM_MATRIX_PATH, mode="r", encoding="utf-8", newline="") as fh:
                reader = csv.reader(fh, delimiter="\t")
                headers = [h.strip() for h in next(reader, [])]
                for row in reader:
                    for index, field in enumerate(row):
                        value = field.strip()
                        if not value or value == "-" or index >= len(headers):
                            continue
                        tag = _PARADIGM_COLUMN_TAGS.get(headers[index])
                        if tag:
                            readings.setdefault(value, set()).add(tag)
        except (OSError, StopIteration):
            logger.warning("Paradigm matrix readings unavailable (%s)", PARADIGM_MATRIX_PATH)
        _PARADIGM_MATRIX_READINGS = {
            form_key: tuple(sorted(tags)) for form_key, tags in readings.items()
        }
        _PARADIGM_MATRIX_READINGS_LOADED = True
    return _PARADIGM_MATRIX_READINGS.get(form, ())


def is_licensed_2char_stem(stem: str) -> bool:
    """True if the token is any genuine 2-character word in the registry (raw text)."""
    if len(stem) != 2:
        return False
    return stem in ALL_LICENSED_2CHAR_TOKENS


def is_licensed_2char_perfective_stem(stem: str) -> bool:
    """Returns True for 2-character forms listed in the registry's first column."""
    if len(stem) != 2:
        return False
    return stem in TWO_CHAR_PERFECTIVE


def is_licensed_2char_verbal_stem(stem: str) -> bool:
    """
    True if verbal affixes may attach to this 2-character stem.

    The registry is a flat word whitelist, not a category table, so verbal
    licensing is the same membership test as any other 2-char licensing; the
    morphotactic firewalls decide whether a verbal parse is actually possible.
    """
    if len(stem) != 2:
        return False
    return stem in ALL_LICENSED_2CHAR_TOKENS


def is_licensed_2char_nominal(stem: str) -> bool:
    """
    True if nominal affixes may attach to this 2-character stem.

    Same flat raw-text membership test as :func:`is_licensed_2char_verbal_stem`
    (the TSV's nominal column is empty in the shipped file, which is exactly why
    the column layout must not gate licensing).
    """
    if len(stem) != 2:
        return False
    return stem in ALL_LICENSED_2CHAR_TOKENS

# Trigger registry load on module import
load_biradical_registry()

# =========================================================================
# CANONICAL ETHIO-SEMITIC VOCALIC TEMPLATES (LOADED FROM JSON)
# =========================================================================
DEFAULT_LESLAU_TEMPLATES_FILENAME = "leslau_templates.json"


def load_leslau_templates(base_dir: Optional[Union[str, Path]] = None) -> Dict[str, Set[Tuple[int, ...]]]:
    """Loads canonical Semitic vocalic templates from leslau_templates.json."""
    search_dirs = []
    if base_dir:
        search_dirs.append(Path(base_dir))
    search_dirs.append(DATA_DIR)
    search_dirs.append(Path(os.path.dirname(__file__)))
    search_dirs.append(Path.cwd())

    json_path = None
    for d in search_dirs:
        cand = d / DEFAULT_LESLAU_TEMPLATES_FILENAME
        if cand.is_file():
            json_path = cand
            break

    if not json_path:
        raise FileNotFoundError(
            f"Critical: {DEFAULT_LESLAU_TEMPLATES_FILENAME} not found across candidate directories."
        )

    with open(json_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    result: Dict[str, Set[Tuple[int, ...]]] = {}
    for key, items in data.items():
        if key.startswith("_"):
            continue
        pattern_set: Set[Tuple[int, ...]] = set()
        for entry in items:
            if isinstance(entry, dict) and "pattern" in entry:
                pattern_set.add(tuple(entry["pattern"]))
            elif isinstance(entry, (list, tuple)):
                pattern_set.add(tuple(entry))
        result[key.upper()] = pattern_set

    return result


# Initialize Sets Dynamically from JSON (Supports both old and new schema keys)
_TEMPLATES = load_leslau_templates()

LESLAU_COVERB_PATTERNS = _TEMPLATES.get("LESLAU_COVERB_PATTERNS", _TEMPLATES.get("EXCLUSIVE_COVERB_PATTERNS", set()))
LESLAU_PERFECTIVE_PATTERNS = _TEMPLATES.get("LESLAU_PERFECTIVE_PATTERNS", _TEMPLATES.get("EXCLUSIVE_PERFECTIVE_PATTERNS", set()))
LESLAU_INFINITIVE_PATTERNS = _TEMPLATES.get("LESLAU_INFINITIVE_PATTERNS", _TEMPLATES.get("EXCLUSIVE_INFINITIVE_PATTERNS", set()))
EXCLUSIVE_COVERB_PATTERNS = LESLAU_COVERB_PATTERNS
EXCLUSIVE_PERFECTIVE_PATTERNS = LESLAU_PERFECTIVE_PATTERNS
EXCLUSIVE_INFINITIVE_PATTERNS = LESLAU_INFINITIVE_PATTERNS

EXCLUSIVE_JUSSIVE_PATTERNS = _TEMPLATES.get("EXCLUSIVE_JUSSIVE_PATTERNS", set())
EXCLUSIVE_IMPERFECTIVE_PATTERNS = _TEMPLATES.get("EXCLUSIVE_IMPERFECTIVE_PATTERNS", set())

SYNCRETIC_IMPERFECTIVE_JUSSIVE_PATTERNS = _TEMPLATES.get(
    "SYNCRETIC_IMPERFECTIVE_JUSSIVE_PATTERNS",
    _TEMPLATES.get("SYNCRETIC_CONTINGENT_JUSSIVE_PATTERNS", set())
)
SYNCRETIC_PERFECTIVE_JUSSIVE_PATTERNS = _TEMPLATES.get("SYNCRETIC_PERFECTIVE_JUSSIVE_PATTERNS", set())
AMBIGUOUS_IMPERFECTIVE_JUSSIVE_PATTERNS = _TEMPLATES.get(
    "AMBIGUOUS_IMPERFECTIVE_JUSSIVE_PATTERNS",
    _TEMPLATES.get("AMBIGUOUS_CONTINGENT_JUSSIVE_PATTERNS", set())
)

PASSIVE_JUSSIVE_ROOT_PATTERNS = _TEMPLATES.get("PASSIVE_JUSSIVE_ROOT_PATTERNS", set())
CAUSATIVE_JUSSIVE_ROOT_PATTERNS = _TEMPLATES.get("CAUSATIVE_JUSSIVE_ROOT_PATTERNS", set())
DERIVED_PERFECTIVE_ROOT_PATTERNS = _TEMPLATES.get("DERIVED_PERFECTIVE_ROOT_PATTERNS", set())
BOUND_JUSSIVE_STEM_PATTERNS = _TEMPLATES.get("BOUND_JUSSIVE_STEM_PATTERNS", SYNCRETIC_IMPERFECTIVE_JUSSIVE_PATTERNS)

# Canonical Parent Sets (computed by union if partitioned, or loaded directly)
LESLAU_IMPERFECTIVE_PATTERNS = _TEMPLATES.get(
    "LESLAU_IMPERFECTIVE_PATTERNS",
    EXCLUSIVE_IMPERFECTIVE_PATTERNS | SYNCRETIC_IMPERFECTIVE_JUSSIVE_PATTERNS | AMBIGUOUS_IMPERFECTIVE_JUSSIVE_PATTERNS
)
LESLAU_JUSSIVE_PATTERNS = _TEMPLATES.get(
    "LESLAU_JUSSIVE_PATTERNS",
    EXCLUSIVE_JUSSIVE_PATTERNS | SYNCRETIC_IMPERFECTIVE_JUSSIVE_PATTERNS | AMBIGUOUS_IMPERFECTIVE_JUSSIVE_PATTERNS
)

UNAMBIGUOUS_JUSSIVE_ORDERS = EXCLUSIVE_JUSSIVE_PATTERNS if EXCLUSIVE_JUSSIVE_PATTERNS else (LESLAU_JUSSIVE_PATTERNS - SYNCRETIC_IMPERFECTIVE_JUSSIVE_PATTERNS)
SHAPE_D_JUSSIVE_ORDERS = (EXCLUSIVE_JUSSIVE_PATTERNS | SYNCRETIC_IMPERFECTIVE_JUSSIVE_PATTERNS) if EXCLUSIVE_JUSSIVE_PATTERNS else (LESLAU_JUSSIVE_PATTERNS - BOUND_JUSSIVE_STEM_PATTERNS)

VERBAL_RADICAL_PATTERNS: Dict[VerbalShape, Set[Tuple[int, ...]]] = {
    VerbalShape.SHAPE_A_PERFECTIVE: LESLAU_PERFECTIVE_PATTERNS,
    VerbalShape.SHAPE_B_COVERB: LESLAU_COVERB_PATTERNS,
    VerbalShape.SHAPE_C_IMPERFECTIVE: LESLAU_IMPERFECTIVE_PATTERNS,
    VerbalShape.SHAPE_D_JUSSIVE_IMPERATIVE: LESLAU_JUSSIVE_PATTERNS,
    VerbalShape.SHAPE_E_INFINITIVE: LESLAU_INFINITIVE_PATTERNS,
}


class AmharicMorphologyEngine:
    STRONG = 3
    NORMAL = 2
    WEAK = 1

    EXCLUSIVE_PERFECTIVE_PATTERNS = EXCLUSIVE_PERFECTIVE_PATTERNS
    EXCLUSIVE_JUSSIVE_PATTERNS = EXCLUSIVE_JUSSIVE_PATTERNS
    EXCLUSIVE_IMPERFECTIVE_PATTERNS = EXCLUSIVE_IMPERFECTIVE_PATTERNS
    EXCLUSIVE_INFINITIVE_PATTERNS = EXCLUSIVE_INFINITIVE_PATTERNS
    EXCLUSIVE_COVERB_PATTERNS = EXCLUSIVE_COVERB_PATTERNS
    SYNCRETIC_IMPERFECTIVE_JUSSIVE_PATTERNS = SYNCRETIC_IMPERFECTIVE_JUSSIVE_PATTERNS
    SYNCRETIC_PERFECTIVE_JUSSIVE_PATTERNS = SYNCRETIC_PERFECTIVE_JUSSIVE_PATTERNS
    AMBIGUOUS_IMPERFECTIVE_JUSSIVE_PATTERNS = AMBIGUOUS_IMPERFECTIVE_JUSSIVE_PATTERNS

    LESLAU_PERFECTIVE_PATTERNS = LESLAU_PERFECTIVE_PATTERNS
    LESLAU_INFINITIVE_PATTERNS = LESLAU_INFINITIVE_PATTERNS
    LESLAU_IMPERFECTIVE_PATTERNS = LESLAU_IMPERFECTIVE_PATTERNS
    LESLAU_JUSSIVE_PATTERNS = LESLAU_JUSSIVE_PATTERNS
    # A bare (0, 3) / (3, 3) stem is the 3MS base perfective of a 2-radical verb
    # (ሰማ, ገባ, ወጣ, በላ, ጠማ), never a command base -- Type A commands use the
    # (5, 3) grade (ስማ, ግባ).  The jussive lock below already refuses these
    # orders; the perfective lock has to claim them in exchange.
    TWO_RADICAL_BASE_PERFECTIVE_ORDERS = frozenset({(0, 3), (3, 3)})
    BOUND_JUSSIVE_STEM_PATTERNS = BOUND_JUSSIVE_STEM_PATTERNS
    UNAMBIGUOUS_JUSSIVE_ORDERS = UNAMBIGUOUS_JUSSIVE_ORDERS
    SHAPE_D_JUSSIVE_ORDERS = SHAPE_D_JUSSIVE_ORDERS
    LESLAU_COVERB_PATTERNS = LESLAU_COVERB_PATTERNS
    PASSIVE_JUSSIVE_ROOT_PATTERNS = PASSIVE_JUSSIVE_ROOT_PATTERNS
    CAUSATIVE_JUSSIVE_ROOT_PATTERNS = CAUSATIVE_JUSSIVE_ROOT_PATTERNS
    DERIVED_PERFECTIVE_ROOT_PATTERNS = DERIVED_PERFECTIVE_ROOT_PATTERNS
    VERBAL_RADICAL_PATTERNS = VERBAL_RADICAL_PATTERNS


    def is_contingent_jussive_syncretic(self, stem: str) -> bool:
        """
        Determines if a stem is morphotactically ambiguous between Contingent (Imperfective)
        and Jussive/Imperative when bound to subject prefixes.
        """
        if not stem:
            return False
            
        stem_orders = tuple(self.phonology.get_vocalic_orders(stem))
        
        # 1. Dynamic Orthographic Syncretism (Intersection of Canonical Semitic Matrices)
        syncretic_patterns = self.LESLAU_IMPERFECTIVE_PATTERNS & self.LESLAU_JUSSIVE_PATTERNS
        if stem_orders in syncretic_patterns:
            return True
            
        # 2. Known Type B palatals and defective verbs
        SYNCRETIC_IMP_JUSS_STEMS = {
            (0, 5),          # ሸኝ, ለይ, በጅ, ተው, ገጭ, ፈጅ, ፈጭ, ነጭ, ረቅ, ቸክ
            (3, 5),          # ዋኝ, ቻል, ሻል, ታሽ, ዳጥ, ጋጭ
            (3,),            # ሻ
        }
        if stem_orders in SYNCRETIC_IMP_JUSS_STEMS:
            return True
        if stem in ("መንን", "ሸኝ", "ለይ", "በጅ", "ተው", "ገጭ", "ፈጅ", "ፈጭ", "ነጭ", "ሻ"):
            return True
        if stem in getattr(self, "biradical_type_b_verbs", set()):
            return True
            
        # 3. Passive, Reciprocal & Bound Roots (ተ-, ተን-, ታ-)
        # Under subject concord (ይ-, ት-, እ-, እን-), passive, reciprocal, and bound roots are 100% syncretic.
        if (
            stem_orders in self.PASSIVE_JUSSIVE_ROOT_PATTERNS
            or stem_orders in self.BOUND_JUSSIVE_STEM_PATTERNS
        ):
            return True
                
        return False

    def is_perfective_jussive_syncretic(
        self,
        stem: str,
        prefixes: Tuple[str, ...] = (),
        suffixes: Tuple[str, ...] = ()
    ) -> bool:
        """
        Determines if a bare stem is ambiguous between a Simple Perfective
        and a 2MS Imperative command (e.g. ተጎዳ, ለካ, ጠጣ, አሰማራ).
        """
        if not stem:
            return False

        stem_orders = tuple(self.phonology.get_vocalic_orders(stem))

        # 1. Type B Biradicals, Hollow Weak & Statives (ለካ, ጠጣ, ቀባ, ቀማ, ላላ, ሳሳ, ሟሟ, ሞላ, ቦካ, ኮራ, ጎዳ)
        if stem in getattr(self, "biradical_type_b_verbs", set()) or stem_orders in {(3, 3), (7, 7)}:
            return True

        # 2. Derived Voice final-weak stems (passive/causative)
        # Checks both intact stems (ተመራ) and peeled forms (prefixes=('ተ',), stem='መራ')
        has_voice_pfx = (
            stem.startswith(("ተ", "ተን", "ታ", "አ", "አስ", "አን", "አስተ"))
            or any(p in ("ተ", "ተን", "ታ", "አ", "አስ", "አን", "አስተ") for p in prefixes)
        )
        has_weak_ending = (
            (bool(stem_orders) and stem_orders[-1] in (3, 7))
            or (bool(suffixes) and suffixes[-1] in ("ኡ", "ኢ") and bool(stem_orders) and stem_orders[-1] in (3, 0, 1))
        )
        if has_voice_pfx and len(stem) >= 2 and has_weak_ending:
            return True

        # 3. Strict overlap in orders
        if stem_orders in self.LESLAU_PERFECTIVE_PATTERNS and stem_orders in self.SHAPE_D_JUSSIVE_ORDERS:
            return True

        return False



    def classify_licensed_verbal_shape(
        self,
        prefixes: Tuple[str, ...],
        suffixes: Tuple[str, ...],
        stem: str = ""
    ) -> VerbalShape:
        """
        Evaluates Lock 1 (Affix Concord Matrix) and Lock 2 (Radical Stem Template)
        under Semitic morphotactics using dynamic morpheme inventories.
        """
        pfx_set = set(prefixes)
        sfx_set = set(suffixes)

        # Exclusive nominal morphology is unambiguous category evidence: an
        # exclusive nominal suffix (e.g. the plural ኦች) can never co-occur with
        # verbal inflection, so the verbal-shape locks must not be evaluated on
        # such a suffix set.  Without this, the ambiguous accusative -ን (which
        # is also an object clitic) pinned forms like ቤቶችን to Shape D and made
        # the nominal plural parse unreachable.
        if sfx_set & self.exclusive_nominal_suffixes and not (
            pfx_set & self.exclusive_verbal_prefixes
        ):
            return VerbalShape.UNKNOWN

        # Dynamic verbal affix checks
        has_auxiliary = any(s in self.suffix_verb_auxiliary or s in ("አል", "ዋል") for s in suffixes)
        has_coverb_suffix = bool(sfx_set & (getattr(self, "suffix_verb_coverb_exclusive", set()) | {"ኦ", "ቼ", "ተው", "ቶ", "ታ"}))
        has_applicative = bool(sfx_set & getattr(self, "suffix_applicative_clitics", set()))

        # A relative-imperfective particle can only introduce an imperfective
        # relative clause, so it settles both questions at once: it rules out
        # Shape E (የሚመለስ is a relative clause, not the verbal noun መለስ) and it
        # fixes the shape at C even when the bare stem matches another template
        # (የሚደብድብ is imperfective although ደብድብ matches the perfective one).
        has_rel_context = bool(pfx_set & self.rel_imperfective_markers)

        # 1. Shape E: Infinitive / Verbal Noun
        # Morphotactically strictly excludes subject concord, tense auxiliaries, converb vowels, and applicatives
        has_imp_pfx_context = bool(pfx_set & self.all_subject_prefixes)
        if not (
            has_imp_pfx_context
            or has_rel_context
            or has_auxiliary
            or has_coverb_suffix
            or has_applicative
        ):
            if any(p.startswith(("መ", "ማ")) for p in prefixes):
                return VerbalShape.SHAPE_E_INFINITIVE
            if self.is_valid_verbal_noun_stem(stem):
                return VerbalShape.SHAPE_E_INFINITIVE

        if has_rel_context:
            return VerbalShape.SHAPE_C_IMPERFECTIVE

        # 2. Shape B: Coverb Base
        shape_b_coverb = getattr(VerbalShape, "SHAPE_B_COVERB", getattr(VerbalShape, "SHAPE_B_COVERB", None))
        exclusive_coverb = getattr(self, "suffix_verb_coverb_exclusive", getattr(self, "suffix_verb_coverb_exclusive", set()))
        ambiguous_coverb = getattr(self, "suffix_verb_coverb_ambiguous", getattr(self, "suffix_verb_coverb_ambiguous", set()))

        # 2a. Coverb Stems (bare, with voice prefixes ተ-/አስ-, or with object clitics: e.g. ሰብሮ, ተ-ሰብሮ, አስ-መርምሮ, ደንቆ-ት, ተ-ሰምቶ, ሆኖ, ሄዶ)
        has_coverb_compatible_sfx = not suffixes or bool(sfx_set.issubset(
            self.suffix_verb_object_clitics | self.suffix_verb_auxiliary | {"ት", "ኝ", "ህ", "ሽ", "ን", "አት", "አቸው", "አችሁ", "ዎ", "ዎት", "አል", "ዋል"}
        ))
        has_no_subj_pfx = not (pfx_set & self.all_subject_prefixes)
        if has_no_subj_pfx and has_coverb_compatible_sfx and stem and len(stem) >= 2:
            stem_orders = tuple(self.phonology.get_vocalic_orders(stem))
            if stem_orders and (stem_orders[-1] == 6 or stem.endswith("ቶ")):
                if stem_orders in self.LESLAU_COVERB_PATTERNS or stem.endswith("ቶ") or self.is_valid_coverb_stem(stem):
                    return shape_b_coverb

        if not (pfx_set & self.all_subject_prefixes):
            # Tier 1: Authoritative Coverb Suffixes (Exclusive Morphemes with -t-/-ch- + 3MS "ኦ")
            if bool(sfx_set & (exclusive_coverb | {"ኦ"})):
                return shape_b_coverb

            # Tier 2: Polysemous Suffixes (ው, ህ, ሽ, ን, አችሁ, አ, ኤ, ዋ)
            # Strictly requires an ambiguous coverb suffix to be PRESENT in sfx_set
            if bool(sfx_set & ambiguous_coverb):
                # Always valid if auxiliary -al is present (Compound Coverb: ሰብረዋል, ፈልገዋል)
                if has_auxiliary:
                    return shape_b_coverb
                # Bare polysemous suffixes REQUIRE the stem to strictly match the non-finite coverb grade
                stem_orders = tuple(self.phonology.get_vocalic_orders(stem)) if stem else ()
                # 1PL coverb '-ን' on strong verbs requires a vowel link (-än, e.g. ሰብረን, መዝነን)
                # A bare Sadis cluster (0, 5) or (5, 5) with bare '-ን' is an imperative command (መዝን, ሽፍን), never a coverb
                if "ን" in sfx_set and stem_orders in ((0, 5), (5, 5)):
                    pass
                elif (
                    stem_orders in ((0, 5), (0, 5, 5), (3, 0, 5), (5, 5), (0, 0, 3, 5, 5))
                    or (
                        stem_orders
                        and stem_orders[-1] == 5
                        and self.is_valid_coverb_stem(stem)
                    )
                    or self.has_reduced_coverb_radical(stem)
                ):
                    # The list above enumerates five bare coverb grades; every
                    # member of it ends in a Sadis (5) stem-final, which is what
                    # the coverb 3PL ው hangs off (ተጯጩኸ-ው, ተፈቃቅረ-ው).  Rather
                    # than enumerate, the same grade test the peeler already
                    # ships (is_valid_coverb_stem, backed by the Leslau coverb
                    # patterns) decides the rest.
                    return shape_b_coverb

        # Relative prefixes
        rel_imp_pfx = self.rel_imperfective_prefixes | {"የሚ", "የምት", "የምን", "የምታ"}
        has_rel_imp_pfx = bool(pfx_set & rel_imp_pfx) or any(
            p.startswith(("እስከሚ", "በሚ", "ለሚ", "ከሚ", "እንደሚ", "ስለሚ")) for p in prefixes
        )

        # Imperfective / Jussive syncretic agreement prefixes
        syncretic_pfx = self.prefix_verb_iv | {"ኢ", "ያ", "አይ", "አት", "አን"}
        has_syncretic_pfx = bool(pfx_set & syncretic_pfx)
        has_subord_imp_pfx = any(
            p in self.prefix_verb_subord_imp
            or p.startswith(("ሲ", "ቢ", "ሊ", "እንዲ", "እንድ", "እስክ", "እስኪ", "እስከ", "በሚ", "ለሚ", "ከሚ", "እንደሚ", "ስለሚ"))
            for p in prefixes
        )
        has_imp_neg_pfx = any(
            p in self.imperfective_negative_prefixes
            or p in ("አይ", "አት", "አን", "የማይ", "እንዳይ", "ሳይ", "ስለማይ", "በማይ", "ከማይ", "ለማይ")
            for p in prefixes
        )

        # 3. Shape D: Jussive / Imperative Base (Lock 1 + Lock 2 Double-Lock)
        if "ል" in pfx_set and not has_auxiliary:
            return VerbalShape.SHAPE_D_JUSSIVE_IMPERATIVE

        stem_orders = tuple(self.phonology.get_vocalic_orders(stem)) if stem else ()

        # Object clitic indicator
        has_obj_sfx = bool(sfx_set & (self.suffix_verb_object_clitics | {"ው", "ኝ", "ህ", "ሽ", "ት", "አት", "ን", "አቸው", "አችሁ", "አቹ", "ዎ", "ዎት"}))

        # Bare 2nd person imperative command (with optional -ኢ/-ኡ agreement or object clitics)
        if not prefixes and (not suffixes or sfx_set.issubset({"ኢ", "ኡ"}) or has_obj_sfx):
            if stem in ("ና", "ሻ"):
                return VerbalShape.SHAPE_D_JUSSIVE_IMPERATIVE
            # A bare command without prefixes never begins with subject concord 'ይ-' or coalesced 'ያ-' (ይ+አ-)
            # 2-character words like ያሽ, ያክ, ያይ, ያጭ, ይል are contingent imperfectives
            if stem.startswith(("ይ", "ያ")) and len(stem) <= 3:
                pass
            # (0, 3) and (3, 3) bare stems are authoritatively 3MS base perfectives (ሰማ, ገባ, ወጣ, በላ, ላላ)
            # Type A commands for these verbs use (5, 3) (ስማ, ግባ, ውጣ, ብላ)
            elif (
                stem_orders
                and stem_orders not in self.TWO_RADICAL_BASE_PERFECTIVE_ORDERS
                and (stem_orders in self.SHAPE_D_JUSSIVE_ORDERS or self.is_valid_jussive_stem(stem))
            ):
                return VerbalShape.SHAPE_D_JUSSIVE_IMPERATIVE

        # Subject prefixes without auxiliary: check Lock 2
        syncretic_subject_pfx = bool(pfx_set & {"ይ", "ት", "እን"})
        if syncretic_subject_pfx and not has_auxiliary and stem_orders:
            # Check master unambiguous jussive patterns strictly
            if stem_orders in self.UNAMBIGUOUS_JUSSIVE_ORDERS:
                return VerbalShape.SHAPE_D_JUSSIVE_IMPERATIVE

        # 4. Shape C: Imperfective Base (Lock 1 Affix + Lock 2 Radical Template Double-Lock)
        has_imp_concord = has_syncretic_pfx or has_rel_imp_pfx or has_subord_imp_pfx or has_imp_neg_pfx
        if has_imp_concord:
            # Auxiliary -al/-wal with subject prefix is 100% authoritative Compound Imperfective
            if has_auxiliary:
                return VerbalShape.SHAPE_C_IMPERFECTIVE
            # Lock 2: Contingent stem MUST match an authentic imperfective template or verified stem
            if (
                stem_orders in self.LESLAU_IMPERFECTIVE_PATTERNS
                or self.is_valid_imperfective_stem(stem)
                or stem in self.verb_stems
            ):
                return VerbalShape.SHAPE_C_IMPERFECTIVE

        # 5. Shape A: Perfective Base
        has_exclusive_perf_sfx = bool(sfx_set & self.suffix_verb_exclusive_subject)
        has_relative_pfx = "የ" in prefixes or any(
            p in self.prefix_verb_exclusive_aspectual_perfective or p.startswith("የ")
            for p in prefixes
        )
        has_past_neg = any(p in ("አል", "አላ") for p in prefixes)
        has_voice_pfx = any(p in self.prefix_verb_voice for p in prefixes)
        has_subord_perf_pfx = any(p in ("ከ", "እንደ", "ስለ", "በ", "ለ") for p in prefixes)
        has_verb_u = ("ኡ" in sfx_set) and (has_relative_pfx or has_past_neg or has_voice_pfx or has_subord_perf_pfx or not prefixes)

        # 3MS Past stem with object clitic (ደነቀ-ው, ቸገረ-ው, ነሰረ-ው, ጨነቀ-ው, ራበ-ው, ጠማ-ው)
        is_perf_stem = (
            stem_orders in self.LESLAU_PERFECTIVE_PATTERNS
            or self.is_valid_perfective_template(stem)
            or stem in self.verb_stems
            or stem_orders in self.TWO_RADICAL_BASE_PERFECTIVE_ORDERS
        )
        has_perf_obj = has_obj_sfx and is_perf_stem and not (pfx_set & self.all_subject_prefixes)

        if has_exclusive_perf_sfx or has_relative_pfx or has_past_neg or has_verb_u or has_perf_obj:
            return VerbalShape.SHAPE_A_PERFECTIVE

        # Derived Voice & Subordinate Perfective Concord
        voice_pfx, root_remnant = self.peeler.extract_voice_and_root(prefixes, stem)
        if (voice_pfx or has_voice_pfx or has_subord_perf_pfx) and not (pfx_set & self.all_subject_prefixes):
            root_eval = root_remnant if voice_pfx else stem
            root_orders = tuple(self.phonology.get_vocalic_orders(root_eval)) if root_eval else ()

           # 1. Voice-Gated Template Verification (ONLY when NO auxiliary -አል is present)
            if not has_auxiliary:
                is_passive = voice_pfx in ("ተ", "ተን", "ታ") or "ተ" in prefixes
                is_causative = voice_pfx in ("አስ", "አስተ", "አ", "አን", "አም") or any(p in ("አስ", "አ", "አን", "አም") for p in prefixes)
                _, coda_o = self.phonology.get_char_info(root_eval[-1]) if root_eval else ("", -1)

                # Passive Voice: strong roots (coda 5) are commands (ተሰበር, ተመርመር).
                # Final-weak roots (coda 3/7, e.g. ተመራ, ተሰማ, ተበላ) are 3MS perfectives homophonous with commands;
                # they must default to Shape A so downstream homograph arbitration emits VB:PERFECTIVE:JUSSIVE.
                if is_passive and not suffixes:
                    if coda_o == 5:
                        if (
                            root_orders in self.PASSIVE_JUSSIVE_ROOT_PATTERNS
                            or root_orders in self.LESLAU_JUSSIVE_PATTERNS
                            or self.is_valid_jussive_stem(root_eval)
                        ):
                            return VerbalShape.SHAPE_D_JUSSIVE_IMPERATIVE


                # Causative Voice: Sadis coda (አድርግ, አዳምጥ, አመልክት, አስተምር) or (5,3)/(1,3) weak commands (አምጣ, አውጣ, አስሙላ)
                if is_causative and not suffixes:
                    is_weak_causative_cmd = root_orders in {(5, 3), (1, 3), (1, 5), (2, 5), (3, 5), (6, 5), (3, 3)}
                    if coda_o == 5 or is_weak_causative_cmd:
                        if (
                            root_orders in self.CAUSATIVE_JUSSIVE_ROOT_PATTERNS
                            or root_orders in self.LESLAU_JUSSIVE_PATTERNS
                            or self.is_valid_jussive_stem(root_eval)
                            or coda_o == 5
                        ):
                            return VerbalShape.SHAPE_D_JUSSIVE_IMPERATIVE

                # 2FS Palatal Imperative suffix -ኢ
                if "ኢ" in sfx_set:
                    return VerbalShape.SHAPE_D_JUSSIVE_IMPERATIVE

            # 2. Derived Past (Shape A) verification: ተመሰለ (0), አደረገ (0), አስመሰለ (0), ተመራ (3)
            return VerbalShape.SHAPE_A_PERFECTIVE

        # Bare 3MS perfective and coalesced contingent stems (no affixes)
        if not prefixes and not suffixes and stem:
            if stem in TWO_CHAR_CONTINGENTS or (len(stem) == 2 and stem.startswith(("ይ", "ያ"))):
                return VerbalShape.SHAPE_C_IMPERFECTIVE
            if stem_orders in self.LESLAU_PERFECTIVE_PATTERNS or self.is_valid_perfective_template(stem) or stem in self.verb_stems:
                return VerbalShape.SHAPE_A_PERFECTIVE

        return VerbalShape.UNKNOWN

    def is_valid_shape_template(self, stem: str, shape: VerbalShape) -> bool:
        """Validates whether a stem's radical vocalic orders conform to the specified VerbalShape."""
        if not stem or len(stem) < 2:
            return False
        if self.is_reduplicative_quadriradical(stem) or self.is_frequentative(stem):
            return True
        orders = tuple(self.phonology.get_vocalic_orders(stem))
        allowed_patterns = self.VERBAL_RADICAL_PATTERNS.get(shape, set())
        return orders in allowed_patterns

    def __init__(
        self,
        morphemes_json_path: Optional[str] = None,
        clusters_json_path: Optional[str] = None,
        two_char_words_path: Optional[str] = None,
        phonology: Optional[AmharicBoundaryPhonology] = None,
    ):
        self.phonology = phonology or AmharicBoundaryPhonology()
        self.valid_2char_stems: Set[str] = set()

        self.perfective_subject_urs: List[Tuple[str, str]] = [
            ("ኩ", "1S"),
            ("ኹ", "1S"),
            ("ሁ", "1S"),
            ("ክ", "2MS"),
            ("ኽ", "2MS"),
            ("ሽ", "2FS"),
            ("ኧች", "3FS"),
            ("ን", "1PL"),
            ("ነን", "1PL"),
            ("አችሁ", "2P"),
            ("አቹ", "2P"),
            ("ኡ", "3P"),
        ]
        self.possessive_urs: List[Tuple[str, str]] = [
            ("ኤ", "1S"),
            ("ዬ", "1S"),
            ("እህ", "2MS"),
            ("እሽ", "2FS"),
            ("አችን", "1PL"),
            ("አችሁ", "2PL"),
            ("አቹ", "2PL"),
            ("ኡ", "3MS"),
            ("አቸው", "3PL"),
            ("ዋ", "3FS"),
            ("ዎ", "2POL"),
            ("ዎት", "2POL"),
        ]
        self.instrument_noun_urs: List[str] = ["ኢያ", "ኢያው"]
        self.single_letter_proclitics: Set[str] = set()

        self._CACHE_MAX: int = 16384
        # Instance-bounded LRU caches (prevent memory leak and cache freezing)
        self._nom_cohort_cache = SimpleLRUCache(self._CACHE_MAX)
        self._verb_cohort_cache = SimpleLRUCache(self._CACHE_MAX)
        self._verbal_lemma_cache = SimpleLRUCache(self._CACHE_MAX)
        self._perf_template_cache = SimpleLRUCache(self._CACHE_MAX)
        self._radical_template_cache = SimpleLRUCache(self._CACHE_MAX)
        self._stem_affix_cache = SimpleLRUCache(self._CACHE_MAX)
        # Candidate generation shares this cache so config reloads clear it in-place.
        self.lemma_candidate_generator = LemmaCandidateGenerator(
            morphology=self, phonology=self.phonology, cache=self._verbal_lemma_cache
        )
        self.load_2char_words(two_char_words_path)

        if morphemes_json_path is None:
            morphemes_json_path = str(MORPHEMES_PATH)
        if clusters_json_path is None:
            clusters_json_path = str(MORPHEMES_CLUSTERS_PATH)

        self.load_morphemes_config(morphemes_json_path)
        self.load_clusters_config(clusters_json_path)

        # Corpus-derived contingent/jussive reading policy (may be absent).
        self.contingent_jussive_policy: Dict[str, str] = {}
        self.contingent_jussive_policy_meta: Dict[str, Any] = {}
        self.load_contingent_jussive_policy()

        # Dedicated Affix Peeler Engine
        self.peeler = AffixPeeler(morphology=self, phonology=self.phonology)

    def load_contingent_jussive_policy(self, json_path: Optional[str] = None) -> None:
        """Load the derived homograph reading tables.

        Amharic orthography does not mark gemination, so whether a lexeme's
        contingent and jussive coincide (Type B/C) or diverge (Type A) is a
        lexical property; likewise a bare 2MS imperative is written identically
        to the 3MS perfective.  A stem's vocalic-order shape is nonetheless
        strongly predictive of which case applies, and these tables record the
        majority reading per shape.  They are generated by
        ``tools/build_contingent_jussive_policy.py``, which also reports held-out
        accuracy over unseen lexemes; a missing file simply disables the policy
        and the engine falls back to its hand-written rules.
        """
        self.contingent_jussive_policy = self._load_reading_policy(
            json_path or str(CONTINGENT_JUSSIVE_POLICY_PATH)
        )
        self.perfective_imperative_policy = self._load_reading_policy(
            str(PERFECTIVE_IMPERATIVE_POLICY_PATH)
        )
        self.experiencer_verbs: Dict[str, Dict[str, Any]] = {}
        self.load_experiencer_verbs()

    def load_experiencer_verbs(self, json_path: Optional[str] = None) -> None:
        """Load the curated experiencer (psych) verb list.

        These verbs do not behave like regular verbs: the active normally carries
        an object clitic (``ደነቀው`` "it amazed him"), the ``ተ-`` passive carries
        the same experiencer meaning without one (``ተደነቀ`` "he was amazed"), and
        the imperative strongly prefers the ``ተ-`` form (``ተቸገር``, rarely
        ``ቸግረው``).  So for these lexemes an *active* form bearing an object
        clitic is a perfective, not an imperative -- a fact no surface-shape rule
        can recover, since in the corpus 313 object-suffixed forms are
        jussive/imperative against 7 that are perfective, and every shape bucket
        is majority-jussive.  A missing file simply disables the override.
        """
        path = json_path or str(EXPERIENCER_VERBS_PATH)
        self.experiencer_verbs = {}
        if not os.path.exists(path):
            logger.warning("Experiencer verb list not found at %s; override disabled.", path)
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
            logger.warning("Could not read experiencer verb list %s: %s", path, exc)
            return
        for key, info in (payload.get("verbs") or {}).items():
            if isinstance(info, dict):
                self.experiencer_verbs[str(key)] = info
        logger.info("Loaded experiencer verb list: %d roots.", len(self.experiencer_verbs))

    def is_experiencer_root(self, stem: str) -> bool:
        """True when ``stem``'s consonantal skeleton is a known experiencer verb.

        Keyed on the skeleton so the check is indifferent to which template the
        inflected form uses: ``ቸገር``, ``ቸገረ`` and ``ቸግር`` all resolve to
        ``ቸገረ``.
        """
        if not self.experiencer_verbs or not stem:
            return False
        return consonantal_skeleton(stem) in self.experiencer_verbs

    def _load_reading_policy(self, path: str) -> Dict[str, str]:
        """Parse one derived reading table; return ``{}`` when unusable."""
        if not os.path.exists(path):
            logger.warning("Reading policy not found at %s; using rule fallbacks.", path)
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
            logger.warning("Could not read reading policy %s: %s", path, exc)
            return {}

        if path == str(CONTINGENT_JUSSIVE_POLICY_PATH):
            self.contingent_jussive_policy_meta = dict(payload.get("_meta", {}))

        valid = ("BOTH", "VB:IMPERFECTIVE", "VB:PERFECTIVE", "VB:JUSSIVE_IMPERATIVE")
        table: Dict[str, str] = {}
        for key, reading in (payload.get("policy") or {}).items():
            if reading not in valid or not self._is_valid_shape_key(str(key)):
                continue
            table[str(key)] = reading
        logger.info("Loaded reading policy %s: %d shapes.", os.path.basename(path), len(table))
        return table

    @classmethod
    def _is_valid_shape_key(cls, key: str) -> bool:
        """True for ``"0,0,3"``, ``"0,0,3|ተ"`` and ``"0,0,3|∅"``; else False."""
        orders_part, separator, voice_part = key.partition("|")
        parts = orders_part.split(",")
        # Every comma-separated slot must be a real integer; "0,,3" is malformed.
        if not all(part.lstrip("-").isdigit() and part.lstrip("-") for part in parts):
            return False
        if not separator:
            # A plain pooled key: partition() found no "|", so nothing follows.
            return True
        return voice_part in VOICE_PREFIXES or voice_part == cls.NO_VOICE_MARKER

    #: Marks "carries no voice prefix" in a voice-specific policy key.  Must stay
    #: in step with ``NO_VOICE`` in tools/build_contingent_jussive_policy.py.
    NO_VOICE_MARKER = "∅"

    def _reading_shape_key(self, word: str, voice_aware: bool = False) -> str:
        """Pooled policy key for ``word``: the vocalic orders of its stem.

        Mirrors ``shape_of()`` in ``tools/build_contingent_jussive_policy.py`` so
        the runtime and the generated table always agree.
        """
        orders = self.phonology.get_vocalic_orders(word)
        return ",".join(str(o) for o in orders) if orders else ""

    def _reading_voice_key(self, word: str) -> str:
        """Voice-specific policy key ``<orders>|<voice>``, or ``""``.

        ``<voice>`` is the leading ተ-/ተን-/ታ- (passive-reflexive) or
        አ-/አስ-/አን-/አስተ- (causative) prefix, or :data:`NO_VOICE_MARKER` when the
        word carries none.  This is what separates ``ተመራ`` -- a ተ- passive whose
        3MS perfective and 2MS imperative share a spelling -- from ``ዘረጋ``, the
        matching active whose imperative is ``ዘርጋ``; both have orders (0,0,3).
        """
        base = self._reading_shape_key(word)
        if not base:
            return ""
        prefix, _residue = split_voice_prefix(word)
        return f"{base}|{prefix or self.NO_VOICE_MARKER}"

    def contingent_jussive_reading(self, stem: str) -> Optional[str]:
        """Majority reading for ``stem``'s shape, or ``None`` when uncovered."""
        if not self.contingent_jussive_policy or not stem:
            return None
        # Pooled shapes only: the subject concord hides the lexeme's voice, and
        # splitting the shapes just fragments the evidence (held-out 84.51% ->
        # 84.25%).
        return self.contingent_jussive_policy.get(self._reading_shape_key(stem))

    def perfective_imperative_reading(self, surface: str) -> Optional[str]:
        """Majority reading for a bare finite surface form's shape.

        Same derivation and same held-out validation as
        :meth:`contingent_jussive_reading`, but keyed on the whole surface form
        (a bare 2MS imperative and a 3MS perfective are written identically, and
        only ~3.6% of paradigms actually collapse the two).
        """
        if not self.perfective_imperative_policy or not surface:
            return None
        # Voice-specific override first, pooled shape as the fallback.  The
        # perfective carries ተ-/አ- overtly, and that is exactly the signal that
        # decides whether the 2MS imperative shares the spelling.
        return self.perfective_imperative_policy.get(
            self._reading_voice_key(surface),
            self.perfective_imperative_policy.get(self._reading_shape_key(surface)),
        )

    @property
    def rel_imperfective_markers(self) -> Set[str]:
        """Relative-imperfective markers: the particles and their composites.

        A relative clause is perfective with the bare relative pronoun (``የሸና``)
        and imperfective with the particle (``የሚሰራ``, ``የምትሰራ``).  The
        composites are built from the shipped tables -- the pronoun, the
        particle allomorphs and the agreement markers -- so the shape lock does
        not carry a literal list of them.
        """
        particles: Set[str] = set(
            getattr(self, "prefix_rel_imperfective_particle", ()) or ()
        )
        particles.update(
            getattr(self, "prefix_rel_imperfective_particle_allomorphs", ()) or ()
        )
        pronouns: Set[str] = set(getattr(self, "prefix_relative_pronoun", ()) or ())
        agreements: Set[str] = set(
            getattr(self, "prefix_rel_imperfective_agreement", ()) or ()
        )

        markers: Set[str] = set(particles)
        for particle in particles:
            markers.add(particle)
            for agreement in agreements:
                markers.add(particle + agreement)
        for pronoun in pronouns:
            for particle in particles:
                markers.add(pronoun + particle)
                for agreement in agreements:
                    markers.add(pronoun + particle + agreement)
        return markers

    @property
    def suffix_verb_coverb_only(self) -> Set[str]:
        """Coverb vowels that cannot play any other suffix role.

        ``suffix_verb_coverb`` holds morphemes that are coverb vowels in one
        reading and object/possessive/definiteness clitics in another (ን, ው,
        ህ, ሽ, ዋ, ኤ).  Those do not by themselves decide between "coverb +
        auxiliary" and "imperfective + auxiliary", so only the morphemes that
        are coverb-exclusive survive: the -t-/-ch- series plus the 3MS ኦ/አ
        vowels.  Derived from the inventories, never enumerated here.
        """
        other_roles: Set[str] = set()
        for name in ("suffix_accusative", "suffix_verb_object_clitics", "suffix_applicative_clitics"):
            other_roles.update(getattr(self, name, ()) or ())
        for name in dir(self):
            if not name.startswith("suffix_nominal"):
                continue
            value = getattr(self, name, None)
            if isinstance(value, (set, frozenset, list, tuple)):
                other_roles.update(value)
        return set(self.suffix_verb_coverb) - other_roles

    @property
    def lemmatizer(self) -> LemmaCandidateGenerator:
        """Backward-compatible alias for :attr:`lemma_candidate_generator`."""
        return self.lemma_candidate_generator

    @lemmatizer.setter
    def lemmatizer(self, value: LemmaCandidateGenerator) -> None:
        self.lemma_candidate_generator = value

    @property
    def lemma_reconstructor(self) -> LemmaCandidateGenerator:
        """Backward-compatible alias for the former reconstruction component."""
        return self.lemma_candidate_generator

    @lemma_reconstructor.setter
    def lemma_reconstructor(self, value: LemmaCandidateGenerator) -> None:
        self.lemma_candidate_generator = value

    def load_2char_words(self, path: Optional[str] = None):
        """Loads and binds 2-character stems directly to the authoritative TSV registry."""
        if path:
            load_biradical_registry(os.path.dirname(path) if os.path.isfile(path) else path)
        elif not ALL_LICENSED_2CHAR_TOKENS:
            load_biradical_registry()
        self.valid_2char_stems = ALL_LICENSED_2CHAR_TOKENS

    def load_morphemes_config(self, json_path: str):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Critical: Morphemes JSON not found at {json_path}.")
        with open(json_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

        # Pydantic Schema Validation (validates types, reports exact structural errors)
        cfg: MorphemesConfig = (
            MorphemesConfig.model_validate(raw_cfg)
            if hasattr(MorphemesConfig, "model_validate")
            else MorphemesConfig.parse_obj(raw_cfg)  # type: ignore
        )

        # 1. Nominal Suffixes & Legacy Aliases
        self.suffix_nominal_plural_exclusive = set(cfg.suffix_nominal_plural_exclusive)
        self.suffix_nominal_plural_ambiguous = set(cfg.suffix_nominal_plural_ambiguous)
        self.suffix_nominal_plural = self.suffix_nominal_plural_exclusive | self.suffix_nominal_plural_ambiguous
        self.suffix_nominal_exclusive_plural = self.suffix_nominal_plural_exclusive

        self.suffix_nominal_definiteness_exclusive = set(cfg.suffix_nominal_definiteness_exclusive)
        self.suffix_nominal_definiteness_ambiguous = set(cfg.suffix_nominal_definiteness_ambiguous)
        self.suffix_nominal_definiteness = self.suffix_nominal_definiteness_exclusive | self.suffix_nominal_definiteness_ambiguous
        self.suffix_nominal_exclusive_definiteness = self.suffix_nominal_definiteness_exclusive

        self.suffix_nominal_possessive_exclusive = set(cfg.suffix_nominal_possessive_exclusive)
        self.suffix_nominal_possessive_ambiguous = set(cfg.suffix_nominal_possessive_ambiguous)
        self.suffix_nominal_possessive = self.suffix_nominal_possessive_exclusive | self.suffix_nominal_possessive_ambiguous
        self.suffix_nominal_exclusive_possessive = self.suffix_nominal_possessive_exclusive

        self.suffix_nominal_abstract_exclusive = set(cfg.suffix_nominal_abstract_exclusive)
        self.suffix_nominal_abstract_ambiguous = set(cfg.suffix_nominal_abstract_ambiguous)
        self.suffix_nominal_abstract = self.suffix_nominal_abstract_exclusive | self.suffix_nominal_abstract_ambiguous
        self.suffix_nominal_exclusive_abstract = self.suffix_nominal_abstract_exclusive

        self.suffix_nominal_derivational_exclusive = set(cfg.suffix_nominal_derivational_exclusive)
        self.suffix_nominal_derivational_ambiguous = set(cfg.suffix_nominal_derivational_ambiguous)
        self.suffix_nominal_derivational = self.suffix_nominal_derivational_exclusive | self.suffix_nominal_derivational_ambiguous
        self.suffix_nominal_exclusive_derivational = self.suffix_nominal_derivational_exclusive

        self.suffix_nominal_instrumental_exclusive = set(cfg.suffix_nominal_instrumental_exclusive)
        self.suffix_nominal_instrumental_ambiguous = set(cfg.suffix_nominal_instrumental_ambiguous)
        self.suffix_nominal_instrumental = self.suffix_nominal_instrumental_exclusive | self.suffix_nominal_instrumental_ambiguous
        self.suffix_nominal_exclusive_instrumental = self.suffix_nominal_instrumental_exclusive

        self.suffix_accusative = set(cfg.suffix_nominal_accusative_ambiguous or ["ን"])

        # 2. Verbal Suffixes & Legacy Aliases
        self.suffix_verb_subject_exclusive = set(cfg.suffix_verb_subject_exclusive)
        self.suffix_verb_subject_ambiguous = set(cfg.suffix_verb_subject_ambiguous)
        self.suffix_verb_pv = self.suffix_verb_subject_exclusive | self.suffix_verb_subject_ambiguous
        self.suffix_verb_exclusive_subject = self.suffix_verb_subject_exclusive

        self.suffix_verb_object_exclusive = set(cfg.suffix_verb_object_exclusive)
        self.suffix_verb_object_ambiguous = set(cfg.suffix_verb_object_ambiguous)
        self.suffix_verb_object_clitics = self.suffix_verb_object_exclusive | self.suffix_verb_object_ambiguous
        self.suffix_verb_exclusive_object = self.suffix_verb_object_exclusive

        self.suffix_verb_applicative_exclusive = set(cfg.suffix_verb_applicative_exclusive)
        self.suffix_verb_applicative_ambiguous = set(cfg.suffix_verb_applicative_ambiguous)
        self.suffix_applicative_clitics = self.suffix_verb_applicative_exclusive | self.suffix_verb_applicative_ambiguous
        self.suffix_verb_exclusive_applicative = self.suffix_verb_applicative_exclusive

        self.suffix_verb_coverb_exclusive = set(cfg.suffix_verb_coverb_exclusive)
        self.suffix_verb_coverb_ambiguous = set(cfg.suffix_verb_coverb_ambiguous)
        self.suffix_verb_coverb = self.suffix_verb_coverb_exclusive | self.suffix_verb_coverb_ambiguous
        self.suffix_verb_exclusive_coverb = self.suffix_verb_coverb_exclusive
        self.suffix_verb_coverb_weak = set(cfg.suffix_verb_coverb_weak)

        self.suffix_verb_auxiliary_exclusive = set(cfg.suffix_verb_auxiliary_exclusive)
        self.suffix_verb_auxiliary_ambiguous = set(cfg.suffix_verb_auxiliary_ambiguous)
        self.suffix_verb_auxiliary = self.suffix_verb_auxiliary_exclusive | self.suffix_verb_auxiliary_ambiguous
        self.suffix_verb_exclusive_auxiliary = self.suffix_verb_auxiliary_exclusive

        self.suffix_verb_negative_finiteness_exclusive = set(cfg.suffix_verb_negative_finiteness_exclusive)
        self.suffix_verb_negative_finiteness_ambiguous = set(cfg.suffix_verb_negative_finiteness_ambiguous)
        self.suffix_verb_negative_finiteness = self.suffix_verb_negative_finiteness_exclusive | self.suffix_verb_negative_finiteness_ambiguous
        self.suffix_verb_exclusive_negative_finiteness = self.suffix_verb_negative_finiteness_exclusive

        # 3. Nominal Prefixes
        self.prefix_nominal_associatives_exclusive = set(cfg.prefix_nominal_associatives_exclusive)
        self.prefix_nominal_associatives_ambiguous = set(cfg.prefix_nominal_associatives_ambiguous)
        self.prefix_nominal_associatives = self.prefix_nominal_associatives_exclusive | self.prefix_nominal_associatives_ambiguous
        self.prefix_nominal_exclusive_associatives = self.prefix_nominal_associatives_exclusive

        self.prefix_nominal_compound_exclusive = set(cfg.prefix_nominal_compound_exclusive)
        self.prefix_nominal_compound_ambiguous = set(cfg.prefix_nominal_compound_ambiguous)
        self.prefix_nominal_compound = self.prefix_nominal_compound_exclusive | self.prefix_nominal_compound_ambiguous
        self.prefix_nominal_exclusive_compound = self.prefix_nominal_compound_exclusive

        self.prefix_nominal_prepositional_exclusive = set(cfg.prefix_nominal_prepositional_exclusive)
        self.prefix_nominal_prepositional_ambiguous = set(cfg.prefix_nominal_prepositional_ambiguous)
        self.prefix_nominal_prepositional = self.prefix_nominal_prepositional_exclusive | self.prefix_nominal_prepositional_ambiguous
        self.prefix_nominal_exclusive_prepositional = self.prefix_nominal_prepositional_exclusive

        self.prefix_nominal_locative = set(cfg.prefix_nominal_locative)
        self.prefix_nominal_privative = set(cfg.prefix_nominal_privative)
        self.prefix_nominal = (
            self.prefix_nominal_associatives
            | self.prefix_nominal_compound
            | self.prefix_nominal_prepositional
            | self.prefix_nominal_locative
            | self.prefix_nominal_privative
        )

        # 4. Verbal Prefixes & Declarative Voice Hierarchy
        self.voice_hierarchy: Dict[str, List[str]] = dict(cfg.voice_morphology) if cfg.voice_morphology else {
            "direct_causative": ["አ"],
            "indirect_causative": ["አስ"],
            "non_active": ["ተ"],
        }
        self.direct_causative_prefixes: Tuple[str, ...] = tuple(self.voice_hierarchy.get("direct_causative", ["አ"]))
        self.indirect_causative_prefixes: Tuple[str, ...] = tuple(self.voice_hierarchy.get("indirect_causative", ["አስ"]))
        self.non_active_prefixes: Tuple[str, ...] = tuple(self.voice_hierarchy.get("non_active", ["ተ"]))
        self.primary_voice_prefixes: Tuple[str, ...] = tuple(sorted(
            set(self.direct_causative_prefixes + self.indirect_causative_prefixes + self.non_active_prefixes),
            key=len,
            reverse=True,
        ))

        self.prefix_verb_voice_exclusive = set(cfg.prefix_verb_voice_exclusive)
        self.prefix_verb_voice_ambiguous = set(cfg.prefix_verb_voice_ambiguous)
        # The declarative voice hierarchy also contains non-active ተ-. Keep it
        # in the verbal prefix inventory so literal decomposition can expose
        # passive forms instead of treating the initial ተ as part of a stem.
        self.prefix_verb_voice = (
            self.prefix_verb_voice_exclusive
            | self.prefix_verb_voice_ambiguous
            | set(self.non_active_prefixes)
        )
        self.prefix_verb_exclusive_voice = self.prefix_verb_voice_exclusive

        self.prefix_verb_exclusive_aspectual_perfective = set(cfg.prefix_verb_aspectual_perfective_exclusive)
        self.prefix_verb_aspectual_perfective_exclusive = self.prefix_verb_exclusive_aspectual_perfective

        self.prefix_verb_negative_exclusive = set(cfg.prefix_verb_negative_exclusive)
        self.prefix_verb_negative_ambiguous = set(cfg.prefix_verb_negative_ambiguous)
        self.prefix_verb_atomic_negative = self.prefix_verb_negative_exclusive | self.prefix_verb_negative_ambiguous
        self.prefix_verb_exclusive_negative = self.prefix_verb_negative_exclusive
        self.prefix_verb_primary_neg = self.prefix_verb_negative_ambiguous
        self.prefix_verb_subord_neg = self.prefix_verb_negative_exclusive

        self.prefix_verb_subordinate_exclusive = set(cfg.prefix_verb_subordinate_exclusive)
        self.prefix_verb_subordinate_ambiguous = set(cfg.prefix_verb_subordinate_ambiguous) | {"እስክ", "እስኪ"}
        self.prefix_verb_subord_imp = self.prefix_verb_subordinate_exclusive | self.prefix_verb_subordinate_ambiguous
        self.prefix_verb_exclusive_subordinate = self.prefix_verb_subordinate_exclusive
        self.prefix_verb_subordinate_imperfective = self.prefix_verb_subord_imp

        self.prefix_verb_jussive = set(["ል", "እን", "ይ", "ት"])
        self.prefix_verb_modal = (
            self.prefix_verb_subordinate_ambiguous
            | self.prefix_verb_subord_imp
            | self.prefix_verb_jussive
        )

        self.prefix_verb_progressive_exclusive = set(cfg.prefix_verb_progressive_exclusive)
        self.prefix_verb_progressive_ambiguous = set(cfg.prefix_verb_progressive_ambiguous)
        self.prefix_verb_progressive = self.prefix_verb_progressive_exclusive | self.prefix_verb_progressive_ambiguous
        self.prefix_verb_continuous = self.prefix_verb_progressive

        self.prefix_verb_iv = set(cfg.prefix_verb_aspectual_imperfective_ambiguous)
        self.prefix_verb_pv = self.prefix_verb_exclusive_aspectual_perfective
        self.prefix_infinitive_coalesced = set(cfg.prefix_infinitive_coalesced)

        # 5. Universal Clitics & Prepositions
        self.prefix_universal_exclusive = set(cfg.prefix_universal_prepositional_exclusive)
        self.prefix_universal_ambiguous = set(cfg.prefix_universal_prepositional_ambiguous)
        self.prefix_universal = self.prefix_universal_exclusive | self.prefix_universal_ambiguous

        self.suffix_universal_clitics_exclusive = set(cfg.suffix_universal_clitics_exclusive)
        self.suffix_universal_clitics_ambiguous = set(cfg.suffix_universal_clitics_ambiguous)
        self.suffix_universal_clitics = self.suffix_universal_clitics_exclusive | self.suffix_universal_clitics_ambiguous
        self.suffix_pragmatic_clitics = self.suffix_universal_clitics

        self.prefix_relative_pronoun = set(cfg.prefix_relative_pronoun)
        self.prefix_rel_imperfective_particle = set(cfg.prefix_rel_imperfective_particle)
        self.prefix_rel_imperfective_particle_allomorphs = set(cfg.prefix_rel_imperfective_particle_allomorphs)
        self.prefix_rel_imperfective_agreement = set(cfg.prefix_rel_imperfective_agreement)
        self.suffix_verb_imperfective_relative_agreement = self.prefix_rel_imperfective_agreement
        self.possessive_1pl_endings = tuple(cfg.possessive_1pl_endings)

        # 6. Whitelists & Invariants
        self.pronoun_stems = set(cfg.pronoun_stems)
        self.verb_stems = set(cfg.verb_stems)
        self.geez_plurals = set(cfg.geez_plurals)
        self.confirmed_sparse_nouns = set(cfg.confirmed_sparse_nouns)
        self.confirmed_sparse_verbs = set(getattr(cfg, "confirmed_sparse_verbs", []))
        self.irregular_verbs_say = set(cfg.irregular_verbs_say)
        self.biradical_type_b_verbs = set(getattr(cfg, "biradical_type_b_verbs", raw_cfg.get("Biradical_Type_B_verbs", []))) | {"ቀማ", "ቀባ", "ለማ"}
        self.biradical_type_a_medial_a_verbs = set(getattr(cfg, "biradical_type_a_medial_a_verbs", []))
        self.verbs_ending_in_gna = set(cfg.verbs_ending_in_gna)

        self.weak_coverb_lemma_endings = tuple(cfg.weak_coverb_lemma_endings)
        self.imperfective_negative_prefixes = set(cfg.imperfective_negative_prefixes)
        self.verbal_prefixes_requiring_attestation = set(cfg.verbal_prefixes_requiring_attestation)
        self.subject_e_exclusions = tuple(cfg.subject_e_exclusions)
        self.derived_voice_prefixes = tuple(cfg.derived_voice_prefixes)
        self.unambiguous_assoc_prefixes = tuple(cfg.unambiguous_assoc_prefixes)

        self.auxiliary_coda_shifts = dict(cfg.auxiliary_coda_shifts)
        self.compound_auxiliary_suffixes = dict(cfg.compound_auxiliary_suffixes)
        self.compound_plural_possessives = dict(cfg.compound_plural_possessives)
        self.bipartite_applicative_mappings = dict(cfg.bipartite_applicative_mappings)

        self.auxiliary_verbs_standalone = set(cfg.auxiliary_verbs_standalone)
        self.closed_class_conjunctions = set(cfg.closed_class_conjunctions)
        self.closed_class_adpositions = set(cfg.closed_class_adpositions)
        self.closed_class_adverbs = set(cfg.closed_class_adverbs)
        self.closed_class_question_words = set(cfg.closed_class_question_words)
        self.closed_class_ideophones = set(cfg.closed_class_ideophones)
        self.closed_class_quantifiers = set(cfg.closed_class_quantifiers)
        self.ambiguous_verbal_nouns = set(getattr(cfg, "ambiguous_verbal_nouns", ["መስቀል", "መብራት"]))
        self.inherent_terminal_u_stems = set(cfg.inherent_terminal_u_stems)
        self.defective_verbal_forms = dict(cfg.defective_verbal_forms)
        self.suppletive_stem_templates = dict(getattr(cfg, "suppletive_stem_templates", raw_cfg.get("suppletive_stem_templates", {})))

        # Coalescence & Shield Data
        self.type_c_shield_words = set(cfg.type_c_shield_words)
        self.proclitic_root_shield = set(getattr(cfg, "proclitic_root_shield", []))
        self.privative_vn_prefix_map = {
            surf: tuple(components)
            for surf, components in cfg.privative_vn_prefix_map.items()
        }
        self.multi_rabe_prefixes = tuple(
            (surf, tuple(meta["components"]), meta["rule"])
            for surf, meta in cfg.multi_rabe_prefixes.items()
            if isinstance(meta, dict) and "components" in meta and "rule" in meta
        )
        self.syncretic_imp_juss_orders = {
            tuple(o) for o in cfg.syncretic_imp_juss_orders
        }
        self.syncretic_imp_juss_lexemes = set(cfg.syncretic_imp_juss_lexemes)

        self._nom_cohort_cache.clear()
        self._verb_cohort_cache.clear()
        self._verbal_lemma_cache.clear()
        self._perf_template_cache.clear()
        self._radical_template_cache.clear()
        self._stem_affix_cache.clear()
        self._rebuild_derived_inventories_and_indices()

    def load_clusters_config(self, json_path: str):
        if not os.path.exists(json_path):
            self.macro_prefix_nominal = set()
            self.macro_prefix_vn = set()
            self.macro_prefix_pv = set()
            self.macro_prefix_iv = set()
            self.macro_suffix_cn = set()
            self.macro_suffix_mn = set()
            self.macro_suffix_benefactive_aux = set()
            self.macro_suffix_malfactive_aux = set()
            self.macro_suffix_coverb_aux = set()
            return

        with open(json_path, "r", encoding="utf-8") as f:
            raw_clusters = json.load(f)

        self.macro_prefix_nominal = set(raw_clusters.get("macro_prefix_nominal", []))
        self.macro_prefix_vn = set(raw_clusters.get("macro_prefix_vn", []))
        self.macro_prefix_pv = set(raw_clusters.get("macro_prefix_pv", []))
        self.macro_prefix_iv = set(raw_clusters.get("macro_prefix_iv", []))
        self.macro_suffix_cn = set(raw_clusters.get("macro_suffix_cn", []))
        self.macro_suffix_mn = set(raw_clusters.get("macro_suffix_mn", []))
        self.macro_suffix_benefactive_aux = set(
            raw_clusters.get("macro_suffix_benefactive_aux", [])
        )
        self.macro_suffix_malfactive_aux = set(
            raw_clusters.get("macro_suffix_malfactive_aux", [])
        )
        self.macro_suffix_coverb_aux = set(
            raw_clusters.get("macro_suffix_coverb_aux", [])
        )

    def _rebuild_derived_inventories_and_indices(self):
        self.suffix_nominal = (
            self.suffix_nominal_derivational
            | self.suffix_nominal_abstract
            | self.suffix_nominal_plural
            | self.suffix_nominal_definiteness
            | self.suffix_nominal_possessive
            | self.suffix_nominal_instrumental
        )
        self.all_nominal_suffixes = self.suffix_nominal | self.suffix_accusative
        self.all_nominal_prefixes = set(self.prefix_nominal)
        self.all_nominal_exclusive_suffixes = (
            self.suffix_nominal_exclusive_plural
            | self.suffix_nominal_exclusive_abstract
            | self.suffix_nominal_exclusive_derivational
            | self.suffix_nominal_exclusive_instrumental
        )

        self.rel_imperfective_prefixes = (
            self.prefix_rel_imperfective_particle
            | self.prefix_rel_imperfective_particle_allomorphs
        )
        self.all_verbal_prefixes = (
            self.prefix_verb_iv
            | self.prefix_verb_pv
            | self.prefix_verb_voice
            | self.prefix_verb_modal
            | self.prefix_verb_continuous
            | self.prefix_verb_atomic_negative
            | self.rel_imperfective_prefixes
        )
        self.prefix_verb = self.all_verbal_prefixes

        self.all_subject_prefixes = (
            self.prefix_verb_iv
            | self.prefix_verb_jussive
            | self.prefix_rel_imperfective_particle
            | self.prefix_rel_imperfective_agreement
            | {
                p
                for p in self.prefix_verb_atomic_negative
                if p.endswith("ት") or p.endswith("ይ") or p.endswith("ን")
            }
        )
        self.all_verbal_suffixes = (
            self.suffix_verb_pv
            | self.suffix_verb_auxiliary
            | self.suffix_verb_negative_finiteness
            | self.suffix_verb_imperfective_relative_agreement
            | self.suffix_verb_coverb
            | self.suffix_verb_object_clitics
            | self.suffix_applicative_clitics
        )
        self.suffix_verb = self.all_verbal_suffixes

        self.exclusive_nominal_suffixes = (
            self.suffix_nominal_exclusive_plural
            | self.suffix_nominal_exclusive_abstract
            | self.suffix_nominal_exclusive_derivational
            | self.suffix_nominal_exclusive_instrumental
            | self.suffix_nominal_exclusive_definiteness
            | self.suffix_nominal_exclusive_possessive
        )

        self.exclusive_verbal_suffixes = (
            self.suffix_verb_exclusive_subject
            | self.suffix_verb_object_exclusive
            | self.suffix_verb_applicative_exclusive
            | self.suffix_verb_exclusive_coverb
            | self.suffix_verb_exclusive_auxiliary
            | self.suffix_verb_exclusive_negative_finiteness
        )

        self.exclusive_nominal_prefixes = (
            self.prefix_nominal_exclusive_associatives
            | self.prefix_nominal_exclusive_compound
            | self.prefix_nominal_exclusive_prepositional
        )

        self.exclusive_verbal_prefixes = (
            self.prefix_verb_exclusive_voice
            | self.prefix_verb_exclusive_aspectual_perfective
            | self.prefix_verb_exclusive_negative
            | self.prefix_verb_exclusive_subordinate
            | self.prefix_rel_imperfective_particle
        )

        self.rel_imp_licensors = self.prefix_universal | self.prefix_relative_pronoun
        self.all_prepositional_or_nominal_prefixes = (
            self.prefix_universal | self.prefix_nominal
        )
        self.single_letter_proclitics = {
            p for p in self.prefix_universal if len(p) == 1
        }
        self.biradical_type_a_jussive_stems = {
            v[:-1] + self.phonology.shift_order(v[-1], 5)
            for v in getattr(self, "biradical_type_a_medial_a_verbs", set())
            if len(v) >= 2
        }

        # Declarative Atomic Multi-Character Affix Shields (Anti-Dissection Invariant)
        # Every multi-character prefix loaded from morphemes.json is inherently atomic
        fused_composite_prefixes = {
            lic + rel
            for lic in ("", "የ", "በ", "ለ", "ከ", "እንደ", "ስለ", "እስከ", "ወደ")
            for rel in ("ሚ", "ሚያ", "ምት", "ምን", "ምና", "ተ")
            if lic + rel not in ("", "ተ")
        }
        # Exclude nasal derived-voice combinations ('ተን', 'አን', 'ተም', 'አም') so passive 'ተ-'
        # and causative 'አ-' can peel cleanly from stems like ተንቀሳቀሰ, አንቀላፋ
        nasal_voice_stems = {"ተን", "አን", "ተም", "አም"}
        self.atomic_multi_prefixes = {
            p for p in (self.prefix_universal | self.prefix_nominal | self.all_verbal_prefixes)
            if len(p) > 1 and p not in fused_composite_prefixes and p not in nasal_voice_stems
        }
        self.atomic_multi_suffixes = {
            s
            for s in (
                self.suffix_nominal_plural_exclusive
                | self.suffix_nominal_abstract_exclusive
                | set(self.compound_plural_possessives.keys())
                | set(self.bipartite_applicative_mappings.keys())
            )
            if len(s) > 1
        }

        self.all_prefixes = sorted(
            self.prefix_universal | self.prefix_nominal | self.all_verbal_prefixes,
            key=len,
            reverse=True,
        )
        self.prefixes_by_initial = defaultdict(list)
        for p in self.all_prefixes:
            self.prefixes_by_initial[p[0]].append(p)

        clitics_list = sorted(
            self.suffix_universal_clitics | self.suffix_accusative,
            key=len,
            reverse=True,
        )
        self.clitics_by_final = defaultdict(list)
        for c in clitics_list:
            self.clitics_by_final[c[-1]].append(c)

        raw_sfx_list = sorted(
            (
                self.suffix_universal_clitics
                | self.suffix_nominal_derivational
                | self.suffix_nominal_abstract
                | self.suffix_nominal_plural
                | self.suffix_verb_pv
                | self.suffix_verb_auxiliary
                | self.suffix_verb_coverb
                | self.suffix_verb_object_clitics
            )
            - self.phonology.PURE_VOWEL_SUFFIXES
            - self.suffix_accusative,
            key=len,
            reverse=True,
        )
        self.fallback_suffixes_by_final = defaultdict(list)
        for s in raw_sfx_list:
            self.fallback_suffixes_by_final[s[-1]].append(s)
        self.applicative_suffixes_tuple = tuple(self.suffix_applicative_clitics)

    def normalize_inventories(self, norm_fn: Callable[[str], str]):
        set_attrs = (
            "prefix_universal",
            "prefix_relative_pronoun",
            "prefix_rel_imperfective_particle",
            "prefix_rel_imperfective_particle_allomorphs",
            "prefix_rel_imperfective_agreement",
            "single_letter_proclitics",
            "prefix_nominal_compound",
            "prefix_nominal_prepositional",
            "prefix_nominal_associatives",
            "prefix_nominal_locative",
            "prefix_nominal_privative",
            "prefix_nominal",
            "suffix_nominal_abstract",
            "suffix_nominal_derivational",
            "suffix_nominal_plural",
            "suffix_nominal_definiteness",
            "suffix_nominal_possessive",
            "suffix_nominal_instrumental",
            "suffix_nominal",
            "suffix_accusative",
            "suffix_universal_clitics",
            "suffix_pragmatic_clitics",
            "prefix_verb_pv",
            "prefix_verb_iv",
            "prefix_verb_jussive",
            "prefix_verb_subord_imp",
            "prefix_verb_modal",
            "prefix_verb_voice",
            "prefix_verb_continuous",
            "prefix_verb_primary_neg",
            "prefix_verb_subord_neg",
            "prefix_verb_atomic_negative",
            "prefix_infinitive_coalesced",
            "prefix_verb",
            "suffix_verb_pv",
            "suffix_verb_auxiliary",
            "suffix_verb_negative_finiteness",
            "suffix_verb_imperfective_relative_agreement",
            "suffix_verb_object_exclusive",
            "suffix_verb_object_clitics",
            "suffix_applicative_clitics",
            "suffix_verb_coverb",
            "suffix_verb_coverb_weak",
            "suffix_verb",
            "suffix_nominal_exclusive_plural",
            "suffix_nominal_exclusive_abstract",
            "suffix_nominal_exclusive_derivational",
            "suffix_nominal_exclusive_instrumental",
            "suffix_nominal_exclusive_definiteness",
            "suffix_nominal_exclusive_possessive",
            "prefix_nominal_exclusive_associatives",
            "prefix_nominal_exclusive_compound",
            "prefix_nominal_exclusive_prepositional",
            "suffix_verb_exclusive_subject",
            "suffix_verb_exclusive_object",
            "suffix_verb_applicative_exclusive",
            "suffix_verb_exclusive_coverb",
            "suffix_verb_exclusive_auxiliary",
            "suffix_verb_exclusive_negative_finiteness",
            "prefix_verb_exclusive_voice",
            "prefix_verb_exclusive_aspectual_perfective",
            "prefix_verb_exclusive_negative",
            "prefix_verb_exclusive_subordinate",
            "macro_prefix_nominal",
            "macro_prefix_vn",
            "macro_prefix_pv",
            "macro_prefix_iv",
            "macro_suffix_cn",
            "macro_suffix_mn",
            "macro_suffix_benefactive_aux",
            "macro_suffix_malfactive_aux",
            "macro_suffix_coverb_aux",
            "pronoun_stems",
            "verb_stems",
            "valid_2char_stems",
            "geez_plurals",
            "confirmed_sparse_nouns",
            "confirmed_sparse_verbs",
            "irregular_verbs_say",
            "biradical_type_b_verbs",
            "biradical_type_a_medial_a_verbs",
            "verbs_ending_in_gna",
            "imperfective_negative_prefixes",
            "verbal_prefixes_requiring_attestation",
            "closed_class_ideophones",
            "closed_class_quantifiers",
            "ambiguous_verbal_nouns",
            "inherent_terminal_u_stems",
        )
        for attr in set_attrs:
            if hasattr(self, attr):
                values = getattr(self, attr)
                if isinstance(values, set):
                    # Mutate in place rather than rebinding via setattr.
                    #
                    # Many inventories are bound to two names pointing at ONE set
                    # object (e.g. suffix_nominal_plural_exclusive and
                    # suffix_nominal_exclusive_plural), and only one spelling is
                    # listed above. Rebinding would leave the other alias pointing
                    # at the un-normalized set, silently desyncing them. The same
                    # applies to references cached by consumers before this runs
                    # (e.g. AmharicMorphologicalSegmenter's sfx_nominal/sfx_verb).
                    # Mutating the shared object keeps every alias and cached
                    # reference consistent.
                    normalized = {norm_fn(v) for v in values if v}
                    if normalized != values:
                        values.clear()
                        values.update(normalized)
        if hasattr(self, "voice_hierarchy"):
            self.voice_hierarchy = {
                k: [norm_fn(v) for v in vals]
                for k, vals in self.voice_hierarchy.items()
            }
            self.direct_causative_prefixes = tuple(self.voice_hierarchy.get("direct_causative", ["አ"]))
            self.indirect_causative_prefixes = tuple(self.voice_hierarchy.get("indirect_causative", ["አስ"]))
            self.non_active_prefixes = tuple(self.voice_hierarchy.get("non_active", ["ተ"]))
            self.primary_voice_prefixes = tuple(sorted(
                set(self.direct_causative_prefixes + self.indirect_causative_prefixes + self.non_active_prefixes),
                key=len,
                reverse=True,
            ))

        if hasattr(self, "possessive_1pl_endings"):
            self.possessive_1pl_endings = tuple(
                norm_fn(v) for v in self.possessive_1pl_endings
            )
        self.perfective_subject_urs = [
            (norm_fn(ur), p) for ur, p in self.perfective_subject_urs
        ]
        self.possessive_urs = [(norm_fn(ur), p) for ur, p in self.possessive_urs]
        self.instrument_noun_urs = [norm_fn(ur) for ur in self.instrument_noun_urs]

        self.weak_coverb_lemma_endings = tuple(
            norm_fn(v) for v in self.weak_coverb_lemma_endings
        )
        self.subject_e_exclusions = tuple(norm_fn(v) for v in self.subject_e_exclusions)
        self.derived_voice_prefixes = tuple(
            norm_fn(v) for v in self.derived_voice_prefixes
        )
        self.unambiguous_assoc_prefixes = tuple(
            norm_fn(v) for v in self.unambiguous_assoc_prefixes
        )

        self.auxiliary_coda_shifts = {
            norm_fn(k): norm_fn(v) for k, v in self.auxiliary_coda_shifts.items()
        }

        norm_compound_aux = {}
        for k, v in self.compound_auxiliary_suffixes.items():
            norm_compound_aux[norm_fn(k)] = {
                "stem_suffix": norm_fn(v.get("stem_suffix", "")),
                "affixes": [norm_fn(a) for a in v.get("affixes", [])],
                "rule": v.get("rule", ""),
            }
        self.compound_auxiliary_suffixes = norm_compound_aux

        self.compound_plural_possessives = {
            norm_fn(k): [norm_fn(a) for a in v]
            for k, v in self.compound_plural_possessives.items()
        }
        self.bipartite_applicative_mappings = {
            norm_fn(k): [norm_fn(a) for a in v]
            for k, v in self.bipartite_applicative_mappings.items()
        }

        self._rebuild_derived_inventories_and_indices()

    # ==================== PARADIGM & ROOT FORMALIZATION ====================

    def is_reduplicative_quadriradical(self, stem: str) -> bool:
        """
        Identifies 1-2-1-2 complete reduplication roots across both:
          - Perfective bases (orders 0,0,0,0 / exact character match): ለሰለሰ, በሰበሰ, ቀጠቀጠ, ፈጠፈጠ
          - Imperative command bases (orders 0,5,5,5 with C1=C3, C2=C4 identity): ለስልስ, በስብስ, ከስክስ, ለብልብ, ለምልም
        """
        if not stem or len(stem) != 4:
            return False
        if stem[:2] == stem[2:4]:
            return True
        b0, o0 = self.phonology.get_char_info(stem[0])
        b1, o1 = self.phonology.get_char_info(stem[1])
        b2, o2 = self.phonology.get_char_info(stem[2])
        b3, o3 = self.phonology.get_char_info(stem[3])
        # Covers command bases (0,5,5,5: ከትክት, ለስልስ) and converb bases (0,5,5,6 / 5,5,5,6: ከትክቶ, ቀልቅሎ, ቅጥቅጦ)
        is_redup_consonants = (b0 == b2 and b1 == b3)
        return is_redup_consonants and (
            (o0 == 0 and o1 == 5 and o2 == 5 and o3 == 5)
            or (o1 == 5 and o2 == 5 and o3 == 6)
        )

    def is_frequentative(self, stem: str) -> bool:
        """
        Identifies authentic frequentative internal reduplication:
        - Triliteral (len 4): ሰባበረ, ደጋገመ, ቀጣቀጠ (orders: 0, 3, 0, 0 / C1-C2-a-C2-ä-C3)
        - Quadriradical (len 5): ደመዳደመ, መርማመረ, ደንባበረ (orders: 0, 0/5, 3, 0, 0 / C1-C2-C3-a-C3-ä-C4)
        """
        if not stem or len(stem) not in (4, 5):
            return False
        orders = self.phonology.get_vocalic_orders(stem)
        if len(stem) == 4:
            if len(orders) != 4 or orders[1] != 3 or orders[2] != 0:
                return False
            base1, _ = self.phonology.get_char_info(stem[1])
            base2, _ = self.phonology.get_char_info(stem[2])
            return base1 == base2
        elif len(stem) == 5:
            if len(orders) != 5 or orders[2] != 3 or orders[3] != 0:
                return False
            base2, _ = self.phonology.get_char_info(stem[2])
            base3, _ = self.phonology.get_char_info(stem[3])
            return base2 == base3
        return False

    def paradigm_readings_for(self, form: str) -> Tuple[str, ...]:
        """Canonical verbal readings licensed for a surface form (see module fn)."""
        if not form:
            return ()
        return paradigm_matrix_readings(form)

    def is_paradigm_matrix_form(self, form: str) -> bool:
        """True when a surface form is listed in the shipped verb paradigm matrix.

        Used as the licence for the verbal side of bare nominal/verbal
        homographs; see :func:`paradigm_matrix_forms`.
        """
        if not form:
            return False
        return form in paradigm_matrix_forms()

    def is_licensed_defective_root(self, stem: str) -> bool:
        """Declaratively licenses biconsonantal/defective verbal bases from morphemes.json inventories."""
        if not stem or len(stem) != 2:
            return False
        if (
            stem in self.verb_stems
            or stem in self.auxiliary_verbs_standalone
            or stem in getattr(self, "defective_verbal_forms", {})
        ):
            return True
        orders = self.phonology.get_vocalic_orders(stem)
        if len(orders) != 2:
            return False
        o1, o2 = orders[0], orders[1]
        # Hollow Jussive / Imperfective bases (ቆም: 6-5, ሂድ: 2-5, ቁም: 1-5, ሄድ: 4-5, ጓዝ/ኳል/ሟጭ: 7-5)
        if o2 == 5 and o1 in (1, 2, 3, 4, 6, 7):
            return True
        # Dikala stative biradicals (7, 7: ሟሟ, ጓጓ)
        if o1 == 7 and o2 == 7:
            return True
        # Final-Weak Imperatives & Perfectives (ስማ: 5-3, ውጣ: 5-3, በላ: 0-3)
        if o1 in (0, 5) and o2 == 3:
            return (
                stem in self.valid_2char_stems
                or stem in getattr(self, "confirmed_sparse_verbs", set())
                or any(
                    c in self.valid_2char_stems
                    for c in self.generate_lemma_candidates(stem)
                )
            )
        # Defective Biradicals Geez-Geez (0, 0: ለየ, ቀረ, ሸጠ, ፈጀ, ፈጨ, ተወ, በጀ, ሰጠ, መሸ)
        if o1 == 0 and o2 == 0:
            return (
                stem in self.valid_2char_stems
                or stem in getattr(self, "confirmed_sparse_verbs", set())
                or any(
                    c in self.valid_2char_stems
                    for c in self.generate_lemma_candidates(stem)
                )
            )

        # Type B / Contracted Biradicals (ለይ, ሸኝ, በጅ, ተው, ፈጅ, ገጭ) (0, 5)
        if o1 == 0 and o2 == 5:
            return True
        # Universal Rabe-Geez (3, 0) (ያዘ, ዋለ, ዳነ, ጣለ, ዋጠ)
        # Words starting with ያ- (ያጨ, ያሸ, ያየ, ያዠ) are inflected (የ+አ-), only ያዘ is an autonomous root
        if o1 == 3 and o2 == 0:
            if stem.startswith("ያ"):
                return stem == "ያዘ"
            return True
        # Stative Biradicals (3, 3) MUST have identical consonants (ላላ, ሣሣ, ሳሳ, ባባ, ዳዳ, ፋፋ, ራራ)
        # Prevents coalesced forms like ያጣ, ያማ, ያራ (C1 != C2) from falsely matching defective roots
        if o1 == 3 and o2 == 3:
            if stem.startswith("አ"):
                return (
                    stem in self.valid_2char_stems
                    or stem in getattr(self, "confirmed_sparse_verbs", set())
                )
            # Prevents coalesced forms like ያጣ, ያማ, ያራ (C1 != C2) from falsely matching defective roots
            b1, _ = self.phonology.get_char_info(stem[0])
            b2, _ = self.phonology.get_char_info(stem[1])
            return b1 == b2
        # Hollow Perfective bases (ሆነ, ቆመ, ሞተ, ጦመ: 6-0, ሔደ, ሴሰ: 4-0)
        if o1 in (4, 6) and o2 == 0:
            return True
        # Hollow Converb stems (5-5: ልቅ, ስቅ, ሽጥ, ውጥ) & Surface 3MS coverbs (5-6: ልቆ, ስቆ, ሽጦ, ውጦ)
        if o1 == 5 and o2 in (5, 6):
            return True
        return False

    def is_licensed_2char_stem(self, stem: str) -> bool:
        """Queries the authoritative 2-radical TSV registry."""
        return is_licensed_2char_stem(stem)

    def is_licensed_2char_perfective_stem(self, stem: str) -> bool:
        """Queries whether stem is listed in the registry's first source column."""
        return is_licensed_2char_perfective_stem(stem)

    def is_licensed_2char_verbal_stem(self, stem: str) -> bool:
        """Queries the flat raw-text registry for verbal-affix licensing."""
        return is_licensed_2char_verbal_stem(stem)

    def is_licensed_2char_contingent_stem(self, stem: str) -> bool:
        """Queries the stem registry derived from the simple-contingent column."""
        return len(stem) == 2 and stem in TWO_CHAR_CONTINGENT_STEMS

    def is_licensed_2char_nominal(self, stem: str) -> bool:
        """Queries whether stem is in nominal column 20."""
        return is_licensed_2char_nominal(stem)

    # ==================== END PARADIGM FORMALIZATION ====================

    def is_verbal_noun_stem(self, stem: str) -> bool:
        return self.is_valid_infinitive_stem(stem)

    def is_abstract_noun_stem(self, stem: str) -> bool:
        """
        Validates genuine abstract noun suffixes (-ነት, -ስና, -ግልና, -ቁና).
        Excludes ambiguous single letters like 'ቴ' or 'ና'.
        """
        if not stem or len(stem) < 3:
            return False
        abstract_suffixes = getattr(
            self, "suffix_nominal_abstract_exclusive", {"ነት", "ስና", "ግልና", "ቁና"}
        )
        return any(stem.endswith(sfx) for sfx in abstract_suffixes)

    # Closed-class families are orthogonal top-level heads: ADP, ADV, CONJ,
    # INT and NUM are not members of the nominal family, and ideophones sit on
    # the verbal side.  Mass/abstract nouns are NOM:MN.  AUX is VB:AUX.
    _CLOSED_ORTHOGONAL_SUBTYPES: frozenset = frozenset({"CONJ", "ADP", "ADV", "INT", "NUM"})

    # Backwards-compatible alias: the closed-class subtypes that historically
    # carried a nominal macro reading.  Kept for callers written before the
    # notation change; membership now spans families.
    _CLOSED_NOMINAL_SUBTYPES: frozenset = frozenset(
        {"CONJ", "ADP", "ADV", "INT", "IDEO", "NUM", "MN"}
    )

    # Verbal template checkers used for dual-tag detection (populated lazily).
    # Maps a VerbalShape value string to the shape tag emitted in dual tags.
    _VERBAL_SHAPE_TAGS = {
        "COVERB":     "VB:COVERB",
        "PERFECTIVE":  "VB:PERFECTIVE",
        "IMPERFECTIVE": "VB:IMPERFECTIVE",
        "VN":          "VN",
    }

    def _closed_class_bare_tag(self, word: str) -> Optional[str]:
        """Return the canonical closed-class tag for a word, or None."""
        if word in ("ነበረ", "አለ"):
            return None
        if word in getattr(self, "auxiliary_verbs_standalone", set()):
            return "VB:AUX"
        if word in getattr(self, "closed_class_conjunctions", set()) or word in (
            "ታድያ", "ታዲያ",
        ):
            return "CONJ"
        if word in getattr(self, "closed_class_adpositions", set()):
            return "ADP"
        if word in getattr(self, "closed_class_adverbs", set()) or word in (
            "እንግዲህ", "እንግድህ", "ወዲህ", "ወድህ",
        ):
            return "ADV"
        if word in getattr(self, "closed_class_question_words", set()):
            return "INT"
        if word in getattr(self, "closed_class_ideophones", set()):
            return "VB:IDEO"
        if word in getattr(self, "closed_class_quantifiers", set()):
            return "NUM"
        return None

    def get_closed_class_pos(self, word: str) -> Optional[str]:
        """Return a canonical closed-class POS tag for a word, or None.

        Returns the bare family tag (``ADP``, ``ADV``, ``CONJ``, ``INT``,
        ``NUM`` for numerals/quantifiers), or ``VB:AUX`` / ``VB:IDEO`` for the
        verbal-side closed classes, without checking whether the word also
        satisfies a verbal template.  Use this for the Stage-1 invariant gate
        where you only need to know *whether* the word is in a closed-class
        list.

        For full dual-tag detection (closed-class AND verbal), use
        ``get_closed_class_pos_with_verbal_check()``.
        """
        return self._closed_class_bare_tag(word)

    def get_closed_class_pos_with_verbal_check(
        self,
        word: str,
        prefixes: Tuple[str, ...] = (),
        suffixes: Tuple[str, ...] = (),
    ) -> Optional[str]:
        """Return a POS tag for closed-class items, with dual-tag support.

        Many Amharic words are lexicalised in closed-class lists (adverbs,
        question words) but are also genuine verbal forms.  Examples:

        - ቀድሞ  (coverb of ቀደመ) → lexicalised as adverb
        - ቀጥሎ  (coverb of ቀጠለ) → lexicalised as adverb
        - ይልቅ  (imperfective of ልቀ) → lexicalised as adverb
        - ይበልጥ  (imperfective of በለጠ) → lexicalised as adverb
        - ለምን  (imperative of ለመነ) → lexicalised as question word

        When a word is in a closed-class list AND satisfies a verbal template,
        this method returns the disjunction of the two complete tags, with the
        superfamily repeated on each member (``ADP|VB:COVERB``) so that Layer 1
        does not silently suppress the verbal reading.

        If only closed-class (no verbal template match): returns the bare
        family tag (same as ``get_closed_class_pos()``).

        If not closed-class: returns None.
        """
        bare = self._closed_class_bare_tag(word)
        if bare is None:
            return None

        # Auxiliaries are always purely verbal — no dual check needed.
        if bare == "VB:AUX":
            return bare

        # Check whether the word also satisfies a verbal template.
        verbal_tag: Optional[str] = None

        if self.is_valid_verbal_noun_stem(word):
            verbal_tag = "VN"
        elif self.is_valid_coverb_stem(word):
            verbal_tag = "VB:COVERB"
        elif self.is_valid_perfective_template(word):
            verbal_tag = "VB:PERFECTIVE"
        elif self.is_valid_imperfective_stem(word):
            verbal_tag = "VB:IMPERFECTIVE"
        else:
            # Check with affix context (e.g. ይልቅ has imperfective prefix ይ-)
            stem_type = self.inspect_full_radical_template(word, prefixes, suffixes)
            if stem_type == "VN":
                verbal_tag = "VN"
            elif stem_type in ("PERFECTIVE", "IMPERFECTIVE", "COVERB"):
                verbal_tag = "VB:" + stem_type

        if verbal_tag is not None:
            return f"{bare}|{verbal_tag}"
        return bare

    def get_closed_class_subtype(self, macro_prefixed_tag: str) -> str:
        """Extract the bare subtype from a closed-class tag.

        ``"CONJ"`` → ``"CONJ"``,  ``"VB:AUX"`` → ``"AUX"``,
        legacy ``"NOM:CONJ"`` → ``"CONJ"``, anything else → ``""``.
        """
        return pos_tags.closed_subtype(macro_prefixed_tag)

    def is_valid_verbal_noun_stem(self, stem: str) -> bool:
        if not stem or len(stem) < 3 or len(stem) > 9:
            return False
        if stem in self.pronoun_stems or stem in ("ማን", "ማንም", "ማንኛ", "ማንኛውም"):
            return False
        if self.is_abstract_noun_stem(stem):
            return False

        # Finite compound auxiliaries (-Cʷal, -wal, -al) attach to subject concord (ይ-, ት-), never to infinitives (መ-, ማ-).
        # Infinitives of roots ending in radical 'ል' (መዋል, መዋዋል, ማስተዋል, መኳል, መጓል) are authentic verbal nouns.
        if stem.endswith("አል") and len(stem) >= 4:
            return False
        if stem.endswith("ል") and len(stem) >= 3:
            if stem[-2] in ("ኗ", "ሯ", "ሷ", "ጧ", "ሟ", "ፏ", "ጇ", "ጯ", "ጿ", "ዟ", "ቷ", "ቿ", "ኛ"):
                return False

        first_c = stem[0]
        infinitive_initials = {"መ", "ማ"} | {
            prefix[0] for prefix in self.prefix_infinitive_coalesced if prefix
        }
        if first_c not in infinitive_initials:
            return False

        orders = self.phonology.get_vocalic_orders(stem)
        if len(orders) != len(stem) or orders[-1] != 5:
            return False
        # Direct template match against canonical Leslau infinitive patterns
        if tuple(orders) in self.LESLAU_INFINITIVE_PATTERNS:
            # Weak infinitives ending natively in -at (-አት) obligatorily end in consonant 'ት'
            # Note: (3, 5, 3, 5) covers both final-weak (ማብላት) and hollow-Rabe (ማስያዝ, ማስጫን), so it is not restricted to 'ት'
            if tuple(orders) in ((0, 5, 3, 5), (0, 0, 3, 5), (0, 1, 3, 5)) and not stem.endswith("ት"):
                return False
            return True

        # Boundary Phonology Exposure for /mä-/ + /a-/ Coalescence (ማ- Infinitives)
        # Covers primary guttural/initial-weak (ማጨድ, ማሰብ) and derived a-, as-, astä-, an- stems (ማሳለፍ, ማስያዝ, ማስተጓጎል)
        if first_c == "ማ" and len(stem) >= 3:
            # Case A: Weak/Stative/Reduplicative infinitives ending in -ት (e.g. ማውጣት, ማጥፋት, ማንኳኳት)
            if stem.endswith("ት") and len(stem) >= 4:
                exposed_weak_verb = "አ" + stem[1:-1]
                if (
                    self.is_valid_perfective_template(exposed_weak_verb)
                    or exposed_weak_verb in self.verb_stems
                ):
                    return True

            # Case B: Reconstruct canonical 3MS perfective verb (Handles simplex, causative, and passive stems)
            reconstructed_verb = "አ" + stem[1:-1] + self.phonology.shift_order(stem[-1], 0)
            if (
                self.is_valid_perfective_template(reconstructed_verb)
                or reconstructed_verb in self.verb_stems
            ):
                return True

        return False

    is_valid_infinitive_stem = is_valid_verbal_noun_stem

    def is_valid_ideophone_template(self, stem: str) -> bool:
        if not stem or len(stem) < 2 or len(stem) > 7:
            return False
        # Ideophone stems cannot harbor unpeeled multi-character grammatical prefixes
        if any(stem.startswith(p) for p in self.all_prefixes if len(p) >= 2):
            return False
        if stem in self.verb_stems or stem in getattr(
            self, "auxiliary_verbs_standalone", set()
        ):
            return False
        if self.is_reduplicative_quadriradical(stem):
            return False

        KAEB, GEEZ, RABE, HAMES, SADES, SABE = 1, 0, 3, 4, 5, 6
        orders = self.phonology.get_vocalic_orders(stem)
        n = len(orders)
        if n < 2 or n != len(stem):
            return False

        if n == 2:
            o1, o2 = orders[0], orders[1]
            if o2 == SADES and o1 in (SADES, RABE, SABE, HAMES, GEEZ, KAEB):
                return True
        elif n == 3:
            o1, o2, o3 = orders[0], orders[1], orders[2]
            if o3 == SADES:
                if o1 == SADES and o2 == SADES:
                    return True
                if o1 == GEEZ and o2 == GEEZ:
                    return True
                if o1 == RABE and o2 == GEEZ:
                    return True
                if o1 == SABE and o2 == GEEZ:
                    return True
                if o1 == SADES and o2 == GEEZ:
                    return True
        elif n in (4, 5, 6, 7):
            half = n // 2
            if stem[:half] == stem[half:]:
                return True
            if stem[-2:] == stem[-4:-2] and all(o == SADES for o in orders):
                return True
            if all(o == SADES for o in orders):
                return True

        return False

    def is_valid_perfective_template(self, stem: str) -> bool:
        """
        Universal Semitic Perfective Template Validator.
        Validates primary roots and derived stems (with voice prefixes stripped)
        strictly against canonical Leslau prosodic radical matrices. Zero hardcoded checks.
        """
        if not stem or len(stem) < 2 or stem.startswith("እ"):
            return False
        if stem in self._perf_template_cache:
            return self._perf_template_cache[stem]
        if self.is_valid_infinitive_stem(stem) or self.is_abstract_noun_stem(stem):
            self._perf_template_cache[stem] = False
            return False
        if stem.endswith("ኛ") and stem not in self.verbs_ending_in_gna:
            self._perf_template_cache[stem] = False
            return False
        norm_s = stem.replace("ሙዋ", "ሟ").replace("ቱዋ", "ቷ").replace("ሩዋ", "ሯ")
        orders = self.phonology.get_vocalic_orders(norm_s)
        if not orders:
            self._perf_template_cache[stem] = False
            return False
        # Reduplicative 1-2-1-2 is ONLY a 3MS perfective if all vowels are Geez (0, 0, 0, 0) (e.g. ደለደለ, ሰበሰበ, ለሰለሰ)
        # Stems with orders (0, 5, 5, 5) (ደልድል, ከልክል, ቀልቅል, ወልውል) are Jussive commands, never 3MS perfectives
        if self.is_reduplicative_quadriradical(norm_s) or self.is_frequentative(norm_s):
            self._perf_template_cache[stem] = True
            return True
        # If stem ends in Sades (6th-order), reconstruct canonical 3MS coda (C_6 -> C_1)
        if norm_s.endswith("ው"):
            self._perf_template_cache[stem] = False
            return False
        # If stem ends in Sades (6th-order), reconstruct canonical 3MS coda (C_6 -> C_1)
        if orders[-1] == 5 and len(norm_s) >= 3:
            c1_coda = self.phonology.shift_order(norm_s[-1], 0)
            eval_stem = norm_s[:-1] + c1_coda
        elif orders[-1] in (0, 3, 7):
            eval_stem = norm_s
        else:
            self._perf_template_cache[stem] = False
            return False
        eval_orders = tuple(self.phonology.get_vocalic_orders(eval_stem))
        # 1. Primary Radical Root match
        if eval_orders in self.LESLAU_PERFECTIVE_PATTERNS:
            if len(norm_s) == 2:
                is_valid_2c = self.is_licensed_defective_root(norm_s)
                self._perf_template_cache[stem] = is_valid_2c
                return is_valid_2c
            self._perf_template_cache[stem] = True
            return True

        # 2. Derived Voice Stem match (Strip voice prefixes by descending length and verify root)
        for pfx_len, prefixes in (
            (3, ("አስተ", "አስታ")),
            (2, ("አስ", "ተን", "አን", "ተም", "አም")),
            (1, ("ተ", "አ")),
        ):
            for p in prefixes:
                if eval_stem.startswith(p) and len(eval_stem) > pfx_len + 1:
                    root = eval_stem[pfx_len:]
                    root_orders = tuple(self.phonology.get_vocalic_orders(root))
                    if (
                        root_orders in self.LESLAU_PERFECTIVE_PATTERNS
                        or self.is_frequentative(root)
                        or self.is_reduplicative_quadriradical(root)
                    ):
                        self._perf_template_cache[stem] = True
                        return True
        self._perf_template_cache[stem] = False
        return False

    def generate_nominal_cohort_bundles(self, stem: str) -> Tuple[
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[str, ...],
    ]:
        """
        Synthesizes 7 diagnostic nominal cohort bundles derived STRICTLY from 100% EXCLUSIVE
        nominal morphology (zero verbal overlap). Prevents verbal person inflections
        (-ህ, -ሽ, -ቸው, እ-) from producing spurious nominal points.

        Cached via the instance-bounded _nom_cohort_cache (cleared on config reload)
        rather than lru_cache, so load_morphemes_config invalidation actually works.
        """
        if not stem or len(stem) < 2:
            return ((), (), (), (), (), (), ())
        if stem in self._nom_cohort_cache:
            return self._nom_cohort_cache[stem]

        last_char = stem[-1]
        _, last_order = self.phonology.get_char_info(last_char)

        # 1. Exclusive Nominal Compounds (ባለ-, ስነ-, ጸረ-)
        compound_prefixes = (
            self.prefix_nominal_compound_exclusive
            if hasattr(self, "prefix_nominal_compound_exclusive")
            else self.prefix_nominal_compound
        )
        b_compound = [p + stem for p in compound_prefixes]

        # 2. Exclusive Prepositions (ወደ-, ያለ- only; NO ambiguous እ-)
        prep_prefixes = (
            self.prefix_nominal_prepositional_exclusive
            if hasattr(self, "prefix_nominal_prepositional_exclusive")
            else {"ወደ", "ያለ"}
        )
        b_prep = [p + stem for p in prep_prefixes]

        # 3. Exclusive Associatives (እነ-, የእነ-, በእነ-, ከእነ-)
        assoc_prefixes = (
            self.prefix_nominal_associatives_exclusive
            if hasattr(self, "prefix_nominal_associatives_exclusive")
            else self.unambiguous_assoc_prefixes
        )
        b_assoc = [p + stem for p in assoc_prefixes]

        # 4. Exclusive Definiteness + Accusative Co-occurrence (-ውን, -ዋን, -ቷን)
        # (Isolated -ው and -ን are ambiguous, but co-occurring Def+Acc is 100% nominal)
        def_acc = []
        if last_order in (0, 2, 3, 4, 6):
            def_acc.extend([stem + "ውን", stem + "ዋን"])
        elif last_order == 5:
            def_acc.extend([stem + "ውን", stem + "ዋን", stem[:-1] + "ቷን"])

        # 5. Exclusive Possessives (-ችን, -አችን, -ዬ, -ዎት only; NO ambiguous -ህ, -ሽ, -ቸው, -ዋ)
        # (Verbs NEVER take -ችን/-አችን or polite -ዎት)
        poss = []
        if last_order in (0, 2, 3, 4, 6):
            poss.extend([stem + "ችን", stem + "ዬ", stem + "ዎት"])
        elif last_order == 5:
            shifted_a = self.phonology.shift_order(last_char, 3)
            poss.extend([stem[:-1] + shifted_a + "ችን", stem + "ችን", stem + "ዎት"])

        # 6. Exclusive Plurals (Phonologically synthesized via phonology.synthesize_plural)
        plurals = list(self.phonology.synthesize_plural(stem))
        
        if last_order == 5:
            shifted_o = self.phonology.shift_order(last_char, 6)
            p_stem = stem[:-1] + shifted_o
            plurals.append(p_stem + "ቹ")
            plurals.append(p_stem + "ቹን")
            plurals.append(p_stem + "ቻችን")
            plurals.append(p_stem + "ቻቸውን")

        # 7. Exclusive Abstract & Derivational (-ነት, -ስና, -ግልና, -ቁና)
        # (Verbal roots never directly carry -ነት)
        abstract_suffixes = (
            self.suffix_nominal_abstract_exclusive
            if hasattr(self, "suffix_nominal_abstract_exclusive")
            else {"ነት", "ስና", "ግልና", "ቁና"}
        )
        b_net = [stem + sfx for sfx in abstract_suffixes]

        bundles = (
            tuple(dict.fromkeys(b_compound)),
            tuple(dict.fromkeys(b_prep)),
            tuple(dict.fromkeys(b_assoc)),
            tuple(dict.fromkeys(def_acc)),
            tuple(dict.fromkeys(poss)),
            tuple(dict.fromkeys(plurals)),
            tuple(dict.fromkeys(b_net)),
        )
        self._nom_cohort_cache[stem] = bundles
        return bundles

    def generate_verbal_cohort_bundles(self, stem: str) -> Tuple[
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[str, ...],
    ]:
        """
        Declaratively synthesizes 7 verbal cohort bundles directly from morphemes.json:
        Tier 1 (Core Finite Concord - 30 pts each):
          [0] b_subj:      suffix_verb_exclusive_subject (ኩ, ኹ, ሁ, ክ, ኽ, ኧች, ንእ - no bare ን)
          [1] b_obj:       suffix_verb_object_exclusive (ኝ, አት, ከን)
          [2] b_aux:       suffix_verb_exclusive_auxiliary (አል, አለሁ, አለች, ል, ሉ, ዋል, ዋለች)
        Tier 2 (Aspectual & Dependent - 10 pts each):
          [3] b_neg:       prefix_verb_exclusive_negative (አል, አላ, አይ, ሳይ, የማይ, እንዳይ, ስለማይ)
          [4] b_voice_asp: prefix_verb_exclusive_aspectual_perfective | prefix_verb_exclusive_voice (የተ, በተ, አስ, አስተ)
          [5] b_conv:      suffix_verb_exclusive_coverb (ቼ, ቶ, ተው, ተን, ታችሁ, ተህ, ተሽ)
          [6] b_appl:      suffix_verb_applicative_exclusive (ልኝ, ለት, ላት, ብኝ, በት, ባቸው)
        """
        if not stem or len(stem) < 2:
            return ((), (), (), (), (), (), ())

        if stem in self._verb_cohort_cache:
            return self._verb_cohort_cache[stem]

        citations = self.generate_lemma_candidates(stem)
        if not citations:
            return ((), (), (), (), (), (), ())

        b_subj, b_obj, b_aux, b_neg, b_voice_asp, b_conv, b_appl = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for cit in citations:
            last_c = cit[-1]
            _, l_ord = self.phonology.get_char_info(last_c)
            s_coda = cit[:-1] + self.phonology.shift_order(last_c, 5)

            # [0] Tier 1: Exclusive Subject Agreement (Driven by suffix_verb_exclusive_subject)
            for s_sfx in self.suffix_verb_exclusive_subject:
                if s_sfx == "ኧች":
                    b_subj.append(cit + "ች")
                elif l_ord == 0 and len(cit) >= 3:
                    b_subj.append(s_coda + s_sfx)
                else:
                    b_subj.append(cit + s_sfx)

            # [1] Tier 1: Exclusive Direct Object Clitics (Driven by suffix_verb_object_exclusive)
            for o_sfx in self.suffix_verb_object_exclusive:
                if o_sfx == "አት":
                    if l_ord == 0 and len(cit) >= 3:
                        b_obj.append(
                            cit[:-1] + self.phonology.shift_order(last_c, 3) + "ት"
                        )
                else:
                    b_obj.append(cit + o_sfx)

            # [2] Tier 1: Exclusive Auxiliaries (Driven by suffix_verb_exclusive_auxiliary)
            for aux in self.suffix_verb_exclusive_auxiliary:
                if aux.startswith("ዋ") and len(cit) >= 2:
                    b_aux.append(s_coda + aux)
                elif aux.startswith("አ") and len(cit) >= 2:
                    b_aux.append(cit + aux)

            # [3] Tier 2: Exclusive Negative Prefixes (Driven by prefix_verb_exclusive_negative)
            for neg_pfx in self.prefix_verb_exclusive_negative:
                b_neg.append(neg_pfx + cit)

            # [4] Tier 2: Exclusive Voice & Aspectual Prefixes
            combined_voice_asp = (
                self.prefix_verb_exclusive_aspectual_perfective
                | self.prefix_verb_exclusive_voice
            )
            for v_pfx in combined_voice_asp:
                b_voice_asp.append(v_pfx + cit)

            # [5] Tier 2: Exclusive Converbs (Driven by suffix_verb_exclusive_coverb)
            for c_sfx in self.suffix_verb_exclusive_coverb:
                if c_sfx == "ቼ" and len(cit) >= 2:
                    b_conv.append(cit[:-1] + "ቼ")
                else:
                    b_conv.append(s_coda + c_sfx)

            # [6] Tier 2: Exclusive Applicatives (Driven by suffix_verb_applicative_exclusive)
            for appl in self.suffix_verb_applicative_exclusive:
                b_appl.append(cit + appl)

        bundles = (
            tuple(dict.fromkeys(b_subj)),
            tuple(dict.fromkeys(b_obj)),
            tuple(dict.fromkeys(b_aux)),
            tuple(dict.fromkeys(b_neg)),
            tuple(dict.fromkeys(b_voice_asp)),
            tuple(dict.fromkeys(b_conv)),
            tuple(dict.fromkeys(b_appl)),
        )
        self._verb_cohort_cache[stem] = bundles
        return bundles

    def generate_lemma_candidates(
        self, stem: str, shape: Optional[VerbalShape] = None
    ) -> Tuple[str, ...]:
        """Return rule-derived citation candidates without lexical attestation."""
        return self.lemma_candidate_generator.generate_candidates(stem, shape=shape)

    def lemmatize(
        self, stem: str, shape: Optional[VerbalShape] = None
    ) -> Tuple[str, ...]:
        """Compatibility alias; prefer :meth:`generate_lemma_candidates`."""
        return self.generate_lemma_candidates(stem, shape=shape)

    def reconstruct_all_verbal_lemmas(
        self, stem: str, shape: Optional[VerbalShape] = None
    ) -> Tuple[str, ...]:
        """Compatibility facade for the former reconstruction API."""
        return self.generate_lemma_candidates(stem, shape=shape)

    def get_citation_forms(self, stem: str) -> Tuple[str, ...]:
        return self.generate_lemma_candidates(stem)

    def is_valid_3ms_perfective_form(
        self, stem: str, checker: Optional[Callable[[str], bool]] = None
    ) -> bool:
        if not stem or len(stem) < 2:
            return False
        _, last_order = self.phonology.get_char_info(stem[-1])
        if last_order == 5:
            return False
        if stem in self.verb_stems or self.is_valid_perfective_template(stem):
            return True
        citations = self.generate_lemma_candidates(stem)
        if any(
            c in self.verb_stems or self.is_valid_perfective_template(c)
            for c in citations
        ):
            return True
        return bool(checker and any(checker(c) for c in citations))

    def is_surface_finite_verb(self, stem: str) -> bool:
        if not stem or len(stem) < 2:
            return False
        _, last_order = self.phonology.get_char_info(stem[-1])
        if last_order == 5:
            return False
        return stem in self.verb_stems or self.is_valid_perfective_template(stem)

    def is_attested_verbal_stem(
        self, stem: str, checker: Optional[Callable[[str], bool]] = None
    ) -> bool:
        if not stem or len(stem) < 2:
            return False
        if stem in self.verb_stems or self.is_valid_perfective_template(stem):
            return True
        citations = self.generate_lemma_candidates(stem)
        if any(
            c in self.verb_stems or self.is_valid_perfective_template(c)
            for c in citations
        ):
            return True
        return bool(checker and any(checker(c) for c in citations))

    def is_valid_weak_coverb_stem(
        self,
        stem: str,
        checker: Optional[Callable[[str], bool]] = None,
        suffix: str = "",
    ) -> bool:
        if not stem or len(stem) < 2:
            return False
        first_c, last_c = stem[0], stem[-1]
        _, first_order = self.phonology.get_char_info(first_c)
        _, last_order = self.phonology.get_char_info(last_c)

        if last_order != 5 and not stem.endswith(("ታ", "ቶ", "ተው", "ተን", "ተህ", "ተሽ")):
            return False
        if len(stem) == 2 and first_order not in (0, 3, 4, 6):
            return False

        citations = self.generate_lemma_candidates(stem)
        if suffix in ("ታ", "ቶ", "ተው", "ተን", "ታችሁ", "ታቹ"):
            if not any(c.endswith(self.weak_coverb_lemma_endings) for c in citations):
                return False

        if any(
            c in self.verb_stems or self.is_valid_perfective_template(c)
            for c in citations
        ):
            return True
        return bool(checker and any(checker(c) for c in citations))

    def inspect_full_radical_template(
        self, stem: str, prefixes: Tuple[str, ...] = (), suffixes: Tuple[str, ...] = ()
    ) -> str:
        """DEPRECATED: Unifies under classify_licensed_verbal_shape single source of truth."""
        cache_key = (stem, prefixes, suffixes)
        if cache_key in self._radical_template_cache:
            return self._radical_template_cache[cache_key]
        shape = self.classify_licensed_verbal_shape(prefixes, suffixes, stem=stem)
        _SHAPE_TO_LEGACY = {
            VerbalShape.SHAPE_A_PERFECTIVE: "PERFECTIVE",
            VerbalShape.SHAPE_B_COVERB: "COVERB",
            VerbalShape.SHAPE_C_IMPERFECTIVE: "IMPERFECTIVE",
            VerbalShape.SHAPE_D_JUSSIVE_IMPERATIVE: "JUSSIVE_IMPERATIVE",
            VerbalShape.SHAPE_E_INFINITIVE: "VN",
        }
        result = _SHAPE_TO_LEGACY.get(shape, "CN")
        self._radical_template_cache[cache_key] = result
        return result


    def is_valid_imperfective_stem(self, stem: str) -> bool:
        if not stem or len(stem) < 2:
            return False
        orders = tuple(self.phonology.get_vocalic_orders(stem))
        if len(orders) != len(stem):
            return False

        if orders in self.LESLAU_IMPERFECTIVE_PATTERNS:
            return True

        # Primary voice inspection: check root under declarative voice prefixes from morphemes.json
        for vp in getattr(self, "primary_voice_prefixes", ("አስ", "ተ", "አ")):
            if stem.startswith(vp) and len(stem) > len(vp) + 1:
                root_orders = tuple(self.phonology.get_vocalic_orders(stem[len(vp):]))
                if root_orders in self.LESLAU_IMPERFECTIVE_PATTERNS:
                    return True

        return False

    def is_valid_jussive_stem(self, stem: str) -> bool:
        if not stem or len(stem) < 2:
            return False
        orders = tuple(self.phonology.get_vocalic_orders(stem))
        if len(orders) != len(stem):
            return False

        if (
            orders in self.LESLAU_JUSSIVE_PATTERNS
            or orders in self.BOUND_JUSSIVE_STEM_PATTERNS
        ):
            return True

        # Primary voice inspection: check root under declarative voice prefixes from morphemes.json
        for vp in getattr(self, "primary_voice_prefixes", ("አስ", "ተ", "አ")):
            if stem.startswith(vp) and len(stem) > len(vp) + 1:
                root_orders = tuple(self.phonology.get_vocalic_orders(stem[len(vp):]))
                if (
                    root_orders in self.LESLAU_JUSSIVE_PATTERNS
                    or root_orders in self.BOUND_JUSSIVE_STEM_PATTERNS
                    or root_orders in self.PASSIVE_JUSSIVE_ROOT_PATTERNS
                    or root_orders in self.CAUSATIVE_JUSSIVE_ROOT_PATTERNS
                ):
                    return True

        return False

    def has_reduced_coverb_radical(self, stem: str) -> bool:
        """True when a non-final radical of *stem* stands in the Sadis grade.

        The 3MS base perfective spells its radicals with full vowels (ቸገረ,
        ራበ, ጠማ); the coverb reduces one radical to the bare consonant (ቸግሮ,
        ተፈቃቅረ-, ተባብለ-, where ቅ/ብ stand in Sadis).  A Sadis anywhere before
        the final position is therefore a coverb grade, and a final ው on such
        a stem is the 3PL coverb ending (ተፈቃቅረ-ው, ተባብለ-ው), not an object
        clitic on a perfective.  Stems of one syllable have no radical to
        reduce, so they are left to the grade test proper.
        """
        if len(stem) < 3:
            return False
        orders = self.phonology.get_vocalic_orders(stem)
        return any(order == 5 for order in orders[:-1])

    def is_valid_coverb_stem(self, stem: str) -> bool:
        """Return True if *stem* is a valid coverb surface form.

        Handles primary roots and derived-voice stems (passive ተ-, indirect
        causative አስ-/አስተ-) by stripping voice prefixes before checking the
        coverb vocalic pattern.  This mirrors the stripping logic already
        present in ``is_valid_perfective_template()``.

        Affected forms that were previously missed:
          ተለምዶ  ተለቅሞ  ተለቃቅሞ  ተለብቦ  ተለግሞ   (passive ተ-)
          አስለብቦ  አስተለምዶ                    (causative አስ-/አስተ-)
        """
        if not stem or len(stem) < 2 or len(stem) > 10:
            return False
        if stem.endswith("ቼ"):
            return True

        _, last_order = self.phonology.get_char_info(stem[-1])
        if last_order == 4 and stem[-1] in self.phonology.PALATAL_CONSONANTS:
            return True

        orders = self.phonology.get_vocalic_orders(stem)
        if len(orders) != len(stem):
            return False

        if tuple(orders) in self.LESLAU_COVERB_PATTERNS:
            return True

        # Strip derived-voice prefixes (longest first) and check the root.
        # Order matters: አስተ (3) before አስ (2) before ተ/አ (1).
        for pfx_len, voice_prefixes in (
            (3, ("አስተ", "አስታ")),
            (2, ("አስ", "ተን", "አን", "ተም", "አም")),
            (1, ("ተ", "አ")),
        ):
            for vp in voice_prefixes:
                if stem.startswith(vp) and len(stem) > pfx_len + 1:
                    root = stem[pfx_len:]
                    root_orders = tuple(self.phonology.get_vocalic_orders(root))
                    if root_orders in self.LESLAU_COVERB_PATTERNS:
                        return True
                    # Root may itself be a frequentative or reduplicative
                    if self.is_frequentative(root) or self.is_reduplicative_quadriradical(root):
                        return True

        return False

    is_valid_gerundive_stem = is_valid_coverb_stem

    def is_valid_stem_for_affixes(
        self, stem: str, prefixes: Tuple[str, ...] = (), suffixes: Tuple[str, ...] = ()
    ) -> bool:
        cache_key = (stem, prefixes, suffixes)
        if cache_key in self._stem_affix_cache:
            return self._stem_affix_cache[cache_key]
        stem_type = self.inspect_full_radical_template(stem, prefixes, suffixes)

        has_exclusive_nom_sfx = any(
            s in self.exclusive_nominal_suffixes for s in suffixes
        )
        has_exclusive_vrb_sfx = any(
            s in self.exclusive_verbal_suffixes for s in suffixes
        )
        has_exclusive_nom_pfx = any(
            p in self.exclusive_nominal_prefixes for p in prefixes
        )
        has_exclusive_vrb_pfx = any(
            p in self.exclusive_verbal_prefixes for p in prefixes
        )

        has_prep_pfx = any(p in self.prefix_universal for p in prefixes)
        has_voice_pfx = any(p in self.prefix_verb_voice for p in prefixes)

        if has_exclusive_nom_sfx and (
            has_exclusive_vrb_sfx or has_exclusive_vrb_pfx or has_voice_pfx
        ):
            return False
        if has_exclusive_vrb_sfx and has_exclusive_nom_pfx:
            return False

        has_pure_verbal_pfx = any(p in self.exclusive_verbal_prefixes for p in prefixes)
        if has_exclusive_nom_pfx and has_pure_verbal_pfx:
            return False

        if (
            has_prep_pfx
            and stem_type == "PERFECTIVE"
            and not suffixes
            and not has_voice_pfx
        ):
            if not (
                self.is_valid_3ms_perfective_form(stem)
                or stem in self.verb_stems
                or self.is_licensed_2char_perfective_stem(stem)
                or any(p in self.prefix_relative_pronoun for p in prefixes)
            ):
                return False

        if stem_type in ("CN", "MN", "PN", "AN", "PRO", "IDEO") and (
            has_exclusive_vrb_sfx or has_voice_pfx
        ):
            return False

        if stem_type == "VN" and has_exclusive_vrb_sfx:
            return False

        if stem_type in ("PERFECTIVE", "IMPERFECTIVE", "JUSSIVE_IMPERATIVE", "COVERB"):
            if has_exclusive_nom_sfx:
                return False

        if stem_type == "VN":
            has_verbal_subject = any(
                s in self.suffix_verb_exclusive_subject for s in suffixes
            )
            has_coverb = any(s in self.suffix_verb_exclusive_coverb for s in suffixes)
            has_applicative = any(
                s in self.suffix_applicative_clitics for s in suffixes
            )
            has_associative = any(
                p in self.prefix_nominal_associatives for p in prefixes
            )
            if has_verbal_subject or has_coverb or has_applicative or has_associative:
                return False

        rel_particle_only = (
            self.prefix_rel_imperfective_particle
            | self.prefix_rel_imperfective_particle_allomorphs
        )
        rel_licensors = self.prefix_universal | self.prefix_relative_pronoun
        for i, p in enumerate(prefixes):
            if p in rel_particle_only and (
                i == 0 or prefixes[i - 1] not in rel_licensors
            ):
                self._stem_affix_cache[cache_key] = False
                return False
        self._stem_affix_cache[cache_key] = True
        return True


    # ------------------------------------------------------------------
    # Master Decomposition Dispatcher
    # ------------------------------------------------------------------

    def decompose_all_morphology(
        self,
        word: str,
        prefixes: Tuple[str, ...] = (),
        suffixes: Tuple[str, ...] = (),
        known_stem_checker: Optional[Callable[[str], bool]] = None,
        known_verbal_stem_checker: Optional[Callable[[str], bool]] = None,
        include_raw_suffix_fallback: bool = False,
    ) -> List[MorphSplit]:
        if not word or len(word) < 2 or self.is_reduplicative_quadriradical(word):
            return []

        ctx = DecomposeContext(
            prefixes=prefixes,
            suffixes=suffixes,
            known_stem_checker=known_stem_checker,
            known_verbal_stem_checker=known_verbal_stem_checker,
            include_raw_suffix_fallback=include_raw_suffix_fallback,
            has_exclusive_verb_pfx=any(p in self.exclusive_verbal_prefixes for p in prefixes),
            has_exclusive_verb_sfx=any(s in self.exclusive_verbal_suffixes for s in suffixes),
            has_poss_or_def_sfx=any(
                s in self.suffix_nominal_possessive or s in self.suffix_nominal_definiteness
                for s in suffixes
            ),
            has_negative_context=(
                any(p in self.prefix_verb_atomic_negative for p in prefixes)
                or word.startswith(("አል", "አላ", "አይ", "አት", "አን", "ሳይ", "እንዳይ", "የማይ", "ስለማይ"))
            ),
            has_prep=(
                any(p in self.prefix_universal for p in prefixes)
                or (bool(word) and word[0] in self.prefix_universal)
            ),
        )

        return self.peeler.decompose(word, ctx)

    def decompose_geez_plural(self, word: str) -> Optional[Tuple[str, str]]:
        """
        Isolated Ge'ez Plural Decomposer.
        Inverts classical -at (-አት) plurals.
        """
        if not word or len(word) < 3:
            return None

        if word not in self.geez_plurals:
            return None

        if word.endswith("ት") and len(word) >= 3:
            penult_c = word[-2]
            _, penult_ord = self.phonology.get_char_info(penult_c)
            if penult_ord == 3:
                sades_c = self.phonology.shift_order(penult_c, 5)
                singular_stem = word[:-2] + sades_c
                return singular_stem, "አት"

        if word.endswith("ያት") and len(word) >= 4:
            return word[:-2], "አት"

        return None
