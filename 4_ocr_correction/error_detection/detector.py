# detector.py — Production OCR Error Detection Engine with LightGBM (Authoritative v3)
# Roadmap: LightGBM is sole authoritative detector, rule-based is fallback only
# Stage 1 fixes: strict NFC+Amharic normalization, NaN for missing, fixed sub key
# Stage 2: no manual floors, isotonic calibration, cost-sensitive thresholds
from __future__ import annotations

# --- make repo_paths importable from any working directory -------------------
import os as _os
import sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(
        _os.path.join(_root, "tools", "stage_paths.py")):
    _root = _os.path.dirname(_root)
# tools/ first so stage_paths and repo_paths resolve, then register every stage
# folder so cross-stage imports work by bare module name. This replaces the
# thirteen root-level shims that used to forward those imports by hand.
_sys.path.insert(0, _os.path.join(_root, "tools"))
import stage_paths  # noqa: E402  (imported after the path bootstrap)
stage_paths.register(_root)
import repo_paths  # noqa: E402
# ----------------------------------------------------------------------------

import logging
import os
import re
import math
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

SymSpell = Verbosity = None
kenlm = None

def _load_symspell():
    global SymSpell, Verbosity
    if SymSpell is None:
        from symspellpy import SymSpell as _SymSpell, Verbosity as _Verbosity
        SymSpell, Verbosity = _SymSpell, _Verbosity
    return SymSpell, Verbosity

def _load_kenlm():
    global kenlm
    if kenlm is None:
        try:
            import kenlm as _kenlm
            kenlm = _kenlm
        except ImportError:
            kenlm = False
    return kenlm

try:
    import joblib
except ImportError:
    joblib = None

try:
    import numpy as np
except ImportError:
    np = None

from error_detection.normalizer import AmharicNormalizer
from error_detection.confusion import ConfusionMatrix
from error_detection.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    NAN_SCHEMA_VERSION,
    FeatureContext,
    FeatureError,
    cheap_signals,
    clean_surface,
    extract_vector,
    missing_features,
    static_features,
    with_context,
)
from error_detection.signals import (
    TokenSignals, Verdict,
    compute_surface_signals, compute_lexical_signals,
    compute_one_flip_signals,
)
from error_detection.scoring import combine_signals, DEFAULT_WEIGHTS
from error_detection.utils import MODEL_RARE_GLYPHS as _MODEL_RARE_GLYPHS

logger = logging.getLogger("ocr_error_detection")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)

# Lexicon the shipped model was trained against: main_i7 with freq>=500,
# which normalizes to exactly 54,057 canonical types (README "FINAL
# authoritative", and the note inside Model/detector_lgbm_conffree.pkl).
# A model payload can carry its own lexicon_min_freq / lexicon_max_types;
# those win. Caller-supplied lexicons keep whatever the caller asks for.
DEFAULT_DICT_MIN_FREQ = 500
DEFAULT_MAX_LEXICON_TYPES = None

CANDIDATE_LEXICON_PATHS = [
    os.path.join(_HERE, "Model", "main_i7_frq10.txt"),  # 737k types, 15MB, from bigger data, freq>=10, max freq 5.6M, cleaner
    os.path.join(_HERE, "main_i7_frq10.txt"),
    os.path.join(_HERE, "Model", "local_lexicon.txt"),  # 750k types, 15MB, from 4.4M tokens
    os.path.join(_HERE, "local_lexicon.txt"),
    os.path.join(_PARENT, "1_data_preparation", "data", "amh_final_training_text.txt"),
]

CANDIDATE_CONFUSION_PATHS = [
    os.path.join(_HERE, "Model", "confusion_matrix.json"),  # single source, no symlinks
    os.path.join(_PARENT, "4_ocr_correction", "confusion_matrix.json"),
    os.path.join(_PARENT, "confusion_matrix.json"),
]

CANDIDATE_MODEL_PATHS = [
    os.path.join(_HERE, "Model", "detector_lgbm_conffree.pkl"),  # conf-free, no ocr_conf, 14 feats, catches 90.8% high-conf errors
    os.path.join(_HERE, "Model", "detector_lgbm_v3.pkl"),  # v3 calibrated, NaN for missing conf
    os.path.join(_HERE, "Model", "detector_lgbm_v4.pkl"),
    os.path.join(_HERE, "Model", "detector_lgbm_v2.pkl"),  # v2: 96.7% recall but trusts conf (56% high-conf caught)
    os.path.join(_HERE, "Model", "detector_lgbm.pkl"),
    os.path.join(_HERE, "detector_lgbm.pkl"),
    os.path.join(_HERE, "..", "archive", "models", "detector_lgbm_conffree.pkl"),
    os.path.join(_HERE, "..", "archive", "models", "detector_lgbm_v3.pkl"),
]

CANDIDATE_CHAR_LM_PATHS = [
    os.path.join(_HERE, "Model", "amharic_char_model.klm"),
    repo_paths.char_kenlm(),
]

PUNCT_RE = re.compile(r'[^\w\s]|\d')
ETHIOPIC_PUNCT_RE = re.compile(r'[፡።፣፤፥፦፧፨]')
# One definition, in utils.py, re-exported here for the analysis scripts in
# tools/analysis/ that import it from this module.
RARE_GLYPHS = set(_MODEL_RARE_GLYPHS)

def _resolve_first_existing(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def _canonical_form(word: str, normalizer: AmharicNormalizer) -> str:
    """Canonical Ingestion Boundary: NFC + Amharic strict normalization"""
    if not word:
        return word
    # NFC first
    try:
        nfc = unicodedata.normalize('NFC', word)
    except Exception:
        nfc = word
    # Then Amharic strict
    try:
        return normalizer.normalize_word_strict(nfc)
    except Exception:
        return nfc

@dataclass
class ConfidencePolicy:
    """Post-model policy layer: uses Tesseract confidence AFTER the model has
    predicted, so the LightGBM core stays 100% confidence-free by design.

    The model produces a calibrated suspicion score; the gates turn it into
    pass/gray/flag. Those gates are a single global threshold pair, and the
    shipped isotonic calibrator emits only 59 distinct values, so the FLAG
    band is thin: at 0.10/0.65 only 2.65% of errors are hard-flagged, at 65.7%
    FLAG precision. Lowering the flag gate to widen that band trades precision
    away fast (gate 0.50 buys 344 more flagged errors but drops precision to
    49.9%).

    Confidence is an *independent* signal, so combining it beats either alone.
    Measured on the 100 held-out pages (62,385 aligned tokens, 4,267 errors):

      rule                  fires   errors   clean   precision of what moves
      A gray->pass            202        3     199   98.5% (correctly clean)
      B gray->flag            523      471      52   90.1%
      C pass->gray            899       52     847    5.8%  <- disabled by default

    Rule B is the win: FLAG coverage 2.65% -> 13.69% at 84.0% FLAG precision,
    and it costs the corrector nothing, because those words were already GRAY
    and already being processed. In the parent pipeline FLAG only removes the
    protective extra_margin, so B routes errors into more aggressive correction.

    Rule C is why `demote_pass_conf` defaults to None. It is the only rule that
    increases the corrector's workload on clean text: +648 clean words for +49
    errors, and the corrector damages ~4.06% of the clean tokens it touches
    (Stage 5 benchmark), so it pays ~26 damaged words to catch 49. It also
    *lowers* attention precision 30.05% -> 28.90%. It still lifts the
    `recall - FF` score, because that score counts GRAY as caught while
    ignoring clean words pushed into GRAY. Opt in deliberately, with a
    measurement, not by default.

    Thresholds are model-specific and must ride with the model artifact:
    `loc_f10` and `m7_f1000` are already optimal at flag gate 0.65 while the
    shipped conffree model wants 0.85. A payload may store them under the
    `confidence_policy` key; an explicitly passed policy wins.

    All confidences are on the 0-1 scale (`verdict()` rescales Tesseract's
    0-100 conf.tsv values onto `sig.ocr_conf` before this runs). Tokens with no
    confidence are never touched — the layer is a no-op when conf is missing,
    which keeps the detector's behaviour unchanged for callers that do not
    supply it.
    """

    # Rule A: gray -> pass. The lexicon already vouches for the word, there is
    # no confusion-matrix candidate within one flip, and Tesseract is confident
    # it read the glyphs correctly. Nothing left to review.
    promote_pass_conf: float = 0.82

    # Rule B: gray -> flag. Tesseract is unsure AND the model is not merely
    # unsure but leans toward error. Both signals agreeing is what makes this
    # 90% precise rather than 50%.
    flag_below_conf: float = 0.65
    flag_min_suspicion: float = 0.35

    # Rule C: pass -> gray. Disabled by default; see the class docstring.
    demote_pass_conf: Optional[float] = None

    @classmethod
    def tuned(cls) -> "ConfidencePolicy":
        """The recommended operating point: A + widened B, no C.

        recall 92.78%, flag-only 13.69%, FF 0.191%, FLAG precision 84.0%,
        attention precision 30.50% (vs 30.05% baseline) on the 100 held-out
        pages. Validated with page-level 5-fold CV: thresholds re-selected on
        unseen pages, so the gain is not fitted to the pages being reported.
        """
        return cls()

    @classmethod
    def from_payload(cls, obj) -> Optional["ConfidencePolicy"]:
        """Build a policy from a model-payload value, or return None.

        Accepts a ConfidencePolicy, a Mapping, or None. Unknown keys are
        rejected rather than silently ignored, so a typo in a retrained
        pickle surfaces instead of quietly disabling a rule.
        """
        if obj is None:
            return None
        if isinstance(obj, ConfidencePolicy):
            return obj
        if isinstance(obj, Mapping):
            known = {f for f in ("promote_pass_conf", "flag_below_conf",
                                 "flag_min_suspicion", "demote_pass_conf")}
            unknown = set(obj) - known
            if unknown:
                raise ValueError(
                    f"confidence_policy has unknown keys: {sorted(unknown)}; "
                    f"expected a subset of {sorted(known)}")
            return cls(**obj)
        raise TypeError(
            f"confidence_policy must be a ConfidencePolicy or a Mapping, "
            f"got {type(obj).__name__}")

    def review(self, label: str, suspicion: float, sig) -> Tuple[str, str]:
        """Apply the policy to one verdict. Returns (label, note).

        `note` is empty when the policy did not move the token, so callers can
        append it to the verdict reason for auditability.
        """
        conf = getattr(sig, "ocr_conf", None)
        if conf is None:
            return label, ""

        if label == "gray":
            if (sig.in_lexicon and not sig.one_flip_words
                    and conf >= self.promote_pass_conf):
                return "pass", (f" -> PASS [conf {conf:.3f} >= {self.promote_pass_conf:.2f},"
                                f" in lexicon, no one-flip]")
            if (conf < self.flag_below_conf
                    and suspicion >= self.flag_min_suspicion):
                return "flag", (f" -> FLAG [conf {conf:.3f} < {self.flag_below_conf:.2f},"
                                f" s {suspicion:.3f} >= {self.flag_min_suspicion:.2f}]")
        elif label == "pass":
            if self.demote_pass_conf is not None and conf < self.demote_pass_conf:
                return "gray", (f" -> GRAY [conf {conf:.3f} < {self.demote_pass_conf:.2f}]")

        return label, ""


class DetectionEngine:
    """Authoritative LightGBM detector — sole source of truth.

    v3 changes per roadmap:
    - Canonical boundary: NFC + normalize_word_strict before any lookup
    - NaN for missing KenLM / doc_freq / ocr_conf (LightGBM handles NaN)
    - No manual safety floors — trusts calibrated P(error)
    - Isotonic calibration support
    - doc_freq optional: NaN for isolated token checks
    - Rule fallback only when LGBM unavailable
    """

    def __init__(
        self,
        dict_vocab_path: Optional[str] = None,
        confusion_matrix_path: Optional[str] = None,
        char_lm_path: Optional[str] = None,
        lgbm_model_path: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        pass_below: float = 0.10,
        flag_above: float = 0.65,
        max_one_flip: int = 8,
        max_lexicon_types: Optional[int] = None,
        dict_min_freq: Optional[int] = None,
        symspell_max_edit_distance: int = 1,
        enable_symspell: bool = False,
        legacy_impute: Optional[bool] = None,
        gates_override_model: bool = False,
        confidence_policy: Optional[ConfidencePolicy] = None,
        verbose: bool = False,
    ):
        """pass_below / flag_above default to the README's gates (0.10 / 0.65);
        a model payload that carries its own thresholds wins unless
        gates_override_model=True.

        gates_override_model: the shipped payload stores pass_thresh /
        flag_thresh, and those have always taken precedence — which made the
        gates impossible to retune without rewriting the pickle. Set this to
        True to make the values you pass authoritative instead. It is opt-in
        because the parent corrector forwards its own concrete defaults
        (suspicion_pass_below=0.30, suspicion_flag_above=0.70) through the
        adapter; honouring those unconditionally would silently move the
        parent's Stage-1 operating point. Retune deliberately, then measure
        with scripts/tune_gates.py.

        confidence_policy: opt-in post-model layer that uses Tesseract
        confidence to move tokens between pass/gray/flag AFTER the model has
        predicted (see ConfidencePolicy). Deliberately a separate switch from
        gates_override_model, not folded into it: gates_override_model changes
        gate *precedence*, while this adds a *layer*, and conflating them would
        make it impossible to measure the policy at a chosen gate pair without
        also changing which gates are in force. Default None keeps the layer
        off, so the parent pipeline (which forwards its own gates through the
        adapter and never passes this) is unaffected. A payload that stores a
        `confidence_policy` key supplies the tuned thresholds for the model it
        was measured with; an explicitly passed policy wins, matching the
        precedence philosophy above.

        dict_vocab_path / dict_min_freq / max_lexicon_types apply to a
        caller-supplied lexicon. A caller-supplied path that does not exist
        raises FileNotFoundError rather than degrading to the minimal
        vocabulary. When the built-in lexicon is used and the caller passes
        None, the model payload decides, falling back to the documented
        main_i7 freq>=500 (54,057 types) — not the old ad-hoc 50,000-type
        truncation.

        legacy_impute: None (default) fills missing values the way the loaded
        model was trained (only models older than the NaN schema need it).
        True/False forces it on/off.
        """
        self.normalizer = AmharicNormalizer()
        self.w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.pass_below = pass_below
        self.flag_above = flag_above
        self.max_one_flip = max_one_flip
        self.legacy_impute = legacy_impute
        self.gates_override_model = gates_override_model
        # An explicitly passed policy wins over one stored in the payload.
        self.confidence_policy = confidence_policy
        self.verbose = verbose
        self._stage1 = {"pass": 0, "gray": 0, "flag": 0}

        # Resolve paths
        caller_lexicon = dict_vocab_path is not None
        if dict_vocab_path is None:
            dict_vocab_path = _resolve_first_existing(CANDIDATE_LEXICON_PATHS)
        if confusion_matrix_path is None:
            confusion_matrix_path = _resolve_first_existing(CANDIDATE_CONFUSION_PATHS)
            if confusion_matrix_path is None:
                confusion_matrix_path = os.path.join(_HERE, "Model", "confusion_matrix.json")
        if char_lm_path is None:
            char_lm_path = _resolve_first_existing(CANDIDATE_CHAR_LM_PATHS)
        if lgbm_model_path is None:
            lgbm_model_path = _resolve_first_existing(CANDIDATE_MODEL_PATHS)

        # Load LGBM + calibrator
        self.lgbm_model = None
        self.lgbm_features: List[str] = []
        self.calibrator = None
        self._legacy_model = False  # True for schema-1 pickles (pre features.py)
        self.model_path = None
        self._payload = {}
        self._feature_warning = ""

        target_pkl = lgbm_model_path
        if target_pkl and not os.path.exists(target_pkl):
            target_pkl = _resolve_first_existing(CANDIDATE_MODEL_PATHS)

        if target_pkl and joblib is not None and os.path.exists(target_pkl):
            if self.verbose:
                print(f"Loading LightGBM Model: {target_pkl}", flush=True)
            try:
                payload = joblib.load(target_pkl)
                self.model_path = target_pkl
                self._payload = payload
                self.lgbm_model = payload.get("model")
                self.lgbm_features = list(payload.get("features", []))
                if not gates_override_model:
                    self.pass_below = payload.get("pass_thresh", self.pass_below)
                    self.flag_above = payload.get("flag_thresh", self.flag_above)
                self.calibrator = payload.get("calibrator") or payload.get("isotonic") or payload.get("calibration")
                if self.confidence_policy is None:
                    self.confidence_policy = ConfidencePolicy.from_payload(
                        payload.get("confidence_policy"))
                schema = payload.get("feature_schema_version", payload.get("feature_schema", 1))
                self._legacy_model = int(schema) < NAN_SCHEMA_VERSION
                if self.verbose:
                    cal_info = "with isotonic calibration" if self.calibrator is not None else "no calibration"
                    pol_info = ""
                    if self.confidence_policy is not None:
                        p = self.confidence_policy
                        pol_info = (f" +confidence policy (promote>={p.promote_pass_conf},"
                                    f" flag<{p.flag_below_conf}&s>={p.flag_min_suspicion}"
                                    f"{'' if p.demote_pass_conf is None else f', demote<{p.demote_pass_conf}'})")
                    print(f"  LightGBM loaded ({len(self.lgbm_features)} feats, gates {self.pass_below}/{self.flag_above}) {cal_info}{pol_info}", flush=True)
            except Exception as e:
                self.lgbm_model = None
                self.lgbm_features = []
                self.calibrator = None
                if self.verbose:
                    print(f"  [WARN] Failed to load LGBM: {e}", flush=True)
        elif self.verbose:
            print(f"  [WARN] LGBM not found, rule fallback active", flush=True)

        # Char LM optional
        self.char_lm = None
        if char_lm_path and os.path.exists(char_lm_path):
            klm = _load_kenlm()
            if klm:
                try:
                    self.char_lm = klm.Model(char_lm_path)
                    if self.verbose:
                        print(f"Loading Char KenLM: {char_lm_path}", flush=True)
                except Exception as e:
                    if self.verbose:
                        print(f"  [WARN] KenLM load failed: {e}", flush=True)

        # Confusion matrix
        if confusion_matrix_path and os.path.exists(confusion_matrix_path):
            if self.verbose:
                print(f"Loading confusion: {confusion_matrix_path}", flush=True)
            self._confusion = ConfusionMatrix(confusion_matrix_path)
            self.reverse_sub_map = self._confusion.reverse_sub_map
        else:
            if self.verbose:
                print(f"  [WARN] Confusion not found, empty map", flush=True)
            self._confusion = None
            self.reverse_sub_map = {}

        # Schema / feature-name guard: a payload that asks for a column this
        # code cannot compute would otherwise be served with NaN or 0 silently.
        self._feature_warning = ""
        if self.lgbm_model is not None and self.lgbm_features:
            unknown = missing_features(self.lgbm_features)
            if unknown:
                self._feature_warning = (
                    "model expects features this code does not define: "
                    + ", ".join(unknown))
                raise FeatureError(f"{target_pkl}: {self._feature_warning}")
        if self._legacy_model:
            # Documented legacy behaviour (docs/lgbm_step1_changes.md §2.7):
            # a model without a feature version is older than the NaN schema,
            # so it is served with the fixed values it was trained under.
            self._feature_warning = (
                f"model has no feature schema version (pre-schema-{NAN_SCHEMA_VERSION}); "
                "missing values are imputed with the trained constants"
            )
            logger.warning("%s: %s", target_pkl, self._feature_warning)
            if self.verbose:
                print(f"  [WARN] {self._feature_warning}", flush=True)

        # Vocab. A built-in lexicon (dict_vocab_path=None) follows the model's
        # own metadata; an explicit caller path follows the caller.
        self.vocab: Dict[str, int] = {}
        self.surface_form: Dict[str, str] = {}
        self.lexicon_info = {"path": dict_vocab_path, "min_freq": None,
                             "max_types": None, "types": 0, "caller_supplied": caller_lexicon}
        if caller_lexicon and not os.path.exists(dict_vocab_path):
            # An explicitly requested lexicon that is missing used to fall through
            # to the 8-word minimal fallback below. That silently invalidated
            # in_lexicon and log_corpus_freq — together ~80% of the shipped
            # model's gain — so every verdict came back wrong with no warning.
            # A typo is the common cause (the real file is main_i7_frq10.txt, not
            # main_i7_freq10.txt). Fail loudly, and show what is actually here.
            raise FileNotFoundError(
                f"dict_vocab_path={dict_vocab_path!r} does not exist. "
                "The lexicon supplies in_lexicon / log_corpus_freq, so refusing to "
                "fall back to the minimal vocabulary. In-repo candidates:\n  "
                + "\n  ".join(p for p in CANDIDATE_LEXICON_PATHS if os.path.exists(p))
                + "\nPass dict_vocab_path=None to auto-resolve one of those.")
        if dict_vocab_path and os.path.exists(dict_vocab_path):
            if caller_lexicon:
                resolved_min_freq = dict_min_freq
                resolved_max_types = max_lexicon_types
            else:
                resolved_min_freq = dict_min_freq
                if resolved_min_freq is None:
                    resolved_min_freq = self._payload.get("lexicon_min_freq",
                                                          DEFAULT_DICT_MIN_FREQ)
                resolved_max_types = max_lexicon_types
                if resolved_max_types is None:
                    resolved_max_types = self._payload.get("lexicon_max_types",
                                                           DEFAULT_MAX_LEXICON_TYPES)
            self.lexicon_info.update(min_freq=resolved_min_freq,
                                     max_types=resolved_max_types)
            if self.verbose:
                print(f"Loading lexicon: {dict_vocab_path}"
                      f" (min freq {resolved_min_freq}, cap {resolved_max_types})",
                      flush=True)
            try:
                self._load_vocabulary(dict_vocab_path, resolved_max_types, resolved_min_freq)
                self.lexicon_info["types"] = len(self.vocab)
                if self.verbose:
                    print(f"  {len(self.vocab):,} types", flush=True)
            except Exception as e:
                if self.verbose:
                    print(f"  [WARN] Lexicon load failed: {e}", flush=True)
        else:
            if self.verbose:
                print(f"  [WARN] Lexicon not found, minimal fallback", flush=True)
            for w in ["እና", "ነው", "ላይ", "እንደ", "አንድ", "ወደ", "ሰላም", "ቤት"]:
                nw = _canonical_form(w, self.normalizer)
                self.vocab[nw] = 1000
                self.surface_form[nw] = w

        # SymSpell — optional, disabled by default for performance
        # n_neighbors feature is 0 in conf-free models, so SymSpell not needed
        # Enable only if you need neighbourhood density
        self.symspell = None
        self._VERBOSITY_TOP = 2
        if enable_symspell:
            try:
                _SS, _V = _load_symspell()
                self._VERBOSITY_TOP = _V.CLOSEST if hasattr(_V, 'CLOSEST') else 2
                self.symspell = _SS(max_dictionary_edit_distance=symspell_max_edit_distance, prefix_length=7)
                if self.verbose:
                    print(f"  Building SymSpell for {len(self.vocab):,} types...", flush=True)
                for norm_w, freq in self.vocab.items():
                    try:
                        self.symspell.create_dictionary_entry(norm_w, freq)
                    except Exception:
                        continue
                if self.verbose:
                    print("  SymSpell ready.", flush=True)
            except Exception as e:
                if self.verbose:
                    print(f"  [WARN] SymSpell build failed: {e}", flush=True)
                self.symspell = None
        else:
            if self.verbose:
                print(f"  SymSpell disabled (enable_symspell=False) for speed — n_neighbors=0", flush=True)

        # Shared feature context — features.py is the single feature definition
        # used by training (Model/train_LightGBM.py) and inference (here).
        self._rebuild_context()

    def _rebuild_context(self) -> None:
        """(Re)build the FeatureContext from the current resources.

        Called at init and again by adopt_lexicon(), so the feature code always
        reads the resources the engine really has in hand.
        """
        self._ctx = FeatureContext(
            vocab=self.vocab,
            norm=self._norm,
            reverse_sub_map=self.reverse_sub_map,
            max_one_flip=self.max_one_flip,
            char_lm=self.char_lm,
            symspell=self.symspell,
            verbosity_top=getattr(self, "_VERBOSITY_TOP", 2),
        )

    def adopt_lexicon(self, vocab: Mapping[str, int],
                      surface_form: Optional[Mapping[str, str]] = None,
                      norm: Optional[Callable[[str], str]] = None) -> int:
        """Take over a lexicon the caller has already loaded.

        README: the parent's adapter reuses its own vocabulary instead of
        reading the dictionary a second time (Model/main_i7_frq10.txt is 15MB).

        The model's own frequency filter still applies first, because the
        features must stay in the distribution the model was trained on
        (main_i7 freq>=500). A lexicon that carries no usable frequencies — a
        bare word list, a test fixture — is honoured exactly as the caller
        loaded it, since there is nothing to filter on.

        Returns the number of types now in use.
        """
        import heapq

        vocab = dict(vocab or {})
        min_freq = self.lexicon_info.get("min_freq")
        max_types = self.lexicon_info.get("max_types")
        filtered = vocab
        if min_freq and min_freq > 1:
            filtered = {w: f for w, f in vocab.items() if f >= min_freq}
            if not filtered and vocab:
                filtered = vocab
        if max_types and len(filtered) > max_types:
            keep = heapq.nlargest(max_types, filtered, key=filtered.__getitem__)
            filtered = {w: filtered[w] for w in keep}

        self.vocab = filtered
        forms = surface_form if surface_form is not None else self.surface_form
        self.surface_form = {k: v for k, v in forms.items() if k in filtered}
        if norm is not None:
            self._norm = norm
        self.lexicon_info["reused_from_caller"] = True
        self.lexicon_info["types"] = len(self.vocab)
        self._rebuild_context()
        return len(self.vocab)

    def _load_vocabulary(self, path: str, max_types: Optional[int], min_freq: Optional[int]):
        import heapq
        # Detect raw text vs lexicon
        is_raw = False
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as probe:
                first = probe.readline()
                if len(first.strip().split()) > 5:
                    is_raw = True
        except Exception:
            pass

        if is_raw:
            from collections import Counter
            counter = Counter()
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    for tok in line.strip().split():
                        tok = tok.strip()
                        if not tok:
                            continue
                        counter[tok] += 1
            for w, freq in counter.items():
                nw = _canonical_form(w, self.normalizer)
                if not nw:
                    continue
                prev = self.vocab.get(nw, 0)
                if freq >= prev:
                    self.surface_form[nw] = w
                self.vocab[nw] = max(prev, freq)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    w = parts[0]
                    freq = 1
                    if len(parts) > 1 and parts[-1].isdigit():
                        try:
                            freq = int(parts[-1])
                        except ValueError:
                            freq = 1
                    nw = _canonical_form(w, self.normalizer)
                    if not nw:
                        continue
                    prev = self.vocab.get(nw, 0)
                    if freq >= prev:
                        self.surface_form[nw] = w
                    self.vocab[nw] = max(prev, freq)

        if min_freq and min_freq > 1:
            keep = [w for w, f in self.vocab.items() if f >= min_freq]
            self.vocab = {w: self.vocab[w] for w in keep}
            self.surface_form = {w: self.surface_form[w] for w in keep if w in self.surface_form}

        if max_types and len(self.vocab) > max_types:
            keep = heapq.nlargest(max_types, self.vocab, key=self.vocab.__getitem__)
            self.vocab = {w: self.vocab[w] for w in keep}
            self.surface_form = {w: self.surface_form[w] for w in keep if w in self.surface_form}

    def _norm(self, word: str) -> str:
        return _canonical_form(word, self.normalizer)

    def _cheap_signals(self, tok: str, context: Sequence[str] = ()) -> TokenSignals:
        """Signals from the shared feature code — one definition, one path."""
        return cheap_signals(tok, self._ctx)

    def _extract_lgbm_features(self, tok: str, sig: TokenSignals = None,
                               doc_freq: Optional[float] = None,
                               ocr_conf: Optional[float] = None) -> List[float]:
        """Feature vector for one token, in the loaded model's column order.

        Everything comes from error_detection.features — the same code path
        Model/train_LightGBM.py uses — so a training row and a serving row for
        the same word are identical (train/serve parity). Older models get
        their trained constants back for missing values.
        """
        if not self.lgbm_features:
            return []
        if self._legacy_model and (self.legacy_impute is None or self.legacy_impute):
            impute = True
        else:
            impute = bool(self.legacy_impute)
        return extract_vector(tok, self._ctx, self.lgbm_features,
                              doc_freq=doc_freq, ocr_conf=ocr_conf,
                              legacy_impute=impute)

    # Public API for tests
    def signals(self, tok: str, context: Sequence[str] = ()) -> TokenSignals:
        return self._cheap_signals(tok, context)

    def score(self, sig: TokenSignals) -> float:
        return combine_signals(sig, self.w)

    def detection_stats(self):
        return {"stage1": dict(self._stage1)}

    def reset_counters(self):
        self._stage1 = {"pass": 0, "gray": 0, "flag": 0}

    def verdict_batch(self, tokens: Sequence[str],
                      context_fn: Optional[Callable] = None,
                      ocr_confs: Optional[Sequence[Optional[float]]] = None,
                      doc_freqs: Optional[Sequence[Optional[float]]] = None) -> List[Verdict]:
        results = []
        for i, tok in enumerate(tokens):
            ctx = context_fn(i) if context_fn else []
            conf = ocr_confs[i] if ocr_confs and i < len(ocr_confs) else None
            df = doc_freqs[i] if doc_freqs and i < len(doc_freqs) else None
            results.append(self.verdict(tok, ctx, ocr_conf=conf, doc_freq=df))
        return results

    def verdict(
        self,
        tok: str,
        context: Sequence[str] = (),
        ocr_conf: Optional[float] = None,
        doc_freq: Optional[float] = None,
    ) -> Verdict:
        """Authoritative verdict: calibrated P(error) -> pass/gray/flag, no manual floors."""
        sig = self._cheap_signals(tok, context)
        if ocr_conf is not None:
            # Store normalized 0-1
            try:
                if ocr_conf > 1:
                    sig.ocr_conf = max(0.0, min(1.0, ocr_conf / 100.0))
                else:
                    sig.ocr_conf = max(0.0, min(1.0, ocr_conf))
            except Exception:
                sig.ocr_conf = None

        # PATH A: LightGBM authoritative
        if self.lgbm_model is not None and self.lgbm_features:
            feats = self._extract_lgbm_features(tok, sig, doc_freq=doc_freq, ocr_conf=ocr_conf)
            try:
                raw_prob = float(self.lgbm_model.predict([feats])[0])
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] LGBM predict failed: {e}, fallback to rule", flush=True)
                raw_prob = combine_signals(sig, self.w)

            # Isotonic calibration if available
            prob = raw_prob
            if self.calibrator is not None:
                try:
                    # IsotonicRegression expects array
                    prob = float(self.calibrator.predict([raw_prob])[0])
                    # Clip to [0,1]
                    prob = max(0.0, min(1.0, prob))
                except Exception:
                    prob = raw_prob

            s = prob

            if s >= self.flag_above:
                label = "flag"
                reason = f"LGBM={s:.3f} (FLAG)"
            elif s >= self.pass_below:
                label = "gray"
                reason = f"LGBM={s:.3f} (GRAY)"
            else:
                label = "pass"
                reason = f"LGBM={s:.3f} (PASS)"

            # No manual floors per roadmap — trust calibrated probabilities
            # Only keep minimal OOV->gray for backward compat if model says pass but OOV
            # But per roadmap we should NOT override. We keep no overrides.
            #
            # The optional confidence policy is NOT such an override: it runs
            # after the model has predicted and never feeds back into the
            # features, so the LightGBM core stays confidence-free. It only
            # re-bands the gate output using an independent signal. Off unless
            # a policy was passed or the payload carries one.
            if self.confidence_policy is not None:
                label, note = self.confidence_policy.review(label, s, sig)
                if note:
                    reason += note

            if label == "pass":
                self._stage1["pass"] += 1
            elif label == "flag":
                self._stage1["flag"] += 1
            else:
                self._stage1["gray"] += 1

            return Verdict(s, label, sig, reason)

        # PATH B: Rule fallback (deprecated, lightweight)
        s = combine_signals(sig, self.w)
        if (sig.in_lexicon and not sig.one_flip_words
                and not sig.mixed_script and sig.rare_glyph_ratio < 0.35
                and (sig.ocr_conf is None or sig.ocr_conf >= 0.70)
                and s < self.pass_below):
            self._stage1["pass"] += 1
            return Verdict(s, "pass", sig, "clean lexicon word (fallback)")

        if not sig.in_lexicon and sig.one_flip_words and s >= 0.50:
            self._stage1["flag"] += 1
            return Verdict(s, "flag", sig, f"OOV, one flip reaches {sig.one_flip_words[0]!r} (fallback)")

        if s >= self.flag_above:
            self._stage1["flag"] += 1
            return Verdict(s, "flag", sig, "flag (fallback)")
        if s >= self.pass_below or not sig.in_lexicon:
            self._stage1["gray"] += 1
            return Verdict(s, "gray", sig, "gray (fallback)")

        self._stage1["pass"] += 1
        return Verdict(s, "pass", sig, "pass (fallback)")
