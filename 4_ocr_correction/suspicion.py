"""
Stage 1: error detection — DEPRECATED per roadmap, kept as lightweight fallback.

Per architectural decision: LightGBM DetectionEngine is authoritative.
This rule-based SuspicionScorer is retained only when LGBM/KenLM unavailable.

The corrector currently generates candidates and runs the LM over EVERY token.
At an 8% base rate that is ~92% wasted work, and worse, it gives the system
~92% opportunities to damage text that was already right.

This module answers one question per token:

    is this token suspicious enough to be worth touching?

and returns one of three verdicts:

    PASS  leave alone. No candidates, no LM, no morphology. Cannot be damaged.
    GRAY  worth a look, but only change it if the evidence is strong.
    FLAG  almost certainly wrong; correct it normally.

Signals, cheapest first
-----------------------
  1. in-lexicon + log frequency          O(1)
  2. surface: length, mixed script, rare glyph classes
  3. SymSpell neighbourhood density      (how many dict words within d=1)
  4. one-flip test                       (Ling #3)
  5. char-LM score                       (Claude's "secret weapon")
  6. word-LM contextual fit              (KenLM)
  7. morphology                          EXPENSIVE -- gray zone only
  8. OCR confidence                      free, but only if TSV was captured

The expensive signals are deliberately last. The segmenter measures at
~21.8 ms/call with a cache that barely hits (31 hits vs 3601 misses on a
100-page run), so using it as a first-pass filter would make Stage 1 slower
than the Stage 3 work it is supposed to avoid.

The one-flip test (Ling #3)
---------------------------
For a token that is NOT in the lexicon, ask: does replacing exactly one
character, by an optically confusable one, produce a real word?

    one flip yields a word  -> likely a mis-scanned real word -> FLAG
    no single flip works    -> likely a genuine OOV (name, loanword) -> PASS

That single rule is the main defense against the dominant false-positive mode
at this error rate: proper nouns being "corrected" into common words.

Calibration
-----------
Raw signal combination is a weighted rule with sensible defaults. Once you
have aligned (OCR, GT) pairs, `train_suspicion.py` fits a logistic regression
on the same features and writes the weights here. The rule is the starting
point, not the endpoint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Amharic-specific surface knowledge
# --------------------------------------------------------------------------- #

ETHIOPIC = range(0x1200, 0x1380)          # Ethiopic block
ETHIOPIC_EXT = range(0x2D80, 0x2DE0)      # Ethiopic Extended

# 6th-order ("sadis") and other sparse series are rare in native vocabulary.
# A token dense in these is either a genuine oddity or mis-scanned. From
# ocr_engine_faults.txt the sadis->geez shift is the single most frequent
# upstream fault, so this is a high-yield check.
RARE_GLYPH_CLASSES = (
    set("ሠሡሢሣሤሥሦሧ"),      # ሠ series
    set("ጰጱጲጳጴጵጶጷ"),      # ጰ series
    set("ፀፁፂፃፄፅፆፇ"),      # ፀ series
    set("ዐዑዒዓዔዕዖ"),        # ዐ series (the classic አ/ዐ confusion)
    set("ሷሹሺሻሼሽሾሿ"),      # 6th/7th-order ዘ/ሸ-type forms
)

# Curated Amharic confusion classes.
#
# These are the homoglyph families that native text distinguishes but a
# scanner does not. They are a FALLBACK: the learned confusion matrix takes
# precedence, and this only fills what that matrix is missing.
#
# It is missing a lot. confusion_matrix.json was built by
# 01_build_confusion_matrix.py against synthetic_eval/ocr_new_model -- the
# output of amh_epoch1_fast, which scores 26.65% CER against the stock
# engine's 2.05%. The cross-series pairs below (notably AIN/PHHARYA,
# the one both external reviews singled out) do not appear there at all,
# because that model's errors are not your engine's errors. Rebuild the
# matrix from ocr_pred and this fallback stops doing any work.
CONFUSION_CLASSES = (
    # AIN / PHHARYA -- the single most-cited Amharic OCR confusion, and the
    # one that is entirely absent from the learned matrix.
    ("\u12a0", "\u12d0", "\u12d3"),                 # አ ዐ ዓ
    # the three H's
    ("\u1200", "\u1210", "\u1280"),                 # ሀ ሐ ኀ
    # sibilants
    ("\u1230", "\u1220"),                            # ሰ ሠ
    # the TSA/PHA series
    ("\u1338", "\u1340"),                            # ጸ ፀ
    ("\u1338", "\u1330"),                            # ጸ ጰ
    # nasals
    ("\u1290", "\u1298"),                            # ነ ኘ
    # labials
    ("\u1260", "\u1268"),                            # በ ቨ
    # velar / pharyngeal slip
    ("\u12a8", "\u12a0"),                            # ከ አ
)


_CODEPOINTS = (  # verified against unicodedata, kept next to the classes
    "አ=U+12A0 ዐ=U+12D0 ዓ=U+12D3 | ሀ=U+1200 ሐ=U+1210 ኀ=U+1280 | "
    "ሰ=U+1230 ሠ=U+1220 | ጸ=U+1338 ፀ=U+1340 ጰ=U+1330 | "
    "ነ=U+1290 ኘ=U+1298 | በ=U+1260 ቨ=U+1268 | ከ=U+12A8"
)


def _confusion_alts(ch: str) -> list:
    """Optically confusable alternatives for `ch`, curated + learned."""
    out = []
    for cls in CONFUSION_CLASSES:
        if ch in cls:
            out.extend(c for c in cls if c != ch)
    return out


# Punctuation that Tesseract routinely confuses or drops.
ETHIOPIC_PUNCT = set("፡።፣፤፥፦፧፨")


def _is_ethiopic(ch: str) -> bool:
    o = ord(ch)
    return o in ETHIOPIC or o in ETHIOPIC_EXT


@dataclass
class TokenSignals:
    """Per-token evidence. `None` means 'not computed' (too expensive)."""
    token: str
    in_lexicon: bool = False
    log_freq: float = 0.0
    n_neighbors: int = 0              # dict words within edit distance 1
    one_flip_words: List[str] = field(default_factory=list)
    char_lm: Optional[float] = None   # avg log10 prob under the char LM
    lm_fit: Optional[float] = None    # logP(w|ctx) - logP_unigram(w)
    morph_ok: Optional[bool] = None   # segmenter verdict (expensive)
    ocr_conf: Optional[float] = None  # Tesseract word confidence, 0-100
    len_ratio: float = 1.0            # length / corpus mean (4.14)
    rare_glyph_ratio: float = 0.0
    mixed_script: bool = False
    has_digit: bool = False

    def as_vector(self) -> Dict[str, float]:
        """Numeric view for the scorer / later for logistic regression."""
        v = {
            "in_lexicon": 1.0 if self.in_lexicon else 0.0,
            "log_freq": self.log_freq,
            "n_neighbors": float(self.n_neighbors),
            "one_flip_hits": float(len(self.one_flip_words)),
            "len_ratio": self.len_ratio,
            "rare_glyph_ratio": self.rare_glyph_ratio,
            "mixed_script": 1.0 if self.mixed_script else 0.0,
            "has_digit": 1.0 if self.has_digit else 0.0,
        }
        if self.char_lm is not None:
            v["char_lm"] = self.char_lm
        if self.lm_fit is not None:
            v["lm_fit"] = self.lm_fit
        if self.morph_ok is not None:
            v["morph_ok"] = 1.0 if self.morph_ok else 0.0
        if self.ocr_conf is not None:
            v["ocr_conf"] = self.ocr_conf
        return v


@dataclass
class Verdict:
    suspicion: float
    label: str              # "pass" | "gray" | "flag"
    signals: TokenSignals
    reason: str = ""

    @property
    def should_correct(self) -> bool:
        return self.label in ("gray", "flag")


# --------------------------------------------------------------------------- #
# default weights -- replaced by train_suspicion.py once labels exist
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS: Dict[str, float] = {
    "bias": 0.10,
    # not being in the lexicon is the strongest single signal
    "in_lexicon": -2.20,
    # ...but a common word is safer than a rare one
    "log_freq": -0.14,
    # many close neighbours => plausible typo, but also plausible word
    "n_neighbors": 0.03,
    # the one-flip test: a reachable real word is strong evidence of error
    "one_flip_hits": 0.85,
    # phonotactically implausible => probably corrupted
    "char_lm": -0.55,
    # common word that does not fit its context => real-word error
    "lm_fit": -0.45,
    # morphologically well-formed => probably a legitimate rare form
    "morph_ok": -1.10,
    # Tesseract's own doubt, scaled to 0-1
    "ocr_conf": -1.60,
    # surface oddities
    "len_ratio": 0.35,
    "rare_glyph_ratio": 1.30,
    "mixed_script": 0.90,
    "has_digit": 0.30,
}


class SuspicionScorer:
    """Turns a token into a suspicion score and a pass/gray/flag verdict.

    Cheap by construction: signals 1-4 need no LM, no morphology, and no
    segmenter. Morphology is computed lazily and only for tokens that survived
    into the gray zone.
    """

    MEAN_TOKEN_LEN = 4.14

    def __init__(
        self,
        corrector,
        weights: Optional[Dict[str, float]] = None,
        pass_below: float = 0.30,
        flag_above: float = 0.70,
        use_morphology: bool = True,
        use_char_lm: bool = True,
        char_lm=None,
        max_one_flip: int = 8,
    ):
        self.c = corrector
        self.w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.pass_below = pass_below
        self.flag_above = flag_above
        self.use_morphology = use_morphology
        self.char_lm = char_lm
        self.use_char_lm = use_char_lm and char_lm is not None
        self.max_one_flip = max_one_flip
        self._morph_calls = 0

    # -- cheap signals ----------------------------------------------------- #
    def _surface(self, tok: str, sig: TokenSignals) -> None:
        n = len(tok)
        sig.len_ratio = n / self.MEAN_TOKEN_LEN
        eth = sum(1 for ch in tok if _is_ethiopic(ch))
        sig.mixed_script = 0 < eth < n
        sig.has_digit = any(ch.isdigit() for ch in tok)
        rare = sum(1 for ch in tok
                   if any(ch in cls for cls in RARE_GLYPH_CLASSES))
        sig.rare_glyph_ratio = rare / max(n, 1)

    def _lexical(self, tok: str, sig: TokenSignals) -> None:
        norm = self.c._norm(tok)
        freq = self.c.vocab.get(norm)
        if freq is not None:
            sig.in_lexicon = True
            sig.log_freq = math.log10(float(freq) + 1.0)
        # neighbourhood density
        try:
            sug = self.c.symspell.lookup(
                tok, self.c._VERBOSITY_TOP if hasattr(self.c, "_VERBOSITY_TOP")
                else 2,
                max_edit_distance=1, include_unknown=False)
            sig.n_neighbors = len(sug)
        except Exception:
            sig.n_neighbors = 0

    def _one_flip(self, tok: str, sig: TokenSignals) -> None:
        """Ling's one-flip test, restricted to OPTICALLY confusable chars.

        Using the confusion matrix rather than every character is what makes
        this precise: ሀ->ህ is a plausible scan error, ሀ->ጸ is not.
        """
        rev = getattr(self.c, "reverse_sub_map", {}) or {}
        hits: List[str] = []
        n = len(tok)
        tok_norm = self.c._norm(tok)
        for i, ch in enumerate(tok):
            learned = list(rev.get(ch) or [])
            if isinstance(learned, dict):
                learned = list(learned.items())
            # Curated classes come FIRST: they are few and high-precision,
            # whereas the learned matrix can list dozens of alternatives and
            # would otherwise push the curated ones past the cap and they
            # would never be tried at all.
            curated = [(c, 0.0) for c in _confusion_alts(ch)]
            seen = {a for a, _ in learned}
            alts = curated + [(a, c) for a, c in learned if a not in
                              {x for x, _ in curated}]
            if not alts:
                continue
            for alt, _cost in alts[:12]:
                cand = tok[:i] + alt + tok[i + 1:]
                if cand == tok:
                    continue
                # must go through the corrector's own normalizer: it collapses
                # the H-family (ሐ/ኀ -> ሀ), so comparing raw strings against
                # vocab keys silently misses words that ARE in the lexicon.
                cn = self.c._norm(cand)
                # A flip that normalizes back to the SAME key is a no-op, not
                # evidence. The normalizer collapses the H-family and the
                # AIN/PHHARYA series (ሐ/ኀ->ሀ, ዐ/ዓ->አ), so without this check
                # almost every clean word "flips" onto itself and looks
                # suspicious -- which is precisely the false-positive mode
                # this test exists to prevent.
                if cn == tok_norm:
                    continue
                if cn in self.c.vocab:
                    hits.append(cand)
                    break
            if len(hits) >= self.max_one_flip:
                break
        sig.one_flip_words = hits

    # -- expensive signals ------------------------------------------------- #
    def _char_lm_score(self, tok: str, sig: TokenSignals) -> None:
        if not self.use_char_lm or not tok:
            return
        # KenLM scores space-separated symbols. Scoring the raw token would
        # treat the whole word as one symbol, so the characters are spaced first.
        spaced = " ".join(tok)
        try:
            s = self.char_lm.score(spaced, bos=True, eos=True)
            sig.char_lm = s / max(len(tok), 1)
        except Exception:
            sig.char_lm = None

    def _lm_fit(self, tok: str, context: Sequence[str], sig: TokenSignals) -> None:
        lm = getattr(self.c, "lm", None)
        if lm is None:
            return
        try:
            ctx = " ".join(context[-2:])
            with_ctx = lm.score(f"{ctx} {tok}".strip(), bos=True, eos=False)
            without = lm.score(tok, bos=True, eos=False)
            base = lm.score(f"{ctx} <unk>".strip(), bos=True, eos=False)
            sig.lm_fit = with_ctx - max(without, base)
        except Exception:
            sig.lm_fit = None

    def _morphology(self, tok: str, sig: TokenSignals) -> None:
        if not self.use_morphology:
            return
        seg = getattr(self.c, "segmenter", None)
        if seg is None:
            return
        try:
            self._morph_calls += 1
            sig.morph_ok = bool(self.c.is_known_word(tok))
        except Exception:
            sig.morph_ok = None

    # -- combination ------------------------------------------------------- #
    def _combine(self, sig: TokenSignals) -> float:
        v = sig.as_vector()
        total = self.w.get("bias", 0.0)
        for k, coef in self.w.items():
            if k == "bias" or k not in v:
                continue
            total += coef * v[k]
        # logistic squash to (0, 1)
        if total >= 0:
            return 1.0 / (1.0 + math.exp(-total))
        e = math.exp(total)
        return e / (1.0 + e)

    # -- public ------------------------------------------------------------ #
    def signals(self, tok: str, context: Sequence[str] = ()) -> TokenSignals:
        sig = TokenSignals(token=tok)
        self._surface(tok, sig)
        self._lexical(tok, sig)
        self._one_flip(tok, sig)
        self._char_lm_score(tok, sig)
        return sig

    def enrich(self, sig: TokenSignals, tok: str,
               context: Sequence[str]) -> None:
        """Add the expensive signals. Only call this in the gray zone."""
        self._lm_fit(tok, context, sig)
        self._morphology(tok, sig)

    def score(self, sig: TokenSignals) -> float:
        return self._combine(sig)

    def verdict(self, tok: str, context: Sequence[str] = (),
                ocr_conf: Optional[float] = None) -> Verdict:
        sig = self.signals(tok, context)
        if ocr_conf is not None:
            sig.ocr_conf = max(0.0, min(1.0, ocr_conf / 100.0))

        s = self.score(sig)

        # Hard PASS: a lexicon word with no optical neighbour and nothing odd
        # about it. This is the ~90% of text we never want to touch.
        if (sig.in_lexicon and not sig.one_flip_words
                and not sig.mixed_script and sig.rare_glyph_ratio < 0.35
                and (sig.ocr_conf is None or sig.ocr_conf >= 0.75)
                and s < self.pass_below):
            return Verdict(s, "pass", sig, "clean lexicon word, no optical neighbour")

        # Hard FLAG: out of lexicon AND a single optical flip reaches a real
        # word. This is the signature of a mis-scanned word.
        if (not sig.in_lexicon and sig.one_flip_words
                and s >= self.flag_above):
            return Verdict(s, "flag", sig,
                           f"OOV, one flip reaches {sig.one_flip_words[0]!r}")

        # Gray: spend the expensive signals to decide.
        self.enrich(sig, tok, context)
        s = self.score(sig)
        if s >= self.flag_above:
            return Verdict(s, "flag", sig, "gray resolved to flag")
        if s >= self.pass_below:
            return Verdict(s, "gray", sig, "uncertain; correct only with a margin")
        return Verdict(s, "pass", sig, "gray resolved to pass")
