"""Label every Stage 1 verdict as clean / error / ambiguous against ground truth.

A logged verdict covers one token that Stage 1 saw. Its label comes from the
alignment of the raw token sequence to the ground truth sequence:
  clean      - the raw token is certainly correct (matched in every optimal alignment)
  error      - the raw token matches nothing in any optimal alignment (a true OCR error)
  ambiguous  - matched in some optimal alignments only

Tokens are normalised and reduced to their cores (punctuation and diacritics
removed) on both sides, the same way the corrector sees them.
"""
from typing import Dict, List

from ocr_metrics import align_tokens
from amharic_ocr_corrector import EthiopicPunctuationManager as P


def _cores(text: str) -> List[str]:
    out = []
    for line in text.splitlines():
        for tok in P.normalize_sentence_text(line).split():
            core = P.decompose_token(tok)[1]
            if core:
                out.append(core)
    return out


def label_verdicts(gold_text: str, raw_text: str, verdict_rows: List[Dict]) -> dict:
    """Return {"rows": [...], "status": ...}.

    rows: one per raw core token, in order, with `status` added.
    status: "ok" if verdicts line up with the raw tokens, else "mismatch"
    (then rows is empty and the page should not be scored).
    """
    raw_cores = _cores(raw_text)
    gold_cores = _cores(gold_text)
    if len(verdict_rows) != len(raw_cores) or any(
            v["word"] != c for v, c in zip(verdict_rows, raw_cores)):
        return {"rows": [], "status": "mismatch",
                "verdicts": len(verdict_rows), "raw_tokens": len(raw_cores)}

    alignment = align_tokens(raw_cores, gold_cores)
    certain, possible = alignment.certain_reference, alignment.possible_reference
    rows = []
    for i, v in enumerate(verdict_rows):
        if i in certain:
            status = "clean"
        elif i in possible:
            status = "ambiguous"
        else:
            status = "error"
        rows.append(dict(v, status=status))
    return {"rows": rows, "status": "ok"}


def threshold_sweep(rows: List[Dict], thresholds) -> List[Dict]:
    """Precision and recall of the flagged set `suspicion >= t`.

    precision: errors / (errors + clean) among flagged tokens (ambiguous excluded).
    error_recall: errors flagged / all errors.
    clean_flagged: clean tokens flagged (damage risk).
    """
    total_err = sum(1 for r in rows if r["status"] == "error")
    out = []
    for t in thresholds:
        flagged = [r for r in rows if r["suspicion"] >= t]
        err = sum(1 for r in flagged if r["status"] == "error")
        clean = sum(1 for r in flagged if r["status"] == "clean")
        amb = sum(1 for r in flagged if r["status"] == "ambiguous")
        judged = err + clean
        out.append(dict(
            threshold=t, flagged=len(flagged), errors=err, clean=clean, ambiguous=amb,
            precision=(err / judged) if judged else None,
            error_recall=(err / total_err) if total_err else None,
        ))
    return out
