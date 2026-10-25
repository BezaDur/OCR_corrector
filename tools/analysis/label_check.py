import csv, os, sys, collections
# --- portable paths: tools/analysis/ is two levels below the repo root -------
import os as _os, sys as _sys
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
if _os.path.join(_REPO, "tools") not in _sys.path:
    _sys.path.insert(0, _os.path.join(_REPO, "tools"))
import stage_paths  # noqa: E402
stage_paths.register(_REPO)
DETECTOR = _os.path.join(_REPO, "4_ocr_correction", "error_detection")  # detector package
EVAL = _os.path.join(_REPO, "Evaluation")                              # shared page corpus
DATA = _os.path.join(_REPO, "1_data_preparation", "data")              # corpora
# ----------------------------------------------------------------------------

from error_detection.normalizer import AmharicNormalizer

N = AmharicNormalizer()
rows = list(csv.DictReader(open(os.path.join(EVAL, 'analysis.tsv'), encoding='utf-8'), delimiter='\t'))
err = [r for r in rows if r['is_error'] == '1']
same_norm = [r for r in err if N.normalize_word_strict(r['ocr_word']) == N.normalize_word_strict(r['gt_word'])]
print('substitution errors: %d; of which normalizer-equivalent to GT: %d (%.1f%%)' % (len(err), len(same_norm), 100*len(same_norm)/len(err)))
print('examples:', [(r['ocr_word'], r['gt_word'], r['detector_verdict']) for r in same_norm[:8]])
c = collections.Counter(r['detector_verdict'] for r in same_norm)
print('their verdicts:', dict(c))
