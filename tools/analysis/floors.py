import csv, os, collections
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


rows = list(csv.DictReader(open(os.path.join(EVAL, 'analysis.tsv'), encoding='utf-8'), delimiter='\t'))
al = [r for r in rows if r['is_error'] in ('0', '1')]
c = collections.Counter()
for r in al:
    s = float(r['detector_score'])
    if 'severe' in r['reason']:
        c[('severe_floor', r['is_error'], 'model_pass' if s < 0.15 else ('model_gray' if s < 0.6 else 'model_flag'))] += 1
    if 'OOV kept' in r['reason']:
        c[('oov_floor', r['is_error'], 'model_pass' if s < 0.15 else 'other')] += 1
    if 'low-conf floor' in r['reason']:
        c[('lowconf_floor', r['is_error'], 'model_pass' if s < 0.15 else 'other')] += 1
for k, v in sorted(c.items()):
    print(k, v)
sev = [r for r in al if 'severe' in r['reason']]
print('severe floor: flags=%d, errors among them=%d => precision=%.3f' % (len(sev), sum(r['is_error']=='1' for r in sev), sum(r['is_error']=='1' for r in sev)/len(sev)))
