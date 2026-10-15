"""Check: (a) one-flip signal direction on the eval data, (b) tesseract_conf distribution in the corpus."""
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
aligned = [r for r in rows if r['is_error'] in ('0', '1')]

print('== one-flip hits vs error rate (eval, in-sample labels) ==')
tab = collections.defaultdict(lambda: [0, 0])  # key -> [n, errors]
for r in aligned:
    k = (r['in_lexicon'], 'flip>0' if int(r['one_flip']) > 0 else 'flip=0')
    tab[k][0] += 1
    tab[k][1] += int(r['is_error'])
for k in sorted(tab):
    n, e = tab[k]
    print('in_lexicon=%s %-7s n=%6d  error_rate=%.4f' % (k[0], k[1], n, e / max(n, 1)))

print('\n== tesseract_conf distribution in harmonized corpus (first 300k rows) ==')
H = os.path.join(DETECTOR, 'Model', 'harmonized_corpus_jiwer.tsv')
vals = collections.Counter()
errs = collections.defaultdict(lambda: [0, 0])
n = 0
with open(H, encoding='utf-8') as fh:
    rd = csv.DictReader(fh, delimiter='\t')
    for r in rd:
        n += 1
        if n > 300000:
            break
        c = r['tesseract_conf']
        vals[c] += 1
        errs[c][0] += 1
        errs[c][1] += int(r['is_error'])
print('distinct conf values:', len(vals))
print('top 8 values:', vals.most_common(8))
print('rows at 75.0 or 0 or -1:', sum(vals[k] for k in ('75.0', '0.0', '-1.0', '-1')), 'of', n)
for c, _ in vals.most_common(5):
    print('  conf=%s  n=%d  error_rate=%.4f' % (c, errs[c][0], errs[c][1] / errs[c][0]))
