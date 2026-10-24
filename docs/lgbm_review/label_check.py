import csv, os, sys, collections
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
ROOT = os.path.join(REPO, 'OCR_error_detection')
sys.path.insert(0, ROOT)
from normalizer import AmharicNormalizer
N = AmharicNormalizer()
rows = list(csv.DictReader(open(os.path.join(ROOT, 'Evaluation/analysis.tsv'), encoding='utf-8'), delimiter='\t'))
err = [r for r in rows if r['is_error'] == '1']
same_norm = [r for r in err if N.normalize_word_strict(r['ocr_word']) == N.normalize_word_strict(r['gt_word'])]
print('substitution errors: %d; of which normalizer-equivalent to GT: %d (%.1f%%)' % (len(err), len(same_norm), 100*len(same_norm)/len(err)))
print('examples:', [(r['ocr_word'], r['gt_word'], r['detector_verdict']) for r in same_norm[:8]])
c = collections.Counter(r['detector_verdict'] for r in same_norm)
print('their verdicts:', dict(c))
