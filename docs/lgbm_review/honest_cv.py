"""Honest page-grouped CV on the 100 eval pages using features that are available at eval time
(no detector score, which is in-sample). Compares simple feature sets to see how much
OCR confidence and the lexicon flag contribute. Not a replacement for the production
training run (which needs the master dictionary and KenLM model, both absent here)."""
import csv, os, sys, collections
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
ROOT = os.path.join(REPO, 'OCR_error_detection')
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, REPO)
sys.path.insert(0, ROOT)
from OCR_error_detection.detector import RARE_GLYPHS

rows = list(csv.DictReader(open(os.path.join(ROOT, 'Evaluation/analysis.tsv'), encoding='utf-8'), delimiter='\t'))
aligned = [r for r in rows if r['is_error'] in ('0', '1')]

page_counts = collections.defaultdict(collections.Counter)
for r in aligned:
    page_counts[r['page']][r['ocr_word']] += 1

X_rows, y, groups = [], [], []
for r in aligned:
    w = r['ocr_word']
    conf = float(r['tesseract_conf']) if r['tesseract_conf'] else 0.0
    f = {
        'in_lexicon': float(r['in_lexicon']),
        'conf': conf / 100.0,
        'conf_missing': 0.0 if r['tesseract_conf'] else 1.0,
        'one_flip': float(int(r['one_flip']) > 0),
        'token_len': float(len(w)),
        'rare_glyph_ratio': sum(ch in RARE_GLYPHS for ch in w) / max(len(w), 1),
        'doc_freq': float(page_counts[r['page']][w]),
    }
    X_rows.append(f)
    y.append(int(r['is_error']))
    groups.append(r['page'])
y = np.array(y); groups = np.array(groups)
names_all = list(X_rows[0].keys())

def matrix(names):
    return np.array([[f[n] for n in names] for f in X_rows], dtype=float)

def cv(names, model_kind):
    X = matrix(names)
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        if model_kind == 'lr':
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            m = LogisticRegression(max_iter=1000, class_weight='balanced')
            m.fit((X[tr] - mu) / sd, y[tr])
            oof[va] = m.predict_proba((X[va] - mu) / sd)[:, 1]
        else:
            m = lgb.train({'objective': 'binary', 'learning_rate': 0.05, 'num_leaves': 31,
                           'min_child_samples': 30, 'verbose': -1, 'seed': 42},
                          lgb.Dataset(X[tr], label=y[tr]), num_boost_round=300)
            oof[va] = m.predict(X[va])
    return roc_auc_score(y, oof), average_precision_score(y, oof)

print('rows', len(y), 'error rate %.4f' % y.mean())
sets = {
    'lexicon only': ['in_lexicon'],
    'conf only': ['conf'],
    'lexicon + conf': ['in_lexicon', 'conf', 'conf_missing'],
    'lexicon + conf + surface': ['in_lexicon', 'conf', 'conf_missing', 'token_len', 'rare_glyph_ratio'],
    'lexicon + conf + surface + doc_freq': ['in_lexicon', 'conf', 'conf_missing', 'token_len', 'rare_glyph_ratio', 'doc_freq'],
    'same without conf (LGBM)': ['in_lexicon', 'token_len', 'rare_glyph_ratio', 'doc_freq', 'one_flip'],
    'all non-score features (LGBM)': names_all,
}
print('%-38s %-6s %-8s %-8s' % ('feature set', 'model', 'AUC', 'AP'))
for label, names in sets.items():
    for kind in ('lr', 'lgbm'):
        auc, ap = cv(names, kind)
        print('%-38s %-6s %.4f   %.4f' % (label, kind, auc, ap))
