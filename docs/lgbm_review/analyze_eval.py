"""Quantify the LightGBM detector on the saved Evaluation/analysis.tsv (in-sample caveat applies)."""
import csv, sys, os, collections
sys.path.insert(0, '/home/user/OCR_corrector')
sys.path.insert(0, '/home/user/OCR_corrector/4_ocr_correction/error_detection')
from normalizer import AmharicNormalizer

ROOT = '/home/user/OCR_corrector/4_ocr_correction/error_detection'
rows = list(csv.DictReader(open(os.path.join(ROOT, 'Evaluation/analysis.tsv'), encoding='utf-8'), delimiter='\t'))
# substitution errors + correct words only (exclude INSERTION=2 / DELETION=3 rows)
aligned = [r for r in rows if r['is_error'] in ('0', '1')]
print('aligned substitution/correct rows:', len(aligned))


def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    pos = sum(labels); neg = len(labels) - pos
    # rank-sum AUC (ties averaged)
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sum_pos = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (sum_pos - pos * (pos + 1) / 2) / (pos * neg)


def ap(scores, labels):
    # average precision (step-wise)
    order = sorted(range(len(scores)), key=lambda k: -scores[k])
    tp = 0; total_pos = sum(labels); s = 0.0
    for n, k in enumerate(order, 1):
        if labels[k]:
            tp += 1; s += tp / n
    return s / total_pos


y = [int(r['is_error']) for r in aligned]
det = [float(r['detector_score']) for r in aligned]
conf = [float(r['tesseract_conf']) if r['tesseract_conf'] != '' else 0.0 for r in aligned]
has_conf = [r['tesseract_conf'] != '' for r in aligned]
inlex = [int(r['in_lexicon']) for r in aligned]
one_flip = [int(r['one_flip']) for r in aligned]
print('conf missing rows:', sum(1 for h in has_conf if not h))

print('\n== Ranking quality (AUC / AP), higher = better ==')
print('LightGBM score           AUC=%.4f  AP=%.4f' % (auc(det, y), ap(det, y)))
print('-tesseract_conf (alone)  AUC=%.4f  AP=%.4f' % (auc([-c for c in conf], y), ap([-c for c in conf], y)))
print('-in_lexicon (alone)      AUC=%.4f  AP=%.4f' % (auc([-v for v in inlex], y), ap([-v for v in inlex], y)))
print('one_flip>0 (alone)       AUC=%.4f  AP=%.4f' % (auc(one_flip, y), ap(one_flip, y)))
print('base error rate: %.4f' % (sum(y) / len(y)))

print('\n== Verdict gates on detector score (in-sample!) ==')
# verdict as in detector.py (pre-floor) vs post-floor labels already in file
for label in ('pass', 'gray', 'flag'):
    e = sum(1 for r, yy in zip(aligned, y) if yy == 1 and r['detector_verdict'] == label)
    c = sum(1 for r, yy in zip(aligned, y) if yy == 0 and r['detector_verdict'] == label)
    print(f'{label:5}: errors={e:5} clean={c:6}')
n_err = sum(y); n_cor = len(y) - n_err
flag_tp = sum(1 for r, yy in zip(aligned, y) if yy == 1 and r['detector_verdict'] == 'flag')
flag_fp = sum(1 for r, yy in zip(aligned, y) if yy == 0 and r['detector_verdict'] == 'flag')
print('flag precision = %.3f   flag false-alarm rate on clean = %.4f' % (flag_tp / (flag_tp + flag_fp), flag_fp / n_cor))
caught = sum(1 for r, yy in zip(aligned, y) if yy == 1 and r['detector_verdict'] in ('flag', 'gray'))
print('caught (flag+gray) recall = %.4f' % (caught / n_err))
gray_clean = sum(1 for r, yy in zip(aligned, y) if yy == 0 and r['detector_verdict'] == 'gray')
print('clean words sent to gray = %d (%.1f%% of clean)' % (gray_clean, 100 * gray_clean / n_cor))

print('\n== Threshold sweep on the raw LightGBM score (ignores floors) ==')
print('thr   flagRec  flagFA(clean)  flagPrec  grayOrFlagRec  grayOrFlagFA')
for thr in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
    f_r = sum(1 for d, yy in zip(det, y) if yy == 1 and d >= thr) / n_err
    f_fa = sum(1 for d, yy in zip(det, y) if yy == 0 and d >= thr) / n_cor
    tp = sum(1 for d, yy in zip(det, y) if yy == 1 and d >= thr)
    fp = sum(1 for d, yy in zip(det, y) if yy == 0 and d >= thr)
    prec = tp / max(tp + fp, 1)
    g_r = sum(1 for d, yy in zip(det, y) if yy == 1 and d >= thr) / n_err
    print('%.2f  %.3f    %.4f        %.3f       %.3f          %.4f' % (thr, f_r, f_fa, prec, g_r, f_fa))

print('\n== Where do the post-floor labels come from? (reason categories) ==')
cats = collections.Counter()
for r, yy in zip(aligned, y):
    rs = r['reason']
    tag = 'severe_lowconf' if 'severe' in rs else ('lowconf_floor' if 'low-conf floor' in rs else ('OOV_kept' if 'OOV kept' in rs else 'model_only'))
    cats[(('err' if yy else 'clean'), tag, r['detector_verdict'])] += 1
for k, v in sorted(cats.items()):
    print(k, v)

print('\n== in_lexicon (analysis column, correct normalization) vs error rate ==')
for v in (0, 1):
    sub = [yy for yy, il in zip(y, inlex) if il == v]
    print('in_lexicon=%d  n=%6d  error_rate=%.4f' % (v, len(sub), sum(sub) / max(len(sub), 1)))
    vc = collections.Counter(r['detector_verdict'] for r, il in zip(aligned, inlex) if il == v)
    print('   verdicts:', dict(vc))

print('\n== Normalization-affected words (surface != normalized key) ==')
norm = AmharicNormalizer()
affected = [norm.normalize_word_strict(r['ocr_word']) != r['ocr_word'] for r in aligned]
for flag in (False, True):
    idx = [i for i, a in enumerate(affected) if a == flag]
    ncor = sum(1 for i in idx if y[i] == 0)
    nerr = sum(1 for i in idx if y[i] == 1)
    fa = sum(1 for i in idx if y[i] == 0 and aligned[i]['detector_verdict'] != 'pass')
    print('affected=%-5s n=%6d  clean=%6d  err=%5d  clean_not_pass=%5d (%.1f%%)  lexicon-OOV_rate_among_clean=%.3f' % (
        flag, len(idx), ncor, nerr, fa, 100 * fa / max(ncor, 1),
        sum(1 for i in idx if y[i] == 0 and inlex[i] == 0) / max(ncor, 1)))
