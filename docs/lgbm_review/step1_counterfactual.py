"""Step 1 checks on the saved Evaluation/analysis.tsv (in-sample, legacy model).

1. Punctuation signature: floors that fired although the cleaned token IS in the lexicon.
2. Raw-token check: for those rows, do the page's raw OCR tokens carry punctuation?
3. Counterfactual: recompute floor labels with the corrected lexicon flag. Model scores are
   unchanged (the model's own in_lexicon feature already used the cleaned token), so only
   floor behaviour moves. Validated first: rows with no floor tag must reproduce recorded labels.
"""
import csv, glob, os, re, collections

ROOT = '/home/user/OCR_corrector/4_ocr_correction/error_detection'
P = re.compile(r'[^\w\s]|\d')
rows = list(csv.DictReader(open(os.path.join(ROOT, 'Evaluation/analysis.tsv'), encoding='utf-8'), delimiter='\t'))
al = [r for r in rows if r['is_error'] in ('0', '1')]

# 1 + 2
raw = {os.path.basename(f).replace('.ocr.txt', ''): open(f, encoding='utf-8').read().split()
       for f in glob.glob(os.path.join(ROOT, 'Evaluation/OCRed/*.ocr.txt'))}
sel = [r for r in al if r['is_error'] == '0' and r['in_lexicon'] == '1' and 'OOV kept' in r['reason']]
c = collections.Counter()
for r in sel:
    matches = [t for t in raw[r['page']] if P.sub('', t).strip() == r['ocr_word']]
    if not matches:
        c['no raw match'] += 1
    elif all(t != r['ocr_word'] for t in matches):
        c['every raw form has punctuation/digits'] += 1
    else:
        c['ambiguous (bare form also on page)'] += 1
print('OOV-kept floor on in-lexicon correct words:', len(sel), dict(c))


def label(score, in_lex, conf):
    lab = 'flag' if score >= 0.6 else ('gray' if score >= 0.15 else 'pass')
    if not in_lex and conf is not None and conf < 0.50:
        return 'flag'
    if not in_lex and lab == 'pass':
        return 'gray'
    if conf is not None and conf < 0.60 and lab == 'pass':
        return 'gray'
    return lab


# 3 validation: rows without a floor tag must reproduce the recorded label
ok = bad = 0
for r in al:
    if any(t in r['reason'] for t in ('OOV kept', 'severe', 'low-conf floor')):
        continue
    conf = float(r['tesseract_conf']) if r['tesseract_conf'] else None
    if label(float(r['detector_score']), r['in_lexicon'] == '1', conf) == r['detector_verdict']:
        ok += 1
    else:
        bad += 1
print('validation, floor-free rows reproduced:', ok, 'mismatched:', bad)


def stats(pairs):
    n0 = collections.Counter(l for y, l in pairs if y == '0')
    n1 = collections.Counter(l for y, l in pairs if y == '1')
    ncor, nerr = sum(n0.values()), sum(n1.values())
    return {
        'clean pass/gray/flag': (n0['pass'], n0['gray'], n0['flag']),
        'clean not-pass %': round(100 * (n0['gray'] + n0['flag']) / ncor, 2),
        'clean flag FA %': round(100 * n0['flag'] / ncor, 2),
        'errors caught % (flag+gray)': round(100 * (n1['gray'] + n1['flag']) / nerr, 2),
        'errors pass/gray/flag': (n1['pass'], n1['gray'], n1['flag']),
    }


before = [(r['is_error'], r['detector_verdict']) for r in al]
after = [(r['is_error'], label(float(r['detector_score']), r['in_lexicon'] == '1',
                               float(r['tesseract_conf']) if r['tesseract_conf'] else None))
         for r in al]
print('before (recorded):', stats(before))
print('after punctuation fix (counterfactual):', stats(after))
