"""Isolate the normalization-lookup bug from the rare-glyph feature.

Groups clean (is_error=0) aligned words in analysis.tsv by:
  - h_family: surface changes under normalization, but no RARE_GLYPHS char
  - rare:     contains a RARE_GLYPHS char (legit feature, learned by the model)
  - plain:    neither
and compares verdicts, split by the analysis in_lexicon column (which uses the CORRECT
normalized lookup). If the bug drives verdicts, h_family + in_lexicon=1 should be
mostly gray/flag, unlike plain + in_lexicon=1.
"""
import csv, sys, os, collections
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
ROOT = os.path.join(REPO, 'OCR_error_detection')
sys.path.insert(0, REPO)
sys.path.insert(0, ROOT)
from normalizer import AmharicNormalizer
from OCR_error_detection.detector import RARE_GLYPHS

rows = list(csv.DictReader(open(os.path.join(ROOT, 'Evaluation/analysis.tsv'), encoding='utf-8'), delimiter='\t'))
aligned = [r for r in rows if r['is_error'] in ('0', '1')]
norm = AmharicNormalizer()

groups = collections.defaultdict(collections.Counter)
for r in aligned:
    w = r['ocr_word']
    changed = norm.normalize_word_strict(w) != w
    rare = any(ch in RARE_GLYPHS for ch in w)
    if rare:
        g = 'rare'
    elif changed:
        g = 'h_family'
    else:
        g = 'plain'
    key = (g, 'err' if r['is_error'] == '1' else 'clean', 'inlex=%s' % r['in_lexicon'])
    groups[key][r['detector_verdict']] += 1

print('%-10s %-6s %-9s  %s' % ('group', 'truth', 'lexicon', 'verdicts (pass/gray/flag)  n   pct_not_pass'))
for key in sorted(groups):
    c = groups[key]
    n = sum(c.values())
    notp = 100.0 * (n - c.get('pass', 0)) / n
    print('%-10s %-6s %-9s  %6d/%6d/%6d  n=%6d  %.1f%%' % (key[0], key[1], key[2],
          c.get('pass', 0), c.get('gray', 0), c.get('flag', 0), n, notp))

# Examples of clean h_family words that are in-lexicon by the correct lookup but OOV-looking to the feature
ex = []
for r in aligned:
    w = r['ocr_word']
    if r['is_error'] == '0' and r['in_lexicon'] == '1' and norm.normalize_word_strict(w) != w and not any(ch in RARE_GLYPHS for ch in w):
        ex.append((w, r['detector_verdict'], r['detector_score']))
print('\nexamples (clean, in-lexicon, H-family only):', ex[:12])
