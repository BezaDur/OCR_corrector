import sys, os, io, contextlib, tempfile, time
ROOT = '/home/user/OCR_corrector'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'OCR_error_detection'))
from OCR_error_detection.detector import DetectionEngine


def mk(words):
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'w.txt')
    with open(p, 'w', encoding='utf-8') as fh:
        for w, f in words:
            fh.write(f'{w} {f}\n')
    with contextlib.redirect_stdout(io.StringIO()):
        e = DetectionEngine(dict_vocab_path=p,
                            confusion_matrix_path=os.path.join(ROOT, 'confusion_matrix.json'),
                            char_lm_path=None, verbose=False)
    return e


W = [("ሰላም", 5000), ("ቤት", 4000), ("ነገር", 3000), ("ትምህርት", 2000), ("ወርቅ", 1000),
     ("ዓለም", 9000), ("ሀገር", 8000)]
e = mk(W)
print('model loaded:', e.lgbm_model is not None)
print('features:', e.lgbm_features)
print('vocab keys containing normalized forms:', [k for k in e.vocab if k in ('አለም', 'ዓለም', 'ሀገር', 'ሐገር')])


def show(tok, doc_freq=1):
    sig = e._cheap_signals(tok)
    feats = e._extract_lgbm_features(tok, sig, doc_freq=doc_freq)
    p = float(e.lgbm_model.predict([feats])[0])
    d = dict(zip(e.lgbm_features, feats))
    label = e.verdict(tok, doc_freq=doc_freq).label
    print(f"{tok!r:12} doc_freq={doc_freq:<3} p={p:.3f} label={label:5} "
          f"in_lex_feat={d['in_lexicon']} logf={d['log_corpus_freq']:.2f} "
          f"singleton={d['is_doc_singleton']} kenlm={d['char_kenlm_score']} "
          f"one_flip_cnt={d['one_flip_count']}")


print('--- clean in-lexicon words, no KenLM, default doc_freq')
for tok in ["ቤት", "ሰላም", "ነገር", "ትምህርት", "ወርቅ"]:
    show(tok)
print('--- doc_freq effect (same word)')
for df in [1, 2, 5, 10]:
    show("ቤት", doc_freq=df)

print('--- lexicon membership: normalizer-aware check vs feature lookup')
for tok in ["ዓለም", "ሀገር", "ሐገር"]:
    sig = e._cheap_signals(tok)
    feat_in_lex = e._extract_lgbm_features(tok, sig)[1]
    print(f"{tok!r}: cheap in_lexicon={sig.in_lexicon}  LGBM feature in_lexicon={feat_in_lex}")

t = time.time()
for _ in range(2000):
    e.verdict("ቤት")
print('ms per verdict (LGBM path, no KenLM): %.3f' % ((time.time() - t) / 2000 * 1000))

# batch vs single: predict call overhead
feats = [e._extract_lgbm_features("ቤት", e._cheap_signals("ቤት"))] * 1000
t = time.time()
e.lgbm_model.predict(feats)
print('ms per 1000 rows, batch predict: %.1f' % ((time.time() - t) * 1000))
