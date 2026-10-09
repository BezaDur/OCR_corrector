#!/usr/bin/env python3
import time
from amharic_ocr_corrector import AmharicOCRCorrector, EthiopicPunctuationManager

corrector = AmharicOCRCorrector(
    kenlm_path="/mnt/mintdata/Users/Des/Mint/Models/KenLM/word_model/amharic_5gram.binary",
    confusion_matrix_path="/media/des/Tess/Tesseract2/confusion_matrix.json",
    dict_vocab_path="/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Projects/p16_morpheme_analysis/data/main_i7.txt",
    beam_width=4,
    lambda_optical=1.0,
    lambda_lm=1.8,
    lambda_morph=1.5
)

test_sentences = [
    # Sentence 1: Tests 'አና' -> 'እና' AND 10-char 'አንደሚያደርጉ' -> 'እንደሚያደርጉ'
    "የኢትዮጵያ፣ ግብጽ አና ሱዳን ባለሙያዎች በግድቡ የቴክኒክ አና የሕግ ጉዳይ ዙሪያ ውይይት አንደሚያደርጉ ይጠበቃል",
    # Sentence 2: Tests ghost loop artifact 'ሶስትዮሽቨ' -> 'ሶስትዮሽ'
    "ሶስተኛው ዙር የሕዳሴው ግድብ ሶስትዮሽቨ ውይይት በግብጽ ተጀመረ።",
    # Sentence 3: Tests root repair 'በሀዳሴው' -> 'በህዳሴው'
    "ሶስተኛው ተመሳሳይ ውይይት ዛሬ በግብጽ ካይሮ በመካሄድ ላይ መሆኑን በሀዳሴው ግድብ የኢትዮጵያ"
]

print("\n" + "=" * 70)
print("TESTING PAGE 1 REPAIRS WITH LENGTH-ADAPTIVE CANDIDATE SEARCH")
print("=" * 70)

# Profile Sentence 1 per-token to inspect candidate generation speed
print("[PER-TOKEN LATENCY PROFILE FOR SENTENCE 1]:")
norm_s1 = EthiopicPunctuationManager.normalize_sentence_text(test_sentences[0])
for raw_tok in norm_s1.split():
    _, word, _ = EthiopicPunctuationManager.decompose_token(raw_tok)
    t0 = time.time()
    cands = corrector.generate_candidates(word)
    dt_tok = (time.time() - t0) * 1000.0
    cand_sample = [c[0] for c in cands[:3]]
    print(f"  Token: {word:<15} (len={len(word):2d}) | Time: {dt_tok:6.2f} ms | Cands: {cand_sample}")

print("\n" + "-" * 70)
print("[END-TO-END SENTENCE RESULTS]:")
print("-" * 70)

for s in test_sentences:
    t0 = time.time()
    res = corrector.correct_sentence(s)
    dt = (time.time() - t0) * 1000.0
    print(f"RAW OCR   : {s}")
    print(f"CORRECTED : {res} ({dt:.1f} ms)\n")
print("=" * 70)