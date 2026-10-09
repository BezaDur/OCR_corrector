#!/usr/bin/env python3
import time
from amharic_ocr_corrector import AmharicOCRCorrector

corrector = AmharicOCRCorrector(
    kenlm_path="/mnt/mintdata/Users/Des/Mint/Models/KenLM/word_model/amharic_5gram.binary",
    confusion_matrix_path="/media/des/Tess/Tesseract2/confusion_matrix.json",
    dict_vocab_path="/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Projects/p16_morpheme_analysis/data/main_i7.txt"
)

test_sentences = [
    # 1. Real-word slip in context (አና -> እና, አንደሚያደርጉ -> እንደሚያደርጉ)
    "የኢትዮጵያ፣ ግብጽ አና ሱዳን ባለሙያዎች በግድቡ የቴክኒክ አና የሕግ ጉዳይ ዙሪያ ውይይት አንደሚያደርጉ ይጠበቃል",
    # 2. Ghost loop artifact (ሶስትዮሽቨ -> ሶስትዮሽ)
    "ሶስተኛው ዙር የሕዳሴው ግድብ ሶስትዮሽቨ ውይይት በግብጽ ተጀመረ።",
    # 3. Optical root error (በሀዳሴው -> በህዳሴው)
    "ሶስተኛው ተመሳሳይ ውይይት ዛሬ በግብጽ ካይሮ በመካሄድ ላይ መሆኑን በሀዳሴው ግድብ የኢትዮጵያ",
    # 4. Multi-order corruption (ፕርጀከት -> ፕሮጀክት)
    "ትልቁ ፕርጀከት ነው፡፡",
    # 5. Clean token protection test (Verified words that must not be stripped or damaged)
    "ተከትሎም እርምጃን ወሰዱ፤ መወሰኗ ተረጋገጠ፤ በህገ-መንግስቱን መሰረት ዓ/ም ተጠናቀቀ።"
]

print("=" * 70)
print("TESTING FULL NORMALIZED PIPELINE (NO HARDCODING)")
print("=" * 70)

for s in test_sentences:
    t0 = time.time()
    res = corrector.correct_sentence(s)
    dt = (time.time() - t0) * 1000.0
    print(f"RAW OCR   : {s}")
    print(f"CORRECTED : {res} ({dt:.2f} ms)\n")

print("=" * 70)