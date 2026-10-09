#!/usr/bin/env python3
import time
from amharic_ocr_corrector import AmharicOCRCorrector

KENLM_BIN = "/mnt/mintdata/Users/Des/Mint/Models/KenLM/word_model/amharic_5gram.binary"
CONFUSION_JSON = "/media/des/Tess/Tesseract2/confusion_matrix.json"
DICT_FILE = "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Projects/p16_morpheme_analysis/data/main_i7.txt"

corrector = AmharicOCRCorrector(
    kenlm_path=KENLM_BIN,
    confusion_matrix_path=CONFUSION_JSON,
    dict_vocab_path=DICT_FILE,
    beam_width=5,
    max_optical_cost=5.0,    # Increased from 4.2 to 5.0 to allow 2 character slips
    lambda_optical=0.9,
    lambda_lm=2.0,           # High LM weight strongly favors common collocations
    lambda_morph=1.5
)

real_ocr_text = """ኢትዮጵያውያን በሚያዋጡት ገንዘብ እና በመንግስት ወጪ እየተገነባ ያለው ይህ ግድብ በ5ሺህ 200 ሜጋ ዋት ሀይል በማመንጨት ከአፍሪካ ትልቁ ፕርጀከት ነው፡፡ ደቡባዊ የአፍሪካ ክፍል ከ4 የአፍሪካ ቀጣናዎች በቀዳሚነት በኮሮና መጠቃቱን የማእከሉ ሪፖርት ያሳያል"""

print("\n" + "=" * 70)
print("TESTING REAL OCR CORRUPTION (ፕርጀከት -> ፕሮጀክት)")
print("=" * 70)

t0 = time.time()
corrected = corrector.correct_text_block(real_ocr_text)
elapsed = (time.time() - t0) * 1000.0

print("\n[ORIGINAL OCR]:")
print(real_ocr_text)

print("\n[CORRECTED TEXT]:")
print(corrected)

print(f"\nProcessing Time: {elapsed:.1f} ms")
print("=" * 70)