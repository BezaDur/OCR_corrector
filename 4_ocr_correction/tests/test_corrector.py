#!/usr/bin/env python3
# Make the repo-root shared modules importable from this subfolder.
# --- make repo_paths importable from any working directory -------------------
import os as _os
import sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(
        _os.path.join(_root, "tools", "stage_paths.py")):
    _root = _os.path.dirname(_root)
# tools/ first so stage_paths and repo_paths resolve, then register every stage
# folder so cross-stage imports work by bare module name. This replaces the
# thirteen root-level shims that used to forward those imports by hand.
_sys.path.insert(0, _os.path.join(_root, "tools"))
import stage_paths  # noqa: E402  (imported after the path bootstrap)
stage_paths.register(_root)
import repo_paths  # noqa: E402
# ----------------------------------------------------------------------------

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import time
from amharic_ocr_corrector import AmharicOCRCorrector

KENLM_BIN = repo_paths.word_kenlm()
CONFUSION_JSON = repo_paths.confusion_matrix()
DICT_FILE = repo_paths.main_i7_dict()

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