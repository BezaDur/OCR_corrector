# Wrapper — main corrector moved to 4_ocr_correction/amharic_ocr_corrector.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "4_ocr_correction"))
from amharic_ocr_corrector import *
