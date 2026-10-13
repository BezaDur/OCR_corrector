#!/usr/bin/env python3
"""
Diagnoses Google's official 'amh.traineddata' accuracy broken down per font.
Reveals which font families Google trained on vs which ones cause it to fail.
"""

import os
import glob
import subprocess
from PIL import Image, ImageDraw, ImageFont

FONTS_DIR = "/media/des/Tess/Tesseract/TessTraining/Fonts"
GOOGLE_TESSDATA = "/media/des/Tess/Tesseract/Google_best"
TEMP_DIR = "/media/des/Tess/Tesseract2/font_diag_google"

BENCHMARK_TEXT = (
    "የኢትዮጵያ ፌዴራላዊ ዴሞክራሲያዊ ሪፐብሊክ መንግሥት የትምህርት ጥራትን ለማረጋገጥ አዲስ መመሪያ አወጣ። "
    "በዚህም መሠረት ሁሉም ዩኒቨርሲቲዎችና ኮሌጆች አዲሱን የትምህርት ሥርዓት ተግባራዊ ያደርጋሉ።"
)

def dp_edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]

def main():
    os.makedirs(TEMP_DIR, exist_ok=True)
    font_files = sorted(glob.glob(os.path.join(FONTS_DIR, "*.[to]tf")))
    
    print(f"Testing {len(font_files)} fonts against GOOGLE BEST Tesseract...\n")
    results = []

    for font_path in font_files:
        font_name = os.path.basename(font_path)
        try:
            font = ImageFont.truetype(font_path, 34)
            img = Image.new("RGB", (1600, 200), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text((30, 40), BENCHMARK_TEXT[:70], fill=(0, 0, 0), font=font)
            draw.text((30, 100), BENCHMARK_TEXT[70:], fill=(0, 0, 0), font=font)

            img_path = os.path.join(TEMP_DIR, f"{font_name}.png")
            out_base = os.path.join(TEMP_DIR, f"{font_name}_ocr")
            img.save(img_path)

            cmd = [
                "tesseract", img_path, out_base,
                "-l", "amh",
                "--tessdata-dir", GOOGLE_TESSDATA,
                "--psm", "6"
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            ocr_txt_file = f"{out_base}.txt"
            pred_text = ""
            if os.path.exists(ocr_txt_file):
                with open(ocr_txt_file, "r", encoding="utf-8") as f:
                    pred_text = " ".join(f.read().strip().split())

            clean_gt = " ".join(BENCHMARK_TEXT.split())
            edits = dp_edit_distance(clean_gt, pred_text)
            cer = (edits / max(1, len(clean_gt))) * 100.0
            acc = max(0.0, 100.0 - cer)

            results.append((font_name, acc, cer, pred_text[:35]))

        except Exception as e:
            results.append((font_name, 0.0, 100.0, f"Error: {e}"))

    results.sort(key=lambda x: x[1], reverse=True)

    print("=" * 75)
    print(f"{'FONT NAME':<35} | {'ACCURACY (GOOGLE)':<18} | {'SAMPLE OUTPUT'}")
    print("=" * 75)
    for name, acc, cer, sample in results:
        status = "✅" if acc >= 90 else ("⚠️" if acc >= 75 else "❌")
        print(f"{status} {name:<33} | {acc:6.2f}%           | {sample}...")
    print("=" * 75)

if __name__ == "__main__":
    main()