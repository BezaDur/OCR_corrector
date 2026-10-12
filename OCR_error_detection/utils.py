# utils.py — Ethiopic character lists used by the detector.
#
# Unicode ranges, rare glyph classes, and confusion pairs.

# Unicode ranges for Ethiopic script
ETHIOPIC = range(0x1200, 0x1380)          # Ethiopic block
ETHIOPIC_EXT = range(0x2D80, 0x2DE0)      # Ethiopic Extended


def _is_ethiopic(ch: str) -> bool:
    """True if `ch` is in the Ethiopic or Ethiopic Extended Unicode blocks."""
    o = ord(ch)
    return o in ETHIOPIC or o in ETHIOPIC_EXT


# 6th-order ("sadis") and other sparse series are rare in native vocabulary.
# A token dense in these is either a genuine oddity or mis-scanned.
# From ocr_engine_faults.txt the sadis->geez shift is the single most frequent
# upstream fault, so this is a high-yield check.
RARE_GLYPH_CLASSES = (
    set("ሠሡሢሣሤሥሦሧ"),      # ሠ series
    set("ጰጱጲጳጴጵጶጷ"),      # ጰ series
    set("ፀፁፂፃፄፅፆፇ"),      # ፀ series
    set("ዐዑዒዓዔዕዖ"),        # ዐ series (the classic አ/ዐ confusion)
    set("ሷሹሺሻሼሽሾሿ"),      # 6th/7th-order ዘ/ሸ-type forms
)


# Curated Amharic confusion classes.
#
# These are the homoglyph families that native text distinguishes but a
# scanner does not. They are a FALLBACK: the learned confusion matrix takes
# precedence, and this only fills what that matrix is missing.
#
# It is missing a lot. confusion_matrix.json was built against synthetic data
# whose error profile differs from the stock engine. The cross-series pairs
# below (notably AIN/PHHARYA) do not appear in the learned matrix at all.
CONFUSION_CLASSES = (
    # AIN / PHHARYA -- the single most-cited Amharic OCR confusion
    ("\u12a0", "\u12d0", "\u12d3"),                 # አ ዐ ዓ
    # the three H's
    ("\u1200", "\u1210", "\u1280"),                  # ሀ ሐ ኀ
    # sibilants
    ("\u1230", "\u1220"),                            # ሰ ሠ
    # the TSA/PHA series
    ("\u1338", "\u1340"),                            # ጸ ፀ
    ("\u1338", "\u1330"),                            # ጸ ጰ
    # nasals
    ("\u1290", "\u1298"),                            # ነ ኘ
    # labials
    ("\u1260", "\u1268"),                            # በ ቨ
    # velar / pharyngeal slip
    ("\u12a8", "\u12a0"),                            # ከ አ
)


_CODEPOINTS = (  # verified against unicodedata, kept next to the classes
    "አ=U+12A0 ዐ=U+12D0 ዓ=U+12D3 | ሀ=U+1200 ሐ=U+1210 ኀ=U+1280 | "
    "ሰ=U+1230 ሠ=U+1220 | ጸ=U+1338 ፀ=U+1340 ጰ=U+1330 | "
    "ነ=U+1290 ኘ=U+1298 | በ=U+1260 ቨ=U+1268 | ከ=U+12A8"
)


def _confusion_alts(ch: str) -> list:
    """Optically confusable alternatives for `ch`, curated + learned."""
    out = []
    for cls in CONFUSION_CLASSES:
        if ch in cls:
            out.extend(c for c in cls if c != ch)
    return out


# Punctuation that Tesseract routinely confuses or drops.
ETHIOPIC_PUNCT = set("፡።፣፤፥፦፧፨")