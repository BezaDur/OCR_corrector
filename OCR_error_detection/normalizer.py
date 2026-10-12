# normalizer.py — Fixes Amharic spelling differences for dictionary lookup.
#
# Example: ሐ→ሀ, ሠ→ሰ, ዐ→አ. Same logic as the main system's normalizer.
import re
import unicodedata
from functools import lru_cache
from typing import Optional, Dict


class AmharicNormalizer:
    """
    Amharic text normalizer for detection and lookup.

    By default applies only the character-level translation and bilabial rules
    needed for consistent lexicon keying.  Text-level features (punctuation
    spacing, numeral conversion, abbreviation expansion) are available but off.
    """

    # Core character-level mapping
    NORMALIZATION_MAP: Dict[str, str] = {
        'ሐ': 'ሀ', 'ሑ': 'ሁ', 'ሒ': 'ሂ', 'ሓ': 'ሀ', 'ሔ': 'ሄ', 'ሕ': 'ህ', 'ሖ': 'ሆ',
        'ኀ': 'ሀ', 'ኁ': 'ሁ', 'ኂ': 'ሂ', 'ኃ': 'ሀ', 'ኄ': 'ሄ', 'ኅ': 'ህ', 'ኆ': 'ሆ',
        'ሃ': 'ሀ', 'ሗ': 'ኋ',
        'ዐ': 'አ', 'ዓ': 'አ', 'ኣ': 'አ', 'ዑ': 'ኡ', 'ዒ': 'ኢ', 'ዔ': 'ኤ', 'ዕ': 'እ', 'ዖ': 'ኦ',
        'ሠ': 'ሰ', 'ሡ': 'ሱ', 'ሢ': 'ሲ', 'ሣ': 'ሳ', 'ሤ': 'ሴ', 'ሥ': 'ስ', 'ሦ': 'ሶ', 'ሧ': 'ሷ',
        'ፀ': 'ጸ', 'ፁ': 'ጹ', 'ፂ': 'ጺ', 'ፃ': 'ጻ', 'ፄ': 'ጼ', 'ፅ': 'ጽ', 'ፆ': 'ጾ',
        'ቈ': 'ቆ', 'ቍ': 'ቁ', 'ኰ': 'ኮ', 'ጐ': 'ጎ', 'ኈ': 'ሆ', 'ጕ': 'ጉ',
        '\u200b': '', '\u200c': '', '\u200d': '', '\ufeff': '',
    }

    _TRANSLATION_TABLE = str.maketrans(NORMALIZATION_MAP)

    # Bilabial / labialization regex rules
    REGEX_RULES = [
        (re.compile(r"ቊ"), "ቁ"), (re.compile(r"ኵ"), "ኩ"),
        (re.compile(r"ሑ(?:ዋአ|ዋ|አ)"), "ኋ"), (re.compile(r"ሁ(?:ዋአ|ዋ|አ)"), "ኋ"),
        (re.compile(r"ኁ(?:ዋአ|ዋ|አ)"), "ኋ"),
        (re.compile(r"ሉ(?:ዋአ|ዋ|አ)"), "ሏ"), (re.compile(r"ሙ(?:ዋአ|ዋ|አ)"), "ሟ"),
        (re.compile(r"ሱ(?:ዋአ|ዋ|አ)"), "ሷ"), (re.compile(r"ሡ(?:ዋአ|ዋ|አ)"), "ሷ"),
        (re.compile(r"ሩ(?:ዋአ|ዋ|አ)"), "ሯ"), (re.compile(r"ሹ(?:ዋአ|ዋ|አ)"), "ሿ"),
        (re.compile(r"ቁ(?:ዋአ|ዋ|አ)"), "ቋ"), (re.compile(r"ቡ(?:ዋአ|ዋ|አ)"), "ቧ"),
        (re.compile(r"ቱ(?:ዋአ|ዋ|አ)"), "ቷ"), (re.compile(r"ቹ(?:ዋአ|ዋ|አ)"), "ቿ"),
        (re.compile(r"ኑ(?:ዋአ|ዋ|አ)"), "ኗ"), (re.compile(r"ኙ(?:ዋአ|ዋ|አ)"), "ኟ"),
        (re.compile(r"ኩ(?:ዋአ|ዋ|አ)"), "ኳ"), (re.compile(r"ዙ(?:ዋአ|ዋ|አ)"), "ዟ"),
        (re.compile(r"ዡ(?:ዋአ|ዋ|አ)"), "ዧ"), (re.compile(r"ጉ(?:ዋአ|ዋ|አ)"), "ጓ"),
        (re.compile(r"ዱ(?:ዋአ|ዋ|አ)"), "ዷ"), (re.compile(r"ጁ(?:ዋአ|ዋ|አ)"), "ጇ"),
        (re.compile(r"ጡ(?:ዋአ|ዋ|አ)"), "ጧ"), (re.compile(r"ጩ(?:ዋአ|ዋ|አ)"), "ጯ"),
        (re.compile(r"ጹ(?:ዋአ|ዋ|አ)"), "ጿ"), (re.compile(r"ፁ(?:ዋአ|ዋ|አ)"), "ጿ"),
        (re.compile(r"ፉ(?:ዋአ|ዋ|አ)"), "ፏ"), (re.compile(r"ፑ(?:ዋአ|ዋ|አ)"), "ፗ"),
    ]

    # Single combined alternation for the fast path
    _COMBINED_BILABIAL_REGEX = re.compile(
        "|".join("(?:%s)" % pattern.pattern for pattern, _ in REGEX_RULES)
    )

    @staticmethod
    @lru_cache(maxsize=100_000)
    def normalize_word_strict(word: str, normalize_bilabials: bool = True) -> str:
        """Normalize a single word for vocabulary lookup.  Cached."""
        if not word:
            return ""
        text = word.translate(AmharicNormalizer._TRANSLATION_TABLE)
        if normalize_bilabials:
            if AmharicNormalizer._COMBINED_BILABIAL_REGEX.search(text):
                for pattern, repl in AmharicNormalizer.REGEX_RULES:
                    text = pattern.sub(repl, text)
        return text.strip()

    @staticmethod
    def normalize_unicode(text: str, form: str = 'NFC') -> str:
        """Normalize Unicode text to NFC."""
        return unicodedata.normalize(form, text) if text else ""

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse multiple whitespace into a single space."""
        text = re.sub(r'[\s\u2000-\u200F\u2028-\u202F\u205F\u3000]+', ' ', text)
        return text.strip()

    def normalize_text(self, text: str, *, unicode_form: Optional[str] = 'NFC') -> str:
        """Full text normalization pipeline (default: NFC + char rules)."""
        if not text:
            return ""
        if unicode_form:
            text = self.normalize_unicode(text, unicode_form)
        text = text.translate(self._TRANSLATION_TABLE)
        if self._COMBINED_BILABIAL_REGEX.search(text):
            for pattern, repl in self.REGEX_RULES:
                text = pattern.sub(repl, text)
        return text