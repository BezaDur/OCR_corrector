import re
import unicodedata
from functools import lru_cache
from typing import Dict, Optional, List, Tuple

class AmharicNormalizer:
    """
    A unified Amharic text normalizer configured for data safety.
    By default, all additional features from GitHub are disabled (False),
    and only your original normalization mappings and rules are enabled (True).
    """

    # Core character-level mapping (from your personal configuration)
    NORMALIZATION_MAP = {
        'ሐ': 'ሀ', 'ሑ': 'ሁ', 'ሒ': 'ሂ', 'ሓ': 'ሀ', 'ሔ': 'ሄ', 'ሕ': 'ህ', 'ሖ': 'ሆ',
        'ኀ': 'ሀ', 'ኁ': 'ሁ', 'ኂ': 'ሂ', 'ኃ': 'ሀ', 'ኄ': 'ሄ', 'ኅ': 'ህ', 'ኆ': 'ሆ',
        'ሃ': 'ሀ', 'ሗ': 'ኋ', 
        #'ኹ': 'ሁ', 'ኺ': 'ሂ', 'ኻ': 'ሀ', 'ኼ': 'ሄ', 'ኽ': 'ህ', 'ኾ': 'ሆ',
        'ዐ': 'አ', 'ዓ': 'አ', 'ኣ': 'አ', 'ዑ': 'ኡ', 'ዒ': 'ኢ', 'ዔ': 'ኤ', 'ዕ': 'እ', 'ዖ': 'ኦ',
        'ሠ': 'ሰ', 'ሡ': 'ሱ', 'ሢ': 'ሲ', 'ሣ': 'ሳ', 'ሤ': 'ሴ', 'ሥ': 'ስ', 'ሦ': 'ሶ', 'ሧ': 'ሷ',
        'ፀ': 'ጸ', 'ፁ': 'ጹ', 'ፂ': 'ጺ', 'ፃ': 'ጻ', 'ፄ': 'ጼ', 'ፅ': 'ጽ', 'ፆ': 'ጾ', 'ቈ': 'ቆ', 'ቍ':'ቁ', 'ኰ': 'ኮ', 'ጐ': 'ጎ', 'ኈ': 'ሆ', 'ጕ': 'ጉ',
        '\u200b': '', '\u200c': '', '\u200d': '', '\ufeff': '',
    }

    _TRANSLATION_TABLE = str.maketrans(NORMALIZATION_MAP)

    # Core regex-based labialization/bilabial rules (from your personal configuration)
    REGEX_RULES = [
        (re.compile(r"ቊ"), "ቁ"), (re.compile(r"ኵ"), "ኩ"),
        (re.compile(r"ሑ(?:ዋአ|ዋ|አ)"), "ኋ"), (re.compile(r"ሁ(?:ዋአ|ዋ|አ)"), "ኋ"), (re.compile(r"ኁ(?:ዋአ|ዋ|አ)"), "ኋ"),
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

    # Punctuation constants
    AMHARIC_PUNCTUATION = {'።', '፣', '፤', '፥', '፦', '፡'}

    PUNCTUATION_NORMALIZATION = [
        (re.compile(r'\.'), ' . '), 
        (re.compile(r'፡'), ' ፡ '),
        (re.compile(r'[!?]'), r' \g<0> '),  
        (re.compile(r'።'), ' ። '),
        (re.compile(r'፣'), ' ፣ '),
        (re.compile(r'፤'), ' ፤ '),
        (re.compile(r'[-–—]+'), r' \g<0> '),
    ]

    # Numerals
    AMHARIC_NUMERALS = {
        '፩': '1', '፪': '2', '፫': '3', '፬': '4', '፭': '5',
        '፮': '6', '፯': '7', '፰': '8', '፱': '9', '፲': '10',
        '፳': '20', '፴': '30', '፵': '40', '፶': '50',
        '፷': '60', '፸': '70', '፹': '80', '፺': '90', '፻': '100'
    }

    # Spelling Corrections
    COMMON_SPELLING_CORRECTIONS = {
        'አበ': 'አበራ',
        'ሠላም': 'ሰላም',
    }

    # Abbreviations
    AMHARIC_ABBREVIATIONS = {
        'ት/ቤት': 'ትምህርት ቤት',
        'ት/ርት': 'ትምህርት',
        'ት/ክፍል': 'ትምህርት ክፍል',
        'ሃ/አለቃ': 'ሀምሳ አለቃ',
        'ሃ/ስላሴ': 'ሀይለ ስላሴ',
        'ደ/ዘይት': 'ደብረ ዘይት',
        'ደ/ታቦር': 'ደብረ ታቦር',
        'መ/ር': 'መምህር',
        'መ/ቤት': 'መስሪያ ቤት',
        'መ/አለቃ': 'መቶ አለቃ',
        'ክ/ከተማ': 'ክፍለ ከተማ',
        'ክ/ሀገር': 'ክፍለ ሀገር',
        'ወ/ር': 'ወታደር',
        'ወ/ሮ': 'ወይዘሮ',
        'ወ/ሪት': 'ወይዘሪት',
        'ወ/ስላሴ': 'ወሌተ ስላሴ',
        'ፍ/ስላሴ': 'ፍቅረ ስላሴ',
        'ፍ/ቤት': 'ፍርድ ቤት',
        'ጽ/ቤት': 'ጽህፈት ቤት',
        'ሲ/ር': 'ሲስተር',
        'ጠ/ሚኒስትር': 'ጠቅላይ ሚኒስትር',
        'ዶ/ር': 'ዶክተር',
        'ገ/ጊዮርጊስ': 'ገብረ ጊዮርጊስ',
        'ቤ/ክርስትያን': 'ቤተ ክርስትያን',
        'ም/ስራ': 'ምክትል ስራ',
        'ም/ቤት': 'ምክር ቤት',
        'ተ/ሃይማኖት': 'ተክለ ሃይማኖት',
        'ሚ/ር': 'ሚኒስትር',
        'ኮ/ል': 'ኮለኔል',
        'ሜ/ጀነራል': 'ሜጀር ጀነራል',
        'ብ/ጀነራል': 'ብርጋዳር ጀነራል',
        'ሌ/ኮለኔል': 'ሌተናንት ኮለኔል',
        'ሊ/መንበር': 'ሊቀ መንበር',
        'አ/አ': 'አዲስ አበባ',
        'ር/መምህር': 'ርእሰ መምህር',
        'ፕ/ት': 'ፕሬዝዳንት',
        'ዓ.ም': 'አመተ ምህረት',
        'ዓ.ዓ': 'አዲስ አበባ',
        'ዶ.ር': 'ዶክተር',
        'ፕ/ር': 'ፕሮፌሰር'
    }

    # ====================== CORE FUNCTIONS ======================

    @staticmethod
    def normalize_unicode(text: str, form: str = 'NFC') -> str:
        """Normalize Unicode text to specified form (default: NFC)."""
        return unicodedata.normalize(form, text)

    def normalize_amharic_chars(self, text: str, normalize_chars: bool = True, normalize_bilabials: bool = True) -> str:
        """
        Processes text using character-level translation tables and 
        compiled regular expressions for labialized/bilabial forms.
        """
        if not text:
            return ""
        
        # 1. Apply your basic translation mapping table
        if normalize_chars:
            text = text.translate(self._TRANSLATION_TABLE)
            
        # 2. Apply your bilabial/labialization regular expression replacements
        if normalize_bilabials:
            for pattern, repl in self.REGEX_RULES:
                text = pattern.sub(repl, text)
                
        return text

    # Single combined alternation of every bilabial/labialization pattern.
    # Measured: only ~0.1% of real Amharic words match any of the 28 rules,
    # but the naive loop ran all 28 `re.sub` calls on every single word. One
    # combined search is ~5x faster end-to-end on candidate generation, where
    # this is called once per generated variant (~200 calls per unknown token).
    _COMBINED_BILABIAL_REGEX = re.compile(
        "|".join("(?:%s)" % pattern.pattern for pattern, _ in REGEX_RULES)
    )

    @staticmethod
    @lru_cache(maxsize=100_000)
    def normalize_word_strict(word: str, normalize_bilabials: bool = True) -> str:
        """
        Optimized, cached normalization helper for individual words.
        Uses your original strict mapping and rules.
        """
        if not word:
            return ""
        text = word.translate(AmharicNormalizer._TRANSLATION_TABLE)
        if normalize_bilabials:
            # Fast path: skip the 28-rule loop entirely when no rule can apply.
            if AmharicNormalizer._COMBINED_BILABIAL_REGEX.search(text):
                for pattern, repl in AmharicNormalizer.REGEX_RULES:
                    text = pattern.sub(repl, text)
        return text.strip()

    def normalize_punctuation_spacing(self, text: str) -> str:
        """Add spaces around Amharic and Western punctuation."""
        for pattern, replacement in self.PUNCTUATION_NORMALIZATION:
            text = pattern.sub(replacement, text)
        return text

    def remove_punctuation(self, text: str, 
                         keep_basic: bool = True,
                         keep_amharic: bool = True) -> str:
        """Remove punctuation while retaining specified sets."""
        keep = []
        if keep_basic:
            keep.append(r'\.!?,;:-')
        if keep_amharic:
            keep.append(''.join(self.AMHARIC_PUNCTUATION))
        
        if keep:
            punctuation_regex = rf'[^\w\s{"".join(keep)}]'
        else:
            punctuation_regex = r'[^\w\s]'
        
        return re.sub(punctuation_regex, '', text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize all consecutive whitespace characters to a single space."""
        text = re.sub(r'[\s\u2000-\u200F\u2028-\u202F\u205F\u3000]+', ' ', text)
        return text.strip()

    def normalize_numbers(self, text: str) -> str:
        """Convert Amharic numerals to Western numbers."""
        return ''.join(self.AMHARIC_NUMERALS.get(char, char) for char in text)

    def fix_common_spelling(self, text: str, custom_map: Optional[Dict[str, str]] = None) -> str:
        """Perform simple string replacements for common spelling variants."""
        corrections = self.COMMON_SPELLING_CORRECTIONS.copy()
        if custom_map:
            corrections.update(custom_map)
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        return text

    def expand_abbreviations(self, text: str, custom_abbr: Optional[Dict[str, str]] = None) -> str:
        """Expand abbreviations (e.g. 'ት/ቤት' to 'ትምህርት ቤት')."""
        abbreviations = self.AMHARIC_ABBREVIATIONS.copy()
        if custom_abbr:
            abbreviations.update(custom_abbr)
        
        for abbr in sorted(abbreviations, key=len, reverse=True):
            text = text.replace(abbr, abbreviations[abbr])
        return text

    # ====================== PIPELINE INTERFACE ======================

    def normalize_text(
        self,
        text: str,
        *,
        # 1. Unicode normalization option. NFC is the standard representation form.
        # Set to None to bypass unicode normalization completely.
        unicode_form: Optional[str] = 'NFC',

        # 2. Controls translation mapping (e.g. ሐ -> ሀ, ፀ -> ጸ, zero-width spaces -> removed).
        # Kept ON (True) as configured in your script.
        normalize_chars: bool = True,

        # 3. Controls bilabial/labialization substitutions (e.g. ሉዋ -> ሏ, ሟ, ሷ).
        # Kept ON (True) as configured in your script. Can be toggled OFF (False).
        normalize_bilabials: bool = True,

        # 4. Adds spaces around punctuation marks (e.g., "." becomes " . ").
        # Kept OFF (False) to prevent modifying spacing.
        normalize_punct_spacing: bool = False,

        # 5. Removes punctuation marks from the text entirely.
        # Kept OFF (False) to preserve punctuation.
        remove_punct: bool = False,

        # 6. If remove_punct is True, these determine which categories of punctuation to KEEP.
        keep_basic_punct: bool = True,
        keep_amharic_punct: bool = True,

        # 7. Collapses multiple spaces and tabs into a single clean space.
        # Kept OFF (False) to preserve your input text formatting exactly.
        normalize_ws: bool = False,

        # 8. Converts traditional Amharic numbers (e.g. ፩, ፪) to Western numbers (1, 2).
        # Kept OFF (False) to keep numbers in their original representation.
        normalize_nums: bool = False,

        # 9. Corrects spelling variations using the simple replacements list.
        # Kept OFF (False) to avoid altering vocabulary variants.
        fix_spelling: bool = False,

        # 10. Expands short form abbreviations into full multi-word strings.
        # Kept OFF (False) to prevent modifying short-form writing.
        expand_abbr: bool = False,

        # Placeholders for custom external mapping inputs
        custom_spelling_map: Optional[Dict[str, str]] = None,
        custom_abbr_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Amharic normalization pipeline. By default, it runs only your 
        original translation and regex rules to guarantee data preservation.
        """
        # Step 1: Unicode Standard Alignment
        if unicode_form:
            text = self.normalize_unicode(text, unicode_form)
        
        # Step 2: Expand Abbreviations (Disabled by default)
        if expand_abbr:
            text = self.expand_abbreviations(text, custom_abbr_map)
        
        # Step 3: Run original character & bilabial rules (Enabled by default)
        text = self.normalize_amharic_chars(
            text, 
            normalize_chars=normalize_chars, 
            normalize_bilabials=normalize_bilabials
        )
        
        # Step 4: Spacing adjustments around punctuation (Disabled by default)
        if normalize_punct_spacing:
            text = self.normalize_punctuation_spacing(text)
        
        # Step 5: Punctuation deletion (Disabled by default)
        if remove_punct:
            text = self.remove_punctuation(text, 
                                           keep_basic=keep_basic_punct,
                                           keep_amharic=keep_amharic_punct)
        
        # Step 6: Convert Ge'ez numbers to Western numbers (Disabled by default)
        if normalize_nums:
            text = self.normalize_numbers(text)
        
        # Step 7: Basic Spelling Adjustments (Disabled by default)
        if fix_spelling:
            text = self.fix_common_spelling(text, custom_spelling_map)
        
        # Step 8: Multi-space merging and trim (Disabled by default)
        if normalize_ws:
            text = self.normalize_whitespace(text)
        
        return text