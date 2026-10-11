"""
AmharicBoundaryPhonology: Centralized Morphophonological Boundary Engine.
Handles 49-cell matrix vowel inversion, coronal palatalization, 
7th-order coverb labialization, coda shifts, prefix a-coalescence,
and vocalic order analysis for Intensive and Attenuative Ideophones.
"""
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple, Union


#: Voice prefixes Amharic writes in front of a stem without altering the stem's
#: own consonants: passive/reflexive ``ተ-``/``ተን-``/``ታ-`` and causative
#: ``አ-``/``አስ-``/``አን-``/``አስተ-``.  Longest first, so ``ተን-`` is never read as
#: ``ተ-`` and ``አስተ-`` is never read as ``አ-``.
#:
#: The voice matters to reading selection because a derived (passive) lexeme
#: behaves differently from a plain one: for a ተ- passive the 3MS perfective and
#: the 2MS imperative are written identically (``ተመራ`` = "he was led" and "be
#: led!"), whereas the matching active lexeme keeps them apart (``ዘረጋ`` /
#: ``ዘርጋ``).  Both share the vocalic-order shape ``(0,0,3)``, so the shape alone
#: cannot tell them apart.
VOICE_PREFIXES: Tuple[str, ...] = ("ተን", "አስተ", "አስ", "አን", "ተ", "ታ", "አ")


def consonantal_skeleton(word: str) -> str:
    """Reduce ``word`` to its consonantal citation skeleton.

    Every syllable is mapped to the Ge'ez-series (1st order) letter of its
    consonant, so vowel order is ignored: ``ቸገር``, ``ቸገረ`` and ``ቸግር`` all reduce
    to ``ቸገረ``.  This is the key used by ``data/experiencer_verbs.json``, where a
    lexical class has to be recognised regardless of which template a given
    inflected form happens to use.
    """
    return "".join(
        AmharicBoundaryPhonology.get_char_info(char)[0] for char in word
    )


def split_voice_prefix(word: str) -> Tuple[str, str]:
    """Split one leading voice prefix off ``word``.

    Returns ``(prefix, residue)``; ``prefix`` is ``""`` when there is none.  The
    residue must keep at least two characters, so a bare prefix is never
    stripped down to a fragment.
    """
    for prefix in VOICE_PREFIXES:
        if word.startswith(prefix) and len(word) - len(prefix) >= 2:
            return prefix, word[len(prefix):]
    return "", word


class AmharicBoundaryPhonology:
    GE_EZ_SERIES = [
        ("ሀ", "ሁ", "ሂ", "ሃ", "ሄ", "ህ", "ሆ", "ኋ"),
        ("ለ", "ሉ", "ሊ", "ላ", "ሌ", "ል", "ሎ", "ሏ"),
        ("ሐ", "ሑ", "ሒ", "ሓ", "ሔ", "ሕ", "ሖ", "ሗ"),
        ("መ", "ሙ", "ሚ", "ማ", "ሜ", "ም", "ሞ", "ሟ"),
        ("ሠ", "ሡ", "ሢ", "ሣ", "ሤ", "ሥ", "ሦ", "ሧ"),
        ("ረ", "ሩ", "ሪ", "ራ", "ሬ", "ር", "ሮ", "ሯ"),
        ("ሰ", "ሱ", "ሲ", "ሳ", "ሴ", "ስ", "ሶ", "ሷ"),
        ("ሸ", "ሹ", "ሺ", "ሻ", "ሼ", "ሽ", "ሾ", "ሿ"),
        ("ቀ", "ቁ", "ቂ", "ቃ", "ቄ", "ቅ", "ቆ", "ቋ"),
        ("በ", "ቡ", "ቢ", "ባ", "ቤ", "ብ", "ቦ", "ቧ"),
        ("ተ", "ቱ", "ቲ", "ታ", "ቴ", "ት", "ቶ", "ቷ"),
        ("ቸ", "ቹ", "ቺ", "ቻ", "ቼ", "ች", "ቾ", "ቿ"),
        ("ኀ", "ኁ", "ኂ", "ኃ", "ኄ", "ኅ", "ኆ", "ኈ"),
        ("ነ", "ኑ", "ኒ", "ና", "ኔ", "ን", "ኖ", "ኗ"),
        ("ኘ", "ኙ", "ኚ", "ኛ", "ኜ", "ኝ", "ኞ", "ኟ"),
        ("አ", "ኡ", "ኢ", "አ", "ኤ", "እ", "ኦ", ""),
        ("ከ", "ኩ", "ኪ", "ካ", "ኬ", "ክ", "ኮ", "ኳ"),
        ("ኸ", "ኹ", "ኺ", "ኻ", "ኼ", "ኽ", "ኾ", "ዃ"),
        ("ወ", "ዉ", "ዊ", "ዋ", "ዌ", "ው", "ዎ", ""),
        ("ዐ", "ዑ", "ዒ", "ዓ", "ዔ", "ዕ", "ዖ", ""),
        ("ዘ", "ዙ", "ዚ", "ዛ", "ዜ", "ዝ", "ዞ", "ዟ"),
        ("ዠ", "ዡ", "ዢ", "ዣ", "ዤ", "ዥ", "ዦ", "ዧ"),
        ("የ", "ዩ", "ዪ", "ያ", "ዬ", "ይ", "ዮ", ""),
        ("ደ", "ዱ", "ዲ", "ዳ", "ዴ", "ድ", "ዶ", "ዷ"),
        ("ጀ", "ጁ", "ጂ", "ጃ", "ጄ", "ጅ", "ጆ", "ጇ"),
        ("ገ", "ጉ", "ጊ", "ጋ", "ጌ", "ግ", "ጎ", "ጓ"),
        ("ጠ", "ጡ", "ጢ", "ጣ", "ጤ", "ጥ", "ጦ", "ጧ"),
        ("ጨ", "ጩ", "ጪ", "ጫ", "ጬ", "ጭ", "ጮ", "ጯ"),
        ("ጰ", "ጱ", "ጲ", "ጳ", "ጴ", "ጵ", "ጶ", ""),
        ("ጸ", "ጹ", "ጺ", "ጻ", "ጼ", "ጽ", "ጾ", "ጿ"),
        ("ፀ", "ፁ", "ፂ", "ፃ", "ፄ", "ፅ", "ፆ", ""),
        ("ፈ", "ፉ", "ፊ", "ፋ", "ፌ", "ፍ", "ፎ", "ፏ"),
        ("ፐ", "ፑ", "ፒ", "ፓ", "ፔ", "ፕ", "ፖ", "ፗ"),
        ("ቨ", "ቩ", "ቪ", "ቫ", "ቬ", "ቭ", "ቮ", ""),
    ]

    MATRIX_LOOKUP_SUFFIX: Dict[int, Dict[int, List[int]]] = {
        0: {0: [5, 0], 3: [5, 0], 5: [5, 0]},
        1: {1: [0, 5], 5: [5]},
        2: {2: [5], 3: [3], 5: [5]},
        3: {0: [5], 3: [5, 3, 0], 5: [5], 7: [5]},
        4: {4: [5, 4], 5: [5]},
        5: {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6]},
        6: {6: [3, 5]},
    }

    PURE_VOWEL_SUFFIXES: Set[str] = {"ኡ", "ኤ", "ኦ"}
    WEAK_DELETION_URS: Dict[str, str] = {"እህ": "ህ", "እሽ": "ሽ", "እት": "ት"}
    
    PALATAL_TO_CORONAL: Dict[str, Tuple[str, ...]] = {
        "ች": ("ት",), "ጅ": ("ድ",), "ጭ": ("ጥ", "ጽ"),
        "ሽ": ("ስ",), "ዥ": ("ዝ",), "ኝ": ("ን",), "ይ": ("ል",)
    }
    CORONAL_TO_PALATAL_SERIES: Dict[str, str] = {
        "ት": "ቸ", "ድ": "ጀ", "ጥ": "ጨ", "ጽ": "ጨ", "ፅ": "ጨ",
        "ስ": "ሸ", "ዝ": "ዠ", "ን": "ኘ", "ል": "የ",
        "ታ": "ቻ", "ዳ": "ጃ", "ጣ": "ጫ", "ጻ": "ጫ", "ፃ": "ጫ",
        "ሳ": "ሻ", "ዛ": "ዣ", "ና": "ኛ", "ላ": "ያ"
    }
    PALATAL_CONSONANTS: Set[str] = {"ኘ", "ጀ", "ጨ", "ሸ", "ቸ", "ዠ", "ኝ", "ጅ", "ጭ", "ሽ", "ች", "ዥ"}

    LABIOVELAR_MAP: Dict[str, Tuple[str, int]] = {
        "ቈ": ("ቀ", 6), "ቊ": ("ቀ", 2), "ቋ": ("ቀ", 7), "ቌ": ("ቀ", 4), "ቍ": ("ቀ", 1),
        "ኰ": ("ከ", 6), "ኲ": ("ከ", 2), "ኳ": ("ከ", 7), "ኴ": ("ከ", 4), "ኵ": ("ከ", 1),
        "ጐ": ("ገ", 6), "ጒ": ("ገ", 2), "ጓ": ("ገ", 7), "ጔ": ("ገ", 4), "ጕ": ("ገ", 1),
        "ኈ": ("ኀ", 6), "ኊ": ("ኀ", 2), "ኋ": ("ኀ", 7), "ኌ": ("ኀ", 4), "ኍ": ("ኀ", 1),
        "ዃ": ("ኸ", 7),
    }

    CHAR_MAP: Dict[str, Tuple[int, int]] = {}
    for _s_idx, _series in enumerate(GE_EZ_SERIES):
        for _o_idx, _char in enumerate(_series):
            if _char:
                CHAR_MAP[_char] = (_s_idx, _o_idx)
    _a_idx = next(i for i, s in enumerate(GE_EZ_SERIES) if s[0] == "አ")
    CHAR_MAP["ኧ"] = (_a_idx, 0)

    def __init__(self):
        self.char_map = self.CHAR_MAP

    @staticmethod
    @lru_cache(maxsize=4096)
    def get_char_info(char: str) -> Tuple[str, int]:
        if not char:
            return "", -1
        if char in AmharicBoundaryPhonology.LABIOVELAR_MAP:
            return AmharicBoundaryPhonology.LABIOVELAR_MAP[char]
        if char in ("አ", "ሀ", "ሐ", "ዐ"):
            return char, 3
        if char == "ኧ":
            return "አ", 0
        if char in AmharicBoundaryPhonology.CHAR_MAP:
            s_idx, o_idx = AmharicBoundaryPhonology.CHAR_MAP[char]
            return AmharicBoundaryPhonology.GE_EZ_SERIES[s_idx][0], o_idx
        code = ord(char)
        if 0x1200 <= code <= 0x135A:
            offset = (code - 0x1200) % 8
            base = chr(code - offset)
            if base in AmharicBoundaryPhonology.CHAR_MAP:
                return base, offset
        return char, -1

    @staticmethod
    @lru_cache(maxsize=8192)
    def get_vocalic_orders(text: str) -> List[int]:
        orders = []
        for char in text:
            _, ord_idx = AmharicBoundaryPhonology.get_char_info(char)
            if ord_idx >= 0:
                orders.append(ord_idx)
        return orders

    def synthesize_plural(self, stem: str) -> List[str]:
        """
        Phonologically synthesizes plural forms for a given nominal stem:
        - Consonant stems (5th order / Sadis): shift final consonant to 7th order + 'ች' (e.g. አሰልጣኝ -> አሰልጣኞች).
        - Vowel stems (orders 0-4): append '-ዎች'.
        - Includes accusative '-ን' variants.
        """
        if not stem:
            return []

        plurals = []
        last_c = stem[-1]
        _, order = self.get_char_info(last_c)

        if order in (0, 1, 2, 3, 4):
            pl_form = stem + "ዎች"
            plurals.append(pl_form)
            plurals.append(pl_form + "ን")
        elif order == 5:
            shifted_7th = self.shift_order(last_c, 6)
            p_base = stem[:-1] + shifted_7th + "ች"
            plurals.append(p_base)
            plurals.append(p_base + "ን")
            pl_form_w = stem + "ዎች"
            plurals.append(pl_form_w)
            plurals.append(pl_form_w + "ን")
        else:
            pl_form = stem + "ዎች"
            plurals.append(pl_form)
            plurals.append(pl_form + "ን")

        return plurals

    @staticmethod
    def get_prosthetic_variants(word: str) -> List[str]:
        """
        Generates phonotactically licensed prosthetic እ- variants for sonorant onsets (ረ-, ር-).
        In Amharic, words with liquid onsets optionally take an initial epenthetic/prosthetic
        'እ' in both speech and writing across all tenses (e.g. ርሳ <-> እርሳ, ረሳ <-> እረሳ).
        """
        if not word:
            return []

        variants = [word]
        if word.startswith(("ር", "ረ")):
            variants.append("እ" + word)
        elif word.startswith(("እር", "እረ")):
            variants.append(word[1:])

        # Preserve order and eliminate duplicates
        seen = set()
        out = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out
    def palatalize_coronal(self, char: str, target_order: int = 2) -> str:
        """
        Shifts a coronal consonant to its palatal counterpart in the target vowel order.
        For 2FS imperative (-i), target_order=2 (e.g. ዳ/ድ -> ጂ, ታ/ት -> ቺ).
        """
        base_c, _ = self.get_char_info(char)
        if char in self.CORONAL_TO_PALATAL_SERIES:
            palatal_base = self.CORONAL_TO_PALATAL_SERIES[char]
            return self.shift_order(palatal_base, target_order)
        if base_c in self.CORONAL_TO_PALATAL_SERIES:
            palatal_base = self.CORONAL_TO_PALATAL_SERIES[base_c]
            return self.shift_order(palatal_base, target_order)
        return self.shift_order(char, target_order)
        
    def is_potential_ideophone_shape(self, stem: str) -> bool:
        if not stem or len(stem) < 2:
            return False

        orders = self.get_vocalic_orders(stem)
        if not orders:
            return False

        if all(o == 5 for o in orders):
            return True

        if len(orders) == 3 and orders[0] == 0 and orders[1] == 0 and orders[2] == 5:
            return True

        if len(orders) == 3 and orders[0] == 3 and orders[1] == 0 and orders[2] == 5:
            return True
        if len(orders) == 2 and orders[0] == 3 and orders[1] == 5:
            return True

        if len(stem) in (4, 6):
            half = len(stem) // 2
            if stem[:half] == stem[half:]:
                return True
        if len(stem) >= 5 and stem[-2:] == stem[-4:-2]:
            return True

        return False

    @staticmethod
    @lru_cache(maxsize=4096)
    def shift_order(char: str, target_order: int) -> str:
        if not char or not (0 <= target_order < 8):
            return char
        if char in AmharicBoundaryPhonology.CHAR_MAP:
            s_idx, o_idx = AmharicBoundaryPhonology.CHAR_MAP[char]
            # Order 7 characters (Dikala labiovelars like ጯ, ቋ, ኳ, ጓ, ኋ) have inherent
            # labial co-articulation; shifting to order 0 must preserve their labial identity.
            if o_idx == 7 and target_order == 0:
                return char
            if target_order < len(AmharicBoundaryPhonology.GE_EZ_SERIES[s_idx]):
                shifted = AmharicBoundaryPhonology.GE_EZ_SERIES[s_idx][target_order]
                return shifted or char
        return char

    def invert_via_49_cell_matrix(self, word: str, ur_suffix: str) -> List[Tuple[str, str, str]]:
        results: List[Tuple[str, str, str]] = []
        if not word or not ur_suffix:
            return results

        if not (ur_suffix.startswith(("አ", "ኡ", "ኢ", "ኤ", "ኦ", "ኧ", "እ")) or ur_suffix in self.PURE_VOWEL_SUFFIXES):
            return results

        if ur_suffix in self.PURE_VOWEL_SUFFIXES and len(ur_suffix) == 1:
            boundary_char = word[-1]
            _, surface_order = self.get_char_info(boundary_char)
            _, sfx_order = self.get_char_info(ur_suffix)
            if sfx_order not in self.MATRIX_LOOKUP_SUFFIX:
                sfx_order = 5
            column_map = self.MATRIX_LOOKUP_SUFFIX.get(sfx_order, {})
            matching_orders = column_map.get(surface_order, [])
            for target_order in matching_orders:
                if target_order == surface_order:
                    continue
                restored_char = self.shift_order(boundary_char, target_order)
                restored_stem = word[:-1] + restored_char
                if restored_stem != word:
                    results.append((restored_stem, ur_suffix, f"pure_vowel_S{target_order}_V{sfx_order}"))
            return results

        if ur_suffix.startswith("አ") and len(ur_suffix) > 1:
            glide_tail = "ያ" + ur_suffix[1:]
            if word.endswith(glide_tail) and len(word) > len(glide_tail):
                stem_cand = word[: -len(glide_tail)]
                boundary_char = stem_cand[-1]
                _, boundary_order = self.get_char_info(boundary_char)
                if boundary_order in (2, 4):
                    results.append((stem_cand, ur_suffix, f"glide_epenthesis_y_{ur_suffix}"))

        sfx_initial_char = ur_suffix[0]
        _, sfx_vowel_order = self.get_char_info(sfx_initial_char)
        if sfx_vowel_order not in self.MATRIX_LOOKUP_SUFFIX:
            sfx_vowel_order = 5

        if ur_suffix in self.WEAK_DELETION_URS:
            surface_tail = self.WEAK_DELETION_URS[ur_suffix]
        elif sfx_vowel_order == 5:
            surface_tail = ur_suffix
            if surface_tail.endswith("እ") and len(surface_tail) > 1:
                surface_tail = surface_tail[:-1]
            elif surface_tail.startswith("እ") and len(surface_tail) > 1:
                surface_tail = surface_tail[1:]
        else:
            surface_tail = ur_suffix[1:] if len(ur_suffix) > 1 else ur_suffix

        if not word.endswith(surface_tail) or len(word) <= len(surface_tail):
            return results

        stem_cand = word[: -len(surface_tail)] if surface_tail else word
        if not stem_cand:
            return results

        boundary_char = stem_cand[-1]
        _, surface_order = self.get_char_info(boundary_char)
        column_map = self.MATRIX_LOOKUP_SUFFIX.get(sfx_vowel_order, {})
        matching_orders = column_map.get(surface_order, [])
        for target_order in matching_orders:
            restored_char = self.shift_order(boundary_char, target_order)
            restored_stem = stem_cand[:-1] + restored_char
            results.append((restored_stem, ur_suffix, f"matrix_cell_S{target_order}_V{sfx_vowel_order}"))
        return results

    def invert_palatalization(self, word: str) -> List[Tuple[str, str, str]]:
        results: List[Tuple[str, str, str]] = []
        if not word or len(word) < 2:
            return results

        last_char = word[-1]
        base_c, order = self.get_char_info(last_char)

        if last_char == "ቼ" and len(word) >= 3:
            stem_body = word[:-1]
            results.append((stem_body, "ቼ", "coverb_1s_che_direct"))
            if stem_body.endswith("ት") and len(stem_body) >= 2:
                weak_stem = stem_body[:-1]
                restored_weak = weak_stem[:-1] + self.shift_order(weak_stem[-1], 3)
                results.append((restored_weak, "ቼ", "coverb_1s_che_weak_t"))

        if order == 4 and base_c in self.PALATAL_CONSONANTS and len(word) >= 2:
            palatal_6th = self.shift_order(last_char, 5)
            if palatal_6th in self.PALATAL_TO_CORONAL:
                for coronal_6th in self.PALATAL_TO_CORONAL[palatal_6th]:
                    restored_stem = word[:-1] + coronal_6th
                    results.append((restored_stem, "ኤ", "palatal_coverb_1s_e"))

        if order == 2 and base_c in self.PALATAL_CONSONANTS and len(word) >= 2:
            palatal_6th = self.shift_order(last_char, 5)
            if palatal_6th in self.PALATAL_TO_CORONAL:
                for coronal_6th in self.PALATAL_TO_CORONAL[palatal_6th]:
                    restored_stem = word[:-1] + coronal_6th
                    results.append((restored_stem, "ኢ", "palatal_2fs_i"))

        # 2FS Imperative/Jussive Palatalization under suffix -i:
        # Sals (Order 2, e.g. ተበጂ, ውረጂ, ክፈቺ, ልበሺ, ለምኚ, ክፈዪ)
        if order == 2 and base_c in self.PALATAL_CONSONANTS and len(word) >= 2:
            palatal_6th = self.shift_order(last_char, 5)
            if palatal_6th in self.PALATAL_TO_CORONAL:
                for coronal_6th in self.PALATAL_TO_CORONAL[palatal_6th]:
                    restored_stem = word[:-1] + coronal_6th
                    results.append((restored_stem, "ኢ", "palatal_2fs_i"))

        # Sadis (Order 5, e.g. ተበጅ, ውረጅ, ክፈች, ልበሽ, ለምኝ, ክፈይ)
        if order == 5 and last_char in self.PALATAL_TO_CORONAL and len(word) >= 2:
            for coronal_6th in self.PALATAL_TO_CORONAL[last_char]:
                restored_stem = word[:-1] + coronal_6th
                results.append((restored_stem, "ኢ", "palatal_2fs_i_sadis"))

        return results

    def invert_coverb_3ms_o(self, word: str) -> List[Tuple[str, Union[str, Tuple[str, ...]], str]]:
        results: List[Tuple[str, Union[str, Tuple[str, ...]], str]] = []
        if not word or len(word) < 2:
            return results

        # This 3MS weak-coverb ending is an independent -ቶ suffix, not a
        # coda-shifted final radical plus -ኦ. Prefer the licensed weak path.
        if word.endswith("ቶ") and len(word) >= 3:
            results.append((word[:-1], "ቶ", "coverb_3ms_to_weak"))
            return results

        last_char = word[-1]
        # Labiovelar consonants (ጐ, ኰ, ቈ, ኈ) are root radicals, never 3MS converb suffixes
        if last_char in self.LABIOVELAR_MAP:
            return results

        _, order = self.get_char_info(last_char)

        if order == 6:
            c6 = self.shift_order(last_char, 5)
            bound_stem = word[:-1] + c6
            results.append((bound_stem, "ኦ", "coverb_3ms_o"))
        
        # Impersonal converbs with 3MS/3FS object enclitic -ት (ደንቆት, ቸግሮት, ነስሮት, ጨንቆት, ርቦት, ጠምቶት)
        if word.endswith("ት") and len(word) >= 3:
            penult_char = word[-2]
            _, penult_order = self.get_char_info(penult_char)
            if penult_order == 6:
                c6 = self.shift_order(penult_char, 5)
                bound_stem = word[:-2] + c6
                results.append((bound_stem, ("ኦ", "ት"), "coverb_3ms_o_obj_t"))
            elif word.endswith("ቶት") and len(word) >= 4:
                results.append((word[:-2], ("ቶ", "ት"), "coverb_3ms_to_obj_t"))

        return results

    def invert_coverb_3pl_w(self, word: str) -> List[Tuple[str, str, str]]:
        results: List[Tuple[str, str, str]] = []
        if not word or len(word) < 3:
            return results

        if word.endswith("ተው"):
            results.append((word[:-2], "ተው", "coverb_3pl_tew"))

        if word.endswith("ው"):
            # In roots of {ተ-ወ} (ተው, ተተው, ከተው) and {ለ-ወ} (ለው, ተለው), 'ው' is root radical C2, not 3PL coverb suffix
            if word in ("ተው", "ተተው", "ከተው", "ለው", "ተለው"):
                return results

            penult_char = word[-2]
            _, penult_order = self.get_char_info(penult_char)
            if penult_order == 0:
                c6 = self.shift_order(penult_char, 5)
                bound_stem = word[:-2] + c6
                results.append((bound_stem, "ው", "coverb_3pl_w"))

        return results

    def invert_labialized_coverb_auxiliary(
        self, word: str
    ) -> List[Tuple[str, Tuple[str, ...], str]]:
        """
        Dynamically inverts 3MS Present Perfect labialized verb codas (-Cʷal -> -C_6 + -o + -al).
        Generalizes across all 34 Ethiopic consonant series (e.g. ጿል -> ጽ, ኗል -> ን, ቋል -> ቅ).
        """
        results: List[Tuple[str, Tuple[str, ...], str]] = []
        if not word or len(word) < 3:
            return results

        # Impersonal Verbs with 3MS object clitic + auxiliary (e.g. ቸግሮታል -> ቸግር + ኦ + ት + አል, ጠምቶታል -> ጠም + ቶ + ት + አል)
        if word.endswith("ታል") and len(word) >= 4:
            if word.endswith("ቶታል") and len(word) >= 5:
                weak_base = word[:-3]
                results.append((weak_base, ("ቶ", "ት", "አል"), "compound_coverb_impersonal_total"))
            else:
                ante_penult_char = word[-3]
                _, ante_order = self.get_char_info(ante_penult_char)
                if ante_order == 6:  # 7th order Sabe (-o)
                    c6 = self.shift_order(ante_penult_char, 5)
                    bound_stem = word[:-3] + c6
                    results.append((bound_stem, ("ኦ", "ት", "አል"), "compound_coverb_impersonal_otal"))
                
        if word.endswith("ል"):
            # A compound coverb with auxiliary -አል never starts with subject concord prefix 'ይ-',
            # EXCEPT for the irregular converb forms of ያዘ: ይዞ/ይዛ/ይዘው -> ይዟል, ይዛለች, ይዘዋል
            if word.startswith("ይ") and not word.startswith(("ይዟ", "ይዛ", "ይዘ", "ይዞ")):
                return results

            # Weak verb compound coverb: -ቷል -> base + ("ቶ", "አል") (e.g. አሽካክቷል, አቅራርቷል, ተዝናንቷል)
            if word.endswith("ቷል") and len(word) >= 3:
                weak_base = word[:-2]
                results.append((weak_base, ("ቶ", "አል"), "compound_coverb_weak_to_al"))
                # Avoid a competing C7->C5 + -ኦ parse of the same weak -ቷል.
                return results

            penult_char = word[-2]
            _, penult_order = self.get_char_info(penult_char)
            # Labiovelars (ቋ, ኳ, ጓ, ኋ) map to order 3 in LABIOVELAR_MAP, so we check them explicitly
            if penult_order == 7:
                c6 = self.shift_order(penult_char, 5)
                bound_stem = word[:-2] + c6
                results.append((bound_stem, ("ኦ", "አል"), "compound_coverb_labialized_wal"))

        # 3PL Compound Converb (e.g. ተባብለዋል -> ተባብል + ው + አል)
        if word.endswith("ዋል") and len(word) >= 3:
            ante_penult_char = word[-3]
            _, ante_order = self.get_char_info(ante_penult_char)
            if ante_order == 0:  # 1st order Geez (-ä, e.g. ተባብለ-ዋል)
                c6 = self.shift_order(ante_penult_char, 5)
                bound_stem = word[:-3] + c6
                results.append((bound_stem, ("ው", "አል"), "compound_coverb_3pl_wal"))
            else:
                # LOOK AT THIS BOGUS ELSE BRANCH:
                direct_stem = word[:-2]
                if len(direct_stem) >= 2:
                    results.append((direct_stem, ("ው", "አል"), "compound_coverb_wal_direct"))

        return results

    def invert_coda_shift(
        self, word: str, suffix_tail: str, source_order: int, target_order: int = 5,
        block_palatals: bool = False, exception_endswith: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        # Auxiliary -al requires at least 4 characters (prefix + 2 radicals + aux: e.g. ይሆናል, ይመስላል).
        # 3-letter verbs like ይማል, ይሳል, ይቻል, ይጣል have radical 'ል', never auxiliary -al.
        # 3-letter verbs starting with ያ- (ያያል, ያጫል, ያሻል, ያዣል) are compound imperfectives of initial-a roots (አየ, አጨ, አሸ, አዠ)
        # Type A jussives with radical 'ል' start with ይ- (ይቻል, ይጣል, ይማል).
        if word.startswith("ያ") and len(word) == 3:
            min_len = len(suffix_tail) + 2
        else:
            min_len = len(suffix_tail) + 3 if suffix_tail == "ል" and source_order == 3 else len(suffix_tail) + 2

        if not word.endswith(suffix_tail) or len(word) < min_len:
            return results
        if exception_endswith and word.endswith(exception_endswith):
            return results
        stem_part = word[: -len(suffix_tail)]
        penult_char = stem_part[-1]
        
        # Prevent labiovelars (which emulate 3rd order Rabe) from being shifted by standard imperfective rules
        if source_order == 3 and penult_char in ("ቋ", "ኳ", "ጓ", "ኋ"):
            return results
            
        base_c, order = self.get_char_info(penult_char)
        if block_palatals and (penult_char in self.PALATAL_CONSONANTS or base_c in self.PALATAL_CONSONANTS):
            return results
        if order == source_order:
            restored = stem_part[:-1] + self.shift_order(penult_char, target_order)
            results.append((restored, f"coda_shift_{source_order}_to_{target_order}"))
        return results

    def invert_coda_shift_4th_to_6th(self, word: str, suffix_tail: str) -> List[Tuple[str, str]]:
        return self.invert_coda_shift(word, suffix_tail, source_order=3, target_order=5)

    def invert_coda_shift_1st_to_6th(self, word: str, suffix_tail: str) -> List[Tuple[str, str]]:
        return self.invert_coda_shift(
            word, suffix_tail, source_order=0, target_order=5,
            block_palatals=True, exception_endswith="ቸው"
        )

    def invert_rule_2_schwa_epenthesis(
        self, word: str, suffixes: Optional[List[str]] = None
    ) -> List[Tuple[str, str, str]]:
        results: List[Tuple[str, str, str]] = []
        sfx_list = suffixes or [
            "ተኝነትን", "ተኝነት", "ተኞችን", "ተኞች", "ተኛው", "ተኛ",
            "ኞችን", "ኞች", "ኛው", "ኛ"
        ]
        for sfx in sfx_list:
            if word.endswith(sfx) and len(word) > len(sfx):
                stem_cand = word[: -len(sfx)]
                if not stem_cand:
                    continue
                boundary_char = stem_cand[-1]
                _, order = self.get_char_info(boundary_char)
                if order in (0, 4):
                    rev_stem = stem_cand[:-1] + self.shift_order(boundary_char, 5)
                    results.append((rev_stem, sfx, "schwa_epenthesis_restored"))
                elif order in (3, 5):
                    results.append((stem_cand, sfx, "schwa_epenthesis_direct"))
        return results

    def invert_rule_5_2nd_order_plural_u(self, word: str) -> List[Tuple[str, str, str]]:
        results: List[Tuple[str, str, str]] = []
        if len(word) >= 2:
            boundary_char = word[-1]
            _, order = self.get_char_info(boundary_char)
            if order == 1:
                rev_0th = word[:-1] + self.shift_order(boundary_char, 0)
                results.append((rev_0th, "ኡ", "verb_plural_u_S0_perfective"))

                # Final-weak verbs ending in Ka'eb restore to Rabe (order 3) across all lengths (ጠጡ -> ጠጣ, ተሰጣጡ -> ተሰጣጣ)
                rev_4th = word[:-1] + self.shift_order(boundary_char, 3)
                results.append((rev_4th, "ኡ", "verb_plural_u_S4_weak"))

                if len(word) >= 3:
                    rev_6th = word[:-1] + self.shift_order(boundary_char, 5)
                    if word.startswith(("ይ", "ት", "እ", "እን", "ሚ")):
                        results.append((rev_6th, "ኡ", "verb_plural_u_S6_imperfective"))
                    else:
                        results.append((rev_6th, "ኡ", "verb_plural_u_S6_imperative"))
        return results

    # =========================================================================
    # UNIVERSAL DECLARATIVE MATRICES OF AMHARIC PREFIX + /a-/ COALESCENCE
    # =========================================================================

    # Class 1 & 2: 4th-order (Rabe) surface prefixes -> inverts to (Prefix + "አ" + tail)
    A_COALESCED_PREFIXES = (
        ("እስካ", ((("እስከ",), "pfx_prep_a_coalescence"),)),
        ("እንዳ", ((("እንደ",), "pfx_prep_a_coalescence"),)),
        ("ስላ",   ((("ስለ",), "pfx_prep_a_coalescence"),)),
        ("ወዳ",   ((("ወደ",), "pfx_prep_a_coalescence"),)),
        ("እያ",   ((("እየ",), "pfx_progressive_eyye_coalescence"),)),
        ("አላ",   ((("አል",), "pfx_neg_atomic_al"),)),
        ("አያ",   ((("አይ",), "pfx_neg_imp_ay_a"),)),
        ("ያልተ", ((("የ", "አል", "ተ"), "pfx_rel_neg_passive_ye_al_te"),)),
        ("ያል",  ((("የ", "አል"), "pfx_rel_neg_ye_al"),)),
        ("አን",   ((("አን",), "pfx_voice_an_a_coalescence"), (("እን",), "pfx_subj_en_a_coalescence"))),
        ("አት",   ((("አት",), "pfx_voice_at_a_coalescence"),)),
        ("ባ",   ((("በ",), "pfx_prep_a_coalescence"), (("ብ",), "pfx_subord_b_a_coalescence"))),
        ("ላ",   ((("ለ",), "pfx_prep_a_coalescence"), (("ል",), "pfx_subord_l_a_coalescence"))),
        ("ካ",   ((("ከ",), "pfx_prep_a_coalescence"),)),
        ("ታ",   ((("ተ",), "pfx_voice_passive_te_a_coalescence"), (("ት",), "pfx_subj_t_a_coalescence"))),
        ("ና",   ((("እን",), "pfx_subj_en_a_coalescence"),)),
        ("ሳ",   ((("ስ",), "pfx_subord_s_a_coalescence"),)),
        ("ያስ",  ((("ይ", "አስ"), "pfx_imp_yi_as_coalescence"), (("የ", "አስ"), "pfx_rel_ye_as_coalescence"))),
        ("ያሥ",  ((("ይ", "አስ"), "pfx_imp_yi_as_coalescence"), (("የ", "አስ"), "pfx_rel_ye_as_coalescence"))),
        ("ያን",  ((("ይ", "አን"), "pfx_imp_yi_an_coalescence"), (("የ", "አን"), "pfx_rel_ye_an_coalescence"))),
        ("ያ",   ((("ይ",), "pfx_imp_yi_a_coalescence"), (("የ",), "pfx_rel_ye_a_coalescence"))),
    )

    # Class 3: 3rd-order (Sals) prefixes with palatal glide epenthesis -> inverts to (Prefixes + "አ" + tail)
    GLIDE_COALESCED_PREFIXES = (
        ("እንደሚያ", ((("እንደ", "እም", "ይ"), "rel_glide_endemiya"),)),
        ("ስለሚያ",   ((("ስለ", "እም", "ይ"), "rel_glide_slemiya"),)),
        ("በሚያ",     ((("በ", "እም", "ይ"), "rel_glide_bemiya"),)),
        ("ከሚያ",     ((("ከ", "እም", "ይ"), "rel_glide_kemiya"),)),
        ("ለሚያ",     ((("ለ", "እም", "ይ"), "rel_glide_lemiya"),)),
        ("የሚያ",     ((("የ", "እም", "ይ"), "rel_glide_yemiya"),)),
        ("የምታ",     ((("የ", "እም", "ት"), "rel_agr_yemta"),)),
        ("የምና",     ((("የ", "እም", "ን"), "rel_agr_yemna"),)),
        ("እንዲያ",   ((("እንድ", "ይ"), "glide_vocal_endiya"),)),
        ("እስኪያ",   ((("እስክ", "ይ"), "glide_vocal_eskiya"),)),
        ("ሲያ",       ((("ስ", "ይ"), "glide_vocal_siya"),)),
        ("ቢያ",       ((("ብ", "ይ"), "glide_vocal_biya"),)),
        ("ሊያ",       ((("ል", "ይ"), "glide_vocal_liya"),)),
        ("ኪያ",       ((("ክ", "ይ"), "glide_vocal_kiya"),)),
        ("ሚያ",       ((("እም", "ይ"), "rel_glide_miya"),)),
        ("ምታ",       ((("እም", "ት"), "rel_agr_mta"),)),
        ("ምና",       ((("እም", "ን"), "rel_agr_mna"),)),
    )

    def invert_prefix_boundary(
        self,
        word: str,
        rel_licensors: Optional[List[str]] = None,
        neg_prefixes: Optional[List[str]] = None,
    ) -> List[Tuple[str, Union[str, Tuple[str, ...]], str]]:
        results: List[Tuple[str, Union[str, Tuple[str, ...]], str]] = []
        if not word or len(word) < 2:
            return results

        # 1. Universal Glide Epenthesis Coalescence (Prefixes + "አ" + tail)
        for surf_pfx, mappings in self.GLIDE_COALESCED_PREFIXES:
            if word.startswith(surf_pfx) and len(word) > len(surf_pfx):
                restored_stem = "አ" + word[len(surf_pfx):]
                for pfx_tuple, rule_name in mappings:
                    results.append((restored_stem, pfx_tuple, rule_name))

        # 2. Universal 4th-Order (Rabe) Coalescence (Prefix + "አ" + tail)
        # Universal across nouns, proper names, adjectives, and verbs.
        for surf_pfx, mappings in self.A_COALESCED_PREFIXES:
            if word.startswith(surf_pfx) and len(word) > len(surf_pfx):
                # Only skip if the 4-char relative negative passive 'ያልተ' already handled it
                if surf_pfx == "ያ" and word.startswith("ያልተ"):
                    continue
                restored_stem = "አ" + word[len(surf_pfx):]
                for pfx_tuple, rule_name in mappings:
                    results.append((restored_stem, pfx_tuple[0] if len(pfx_tuple) == 1 else pfx_tuple, rule_name))

         # 3. Non-Coalesced Concord / Subordinators (Following consonant is NOT /a-/)
        # Exposes the hidden 3MS subject concord 'ይ' (e.g., ሊ = ል + ይ)
        simple_triggers = {
            "ሊ": ("ል", "ይ"), 
            "ሲ": ("ስ", "ይ"), 
            "ቢ": ("ብ", "ይ"), 
            "ኪ": ("ክ", "ይ")
        }
        for surf_pfx, pfx_tuple in simple_triggers.items():
            if word.startswith(surf_pfx) and not word.startswith(surf_pfx + "ያ") and len(word) > 1:
                results.append((word[1:], pfx_tuple, f"modal_prefix_{surf_pfx}"))

        if word.startswith("እንዲ") and not word.startswith("እንዲያ") and len(word) > 2:
            results.append((word[2:], ("እንድ", "ይ"), "glide_vocal_endi"))
        elif word.startswith("እንድ") and len(word) > 3:
            results.append((word[3:], "እንድ", "pfx_purposive_end"))

        if word.startswith("እስኪ") and not word.startswith("እስኪያ") and len(word) > 3:
            results.append((word[3:], ("እስክ", "ይ"), "glide_vocal_eski"))

        if word.startswith("ልት") and len(word) > 2:
            results.append((word[2:], ("ል", "ት"), "pfx_modal_lt"))
        elif word.startswith("ስት") and len(word) > 2:
            results.append((word[2:], ("ስ", "ት"), "pfx_modal_st"))
        elif word.startswith("ብት") and len(word) > 2:
            results.append((word[2:], ("ብ", "ት"), "pfx_modal_bt"))
        elif word.startswith("እንድት") and len(word) > 4:
            results.append((word[4:], ("እንድ", "ት"), "pfx_modal_endet"))

        # Relative imperfective non-glide markers
        licensors = sorted(
            rel_licensors or ["እስከ", "ወደ", "እንደ", "ስለ", "የ", "በ", "ከ", "ለ"],
            key=len, reverse=True
        )
        for lic in licensors + [""]:
            if word.startswith(lic + "ሚ") and not word.startswith(lic + "ሚያ") and len(word) > len(lic) + 1:
                results.append((word[len(lic) + 1:], (lic, "እም", "ይ") if lic else ("እም", "ይ"), f"rel_glide_{lic}mi"))
            elif word.startswith(lic + "ምን") and not word.startswith(lic + "ምና") and len(word) > len(lic) + 2:
                results.append((word[len(lic) + 2:], (lic, "እም", "ን") if lic else ("እም", "ን"), f"rel_agr_{lic}mn"))
            elif word.startswith(lic + "ምት") and not word.startswith(lic + "ምታ") and len(word) > len(lic) + 2:
                results.append((word[len(lic) + 2:], (lic, "እም", "ት") if lic else ("እም", "ት"), f"rel_agr_{lic}mt"))

        if word.startswith("እየ") and len(word) > 2:
            results.append((word[2:], "እየ", "pfx_progressive_eyye_direct"))

        for p_passive in ("እንደተ", "ስለተ", "የተ", "በተ", "ከተ", "ለተ"):
            if word.startswith(p_passive) and len(word) > len(p_passive):
                core_lic = p_passive[:-1]
                results.append((word[len(p_passive) - 1:], (core_lic, "ተ"), f"pfx_passive_{core_lic}_te"))

        negs = sorted(
            neg_prefixes or [
                "እንዳይ", "ስለማይ", "የማይ", "ከማይ", "በማይ", "ለማይ",
                "አይ", "አት", "አን", "ሳይ", "አል"
            ],
            key=len, reverse=True
        )
        for neg_pfx in negs:
            if word.startswith(neg_pfx) and len(word) > len(neg_pfx):
                results.append((word[len(neg_pfx):], neg_pfx, f"pfx_neg_atomic_{neg_pfx}"))

        return results