#!/usr/bin/env python3
"""
confusion_matrix_coverage_audit.py
===================================
Audits confusion_matrix.json for coverage of "same-consonant, different-order"
confusions across the Ethiopic syllabary — the single most common class of
Amharic OCR substitution error (e.g. ር/ሮ, ሁ/ሆ, ለ/ሎ, ...).

Grouping heuristic: within the core Ethiopic block (U+1200-U+137F), each
consonant family occupies a contiguous run of (up to) 8 codepoints
representing orders 1-7 plus a reserved/labialized slot. This is a standard
heuristic, not 100% perfect for every extended/labialized family, but it
correctly covers the vast majority of the syllabary and is more than
sufficient for a coverage audit.

Usage:
  python3 confusion_matrix_coverage_audit.py --confusion-matrix confusion_matrix.json
  python3 confusion_matrix_coverage_audit.py --confusion-matrix confusion_matrix.json --generate-missing out_patch.json
"""

import json
import argparse
import unicodedata
import collections

ETHIOPIC_BLOCK_START = 0x1200
ETHIOPIC_BLOCK_END = 0x137F
FAMILY_WIDTH = 8  # orders 1-7 + 1 reserved/labialized slot, per family


def get_family_and_order(ch: str):
    cp = ord(ch)
    if not (ETHIOPIC_BLOCK_START <= cp <= ETHIOPIC_BLOCK_END):
        return None, None
    offset = cp - ETHIOPIC_BLOCK_START
    family = offset // FAMILY_WIDTH
    order = offset % FAMILY_WIDTH  # 0..6 = orders 1-7, 7 = reserved/labialized
    return family, order


def build_families():
    families = collections.defaultdict(dict)  # family_id -> {order: char}
    for cp in range(ETHIOPIC_BLOCK_START, ETHIOPIC_BLOCK_END + 1):
        ch = chr(cp)
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue  # unassigned codepoint
        if "ETHIOPIC SYLLABLE" not in name:
            continue
        family, order = get_family_and_order(ch)
        if order is not None and order < 7:  # skip reserved slot
            families[family][order] = ch
    return families


def audit(confusion_matrix_path: str, generate_missing_path: str = None):
    with open(confusion_matrix_path, "r", encoding="utf-8") as f:
        costs = json.load(f)
    sub_table = costs.get("sub", {})

    families = build_families()

    total_pairs = 0
    covered_pairs = 0
    missing_pairs = []

    for family_id, order_map in families.items():
        chars = list(order_map.values())
        if len(chars) < 2:
            continue
        for i, true_ch in enumerate(chars):
            for j, obs_ch in enumerate(chars):
                if i == j:
                    continue
                total_pairs += 1
                cost = sub_table.get(true_ch, {}).get(obs_ch)
                if cost is not None:
                    covered_pairs += 1
                else:
                    missing_pairs.append((true_ch, obs_ch, family_id))

    print("=" * 70)
    print("ETHIOPIC SAME-FAMILY ORDER-CONFUSION COVERAGE AUDIT")
    print("=" * 70)
    print(f"Total directional order-pairs checked : {total_pairs}")
    print(f"Covered in confusion_matrix.json      : {covered_pairs}  "
          f"({100*covered_pairs/total_pairs:.1f}%)")
    print(f"MISSING entirely                      : {len(missing_pairs)}  "
          f"({100*len(missing_pairs)/total_pairs:.1f}%)")

    # Show which families are worst-affected
    missing_by_family = collections.Counter(f for _, _, f in missing_pairs)
    print(f"\nFamilies with missing coverage (family_id: n_missing_pairs):")
    for fam, n in sorted(missing_by_family.items(), key=lambda x: -x[1])[:15]:
        sample_chars = list(families[fam].values())
        print(f"  family {fam:3d} {''.join(sample_chars)!r:12} -> {n} missing directional pairs")

    print(f"\nSample of missing pairs (first 25):")
    for true_ch, obs_ch, fam in missing_pairs[:25]:
        print(f"  true={true_ch!r} obs={obs_ch!r}  (family {fam})")

    if generate_missing_path:
        # Emit a *patch* confusion matrix with a conservative default cost for
        # every missing same-family order pair, suitable for manual review
        # and merging into confusion_matrix.json.
        FALLBACK_ORDER_CONFUSION_COST = 2.0  # tune based on your existing cost scale
        patch = {"sub": {}}
        for true_ch, obs_ch, _ in missing_pairs:
            patch["sub"].setdefault(true_ch, {})[obs_ch] = FALLBACK_ORDER_CONFUSION_COST
        with open(generate_missing_path, "w", encoding="utf-8") as f:
            json.dump(patch, f, ensure_ascii=False, indent=2)
        print(f"\nPatch file with {len(missing_pairs)} fallback entries written to: "
              f"{generate_missing_path}")
        print(f"Review the fallback cost ({FALLBACK_ORDER_CONFUSION_COST}) against your "
              f"existing cost scale before merging — same-order-family confusions "
              f"should generally be CHEAPER than your default_sub ({costs.get('default_sub')}), "
              f"since they are visually closer than arbitrary substitutions.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confusion-matrix", required=True)
    parser.add_argument("--generate-missing", default=None,
                         help="Optional path to write an auto-generated patch JSON "
                              "for all missing same-family order pairs")
    args = parser.parse_args()
    audit(args.confusion_matrix, args.generate_missing)


if __name__ == "__main__":
    main()
