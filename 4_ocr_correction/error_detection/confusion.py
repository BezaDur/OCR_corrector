# confusion.py — Loads confusion_matrix.json.
#
# Reads the file and builds the reverse substitution map
# (what character could have been misread as what).
import collections
import json
from typing import Dict, List, Tuple


class ConfusionMatrix:
    """Loads confusion_matrix.json and provides cost lookup + reverse sub map."""

    def __init__(self, path: str, max_reverse_sub_cost: float = 4.0):
        with open(path, "r", encoding="utf-8") as f:
            self._raw = json.load(f)

        self.default_sub = float(self._raw.get("default_sub", 4.5))
        self.default_del = float(self._raw.get("default_del", 2.4))
        self.default_ins = float(self._raw.get("default_ins", 2.4))

        # Precompute reverse optical substitutions.
        # Keys: observed character -> list of (true_char, cost).
        # Gated by max_reverse_sub_cost (component-level, NOT the pairwise sum).
        # FIXED: now handles both "sub" dict format and old "substitutions" list format
        self.reverse_sub_map: Dict[str, List[Tuple[str, float]]] = \
            collections.defaultdict(list)

        # New format: {"sub": {"true_char": {"observed_char": cost}}}
        if "sub" in self._raw:
            for true_ch, obs_map in self._raw.get("sub", {}).items():
                if isinstance(obs_map, dict):
                    for obs_ch, cost in obs_map.items():
                        try:
                            c_flt = float(cost)
                        except Exception:
                            c_flt = self.default_sub
                        if c_flt <= max_reverse_sub_cost and true_ch != obs_ch:
                            self.reverse_sub_map[obs_ch].append((true_ch, c_flt))
                elif isinstance(obs_map, list):
                    # Alternative: list of observed chars
                    for obs_ch in obs_map:
                        if true_ch != obs_ch:
                            self.reverse_sub_map[obs_ch].append((true_ch, 2.2))

        # Old format: {"substitutions": [{"target":..., "ocr":..., "cost":...}]}
        if "substitutions" in self._raw:
            for entry in self._raw.get("substitutions", []):
                if not isinstance(entry, dict):
                    continue
                true_ch = entry.get("target") or entry.get("true") or entry.get("gt")
                obs_ch = entry.get("ocr") or entry.get("observed") or entry.get("pred")
                cost = entry.get("cost", 2.2)
                if not true_ch or not obs_ch:
                    continue
                try:
                    c_flt = float(cost)
                except Exception:
                    c_flt = 2.2
                if c_flt <= max_reverse_sub_cost and true_ch != obs_ch:
                    self.reverse_sub_map[obs_ch].append((true_ch, c_flt))

    @property
    def sub_table(self) -> dict:
        return self._raw.get("sub", {})

    @property
    def ins_table(self) -> dict:
        return self._raw.get("ins", {})

    @property
    def del_table(self) -> dict:
        return self._raw.get("del", {})

    def ins_cost(self, ch: str, cap: float = 2.4) -> float:
        raw = float(self.ins_table.get(ch, self.default_ins))
        return min(raw, cap) if cap is not None else raw

    def del_cost(self, ch: str, cap: float = 2.4) -> float:
        raw = float(self.del_table.get(ch, self.default_del))
        return min(raw, cap) if cap is not None else raw

    def sub_cost(self, true_ch: str, obs_ch: str) -> float:
        if true_ch == obs_ch:
            return 0.0
        row = self.sub_table.get(true_ch, {})
        val = row.get(obs_ch)
        return self.default_sub if val is None else float(val)