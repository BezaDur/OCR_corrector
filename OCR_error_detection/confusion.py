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
        self.reverse_sub_map: Dict[str, List[Tuple[str, float]]] = \
            collections.defaultdict(list)
        for true_ch, obs_map in self._raw.get("sub", {}).items():
            if isinstance(obs_map, dict):
                for obs_ch, cost in obs_map.items():
                    c_flt = float(cost)
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