# 5_correction_evaluation/data — last audit outputs

Produced by `02_collect_error_cases.py` from the 150-page benchmark run
(the older corrector revision — see `docs/current_state.md` §5.1).

| File | Size | Rows |
|---|---:|---:|
| `correction_decisions.tsv` | 3.1 MB | 21,927 token events (976 `TRUE_POSITIVE`, 1,348 `FALSE_POSITIVE`, 1,370 `SUBOPTIMAL_FLIP`, 18,233 `FALSE_NEGATIVE`) |
| `failed_cases.tsv` | 712 KB | 7,979 (`OVER_CORRECTION`, `UNREPAIRED_TYPO`, …) |

The categories are derived from optimal word alignments (see
`docs/decision_audit.md`); they are estimates of repair/damage, not causal
provenance. Regenerate them against a fresh run before drawing conclusions.
