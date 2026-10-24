# 5_correction_evaluation/benchmarks — benchmark reports

`benchmark_full_symspell.json` (160 KB) — the 150-page, 63,518-word benchmark.
Headline: strict CER 1.178 % → 1.930 %, canonical CER 1.059 % → 1.140 %,
clean-token damage 4.06 % strict / 2.66 % canonical, character-error reduction
−63.8 % strict / −7.7 % canonical.

**Provenance matters:** it was produced by an earlier corrector revision (its
`options` block lists `candidate_backend`, `trie_search`, `trusted_*_frequency`,
none of which exist in HEAD). It is not a measurement of the current code.
`docs/current_state.md` §5.1 has the table and the caveats; re-run Phase B
(`docs/two_phase_benchmark.md`) to get a number that describes HEAD.
