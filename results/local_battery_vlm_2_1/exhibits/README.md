# vlm.2.1 paper exhibits (regenerated)

Tables and figures regenerated from the vlm.2.1 battery via the standard
builders (`build_corrected_results_change_map` + `build_corrected_paper_artifacts`),
run on the materialized normalized tree. Every value was cross-checked against
`../per_run.csv` before deposit.

Key changes vs the v17 exhibits (in `results/corrected/`, kept for reference):
- T1 main integrity: 100/100 -> 99/100 zero-E1-E5; citations/run 31.8 -> 47.7;
  body words 1682 -> 2064 (paper WORD_RE metric).
- T2/robustness: f1 98->100, f2 95->98, gender 29->30, colonial 28->29.
- Validation-gate ablation (B): 95% -> 85% clean (the denser-citing writer
  leaks more when validation is off; the full-RRR vs ablation contrast widens
  from 5 to 14 points).
- F5 model ladder: attempts denominator, recombined with the 1,184 pre-writer
  terminals (the correct end-to-end view); large weak-model gains (qwen3:4b,
  mistral:7b).

Writer-independent exhibits (B2/C/C2 baselines, H arms, H3, corpus size,
refusal calibration) carry over unchanged.

These are DATA artifacts. Prose integration into the manuscript is the author's.
