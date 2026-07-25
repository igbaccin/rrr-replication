# Deposited results

`local_battery_vlm_2_0/` is the current local battery: 941 writer replays on
the vlm.2.0 writer (fingerprint `7297d941...`, Ollama 0.32.3). It holds the
scored outcomes computed directly from the outputs — `summary.json` (overall,
per-model, per-condition), `per_run.csv` (one row per run), and
`conditions/*.json`. Headline: mistral-small:24b 647/660 success, 631 of those
with zero hard E1-E5, zero stub closings. Weak models fail far more than under
the prior generation because the citation-integrity gate now fails malformed
citations instead of shipping them as "success"; report the ladder separately.

`corrected/` is the PRIOR (v17) generation's full paper exhibit suite — all
tables and figures. The new battery supersedes only its writer-dependent cells
(integrity, conditions, citation taxonomy, local-model outcomes, word counts).
Its writer-independent exhibits (controlled staircase baselines, refusal
calibration, corpus size, stability, the H3 Claude comparison) are unaffected.
Regenerating the writer-dependent exhibits from the new battery is a pending
analytical step: the exhibit builder requires every replay to be "success", and
the new battery has 95 honest weak-model failures whose treatment in the paper
is an author decision.


`corrected/` is the accepted analytical and exhibit bundle. Its manifest records 941 corrected writer replays, 300 independent baselines, and 1,184 pre-writer terminal outcomes.

`replay_inputs/` contains the compressed frozen ledgers required to regenerate the corrected writer outputs. The archive expands under the ignored runtime directory `replay_inputs/`.

`external_comparisons/claude_code/` contains the completed off-the-shelf comparison summaries. `external_comparisons/notebooklm_pilot/` is provisional and does not represent a completed comparison arm.

`rebuild_inputs/pre_correction_snapshot/` contains labelled table inputs used by the corrected exhibit builder for invariant labels and writer-independent cells. These files are rebuild dependencies. The accepted values are under `corrected/`.
