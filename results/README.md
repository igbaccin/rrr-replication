# Deposited results

`corrected/` is the accepted vlm.2.1 analytical and exhibit bundle. Its
manifest records 941 writer-stage attempts, of which 907 produced reviews and
34 ended in writer failure, together with 300 independent baselines and 1,184
pre-writer terminal outcomes.

`replay_inputs/` contains the compressed frozen ledgers required to regenerate the corrected writer outputs. The archive expands under the ignored runtime directory `replay_inputs/`.

`external_comparisons/claude_code/` contains the completed off-the-shelf
comparison summaries. `external_comparisons/notebooklm/` contains the five
completed NotebookLM records, the scoring sheet, and the separate five-claim
page check.

`rebuild_inputs/pre_correction_snapshot/` contains labelled table inputs used by the corrected exhibit builder for invariant labels and writer-independent cells. These files are rebuild dependencies. The accepted values are under `corrected/`.
