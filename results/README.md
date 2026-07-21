# Deposited results

`corrected/` is the accepted analytical and exhibit bundle. Its manifest records 941 corrected writer replays, 300 independent baselines, and 1,184 pre-writer terminal outcomes.

`replay_inputs/` contains the compressed frozen ledgers required to regenerate the corrected writer outputs. The archive expands under the ignored runtime directory `replay_inputs/`.

`external_comparisons/claude_code/` contains the completed off-the-shelf comparison summaries. `external_comparisons/notebooklm_pilot/` is provisional and does not represent a completed comparison arm.

`rebuild_inputs/pre_correction_snapshot/` contains labelled table inputs used by the corrected exhibit builder for invariant labels and writer-independent cells. These files are rebuild dependencies. The accepted values are under `corrected/`.
