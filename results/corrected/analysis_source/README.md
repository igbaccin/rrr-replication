# Corrected paper analysis source

This directory contains the paired July and corrected writer results used to
map changes before manuscript revision. The full join key is `(batch,
condition, run)`. The frozen July paper bundle remains under
`private/artifacts/pre_correction_snapshot/`.

Rebuild from the repository root with:

```powershell
.\.venv\Scripts\python.exe scripts\build_corrected_results_change_map_vlm25.py --corrected-root pod_outputs\battery_vlm25_materialized --out private\artifacts\vlm25_results\analysis_source --report revision_notes\vlm25_results_change_map.md
```

`validation.json` records the required count and historical-reproduction
checks. The narrative change map is
`revision_notes/corrected_results_change_map.md`.
