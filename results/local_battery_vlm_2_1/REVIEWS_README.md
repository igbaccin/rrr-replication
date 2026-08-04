# Raw reviews and reproduction

This directory preserves the as-released vlm.2.1 record for provenance. Its
stored outcomes predate the checker rule that counts any unrendered internal
evidence identifier as E3. Current accepted scores and exhibits are under
`results/corrected/`.

The battery contains 941 writer-stage attempts. The directory
`reviews_raw/<batch>/<condition>/<run>.md` contains the 907 attempts that
produced a review: 887 successful reviews and 20 checker-flagged reviews. The
remaining 34 attempts ended in writer failure and produced no review file.
These are the exact artifacts to which the vlm.2.1 citation normalizer was
applied.

`vlm.2.1` differs from `vlm.2.0` only by a deterministic final-assembly pass that
canonicalizes internal document IDs leaked into prose (`(Doc_Year)`,
`(paper Doc_Year, page N)`, `(Doc_Year (Author Year, p.N))`). It runs downstream
of every model call. Applying it to these frozen reviews therefore reproduces
the released review texts without rerunning a model.

To reproduce the deterministic text transformation:

```bash
python -m pip install .
python results/local_battery_vlm_2_1/reproduce_normalization.py
```

The script reports the 49 review files changed by the normalizer. The transform
is gated by each run's allowed documents in `allowed_docs.json`.

`per_run.csv` records the before-and-after outcome in `outcome_asrun` (vlm.2.0)
and `outcome` (vlm.2.1). It contains 41 rows that move from `checker_failed` to
`success` under the historical checker. Reproducing the complete current E1
through E5 check requires the source
corpus, including copyrighted documents that replicators must obtain
separately.

## Denominators

The historical per-model counts in `summary.json` describe attempts that
reached the writer under the as-released checker. The 1,184 attempts that
refused or failed earlier are preserved separately. The model table and graph
under `exhibits/` combine both groups and use all 105 attempted runs per ladder
model. This first-generation exhibit snapshot is retained for provenance.
Current accepted exhibits are under `results/corrected/`.
