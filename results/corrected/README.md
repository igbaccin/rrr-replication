# vlm.2.5 paper artifacts (vlm.2.5 materialized on the vlm.2.4 generation)

This bundle uses 941 writer-stage attempts, 300 independent baselines, and 1,184 preserved pre-writer outcomes. Writer-stage outcome totals are in `analysis_source/overall_writer_changes.csv`. Writer-dependent exhibits are regenerated from `analysis_source/`.

The analytical source passes 21 checks. This artifact bundle passes ten additional checks, including complete attempted-run accounting across all 60 model-by-probe rows and all 40 writer failures in the model ladder. T3 is copied from the frozen bundle because it uses the independent C and C2 baselines.

The historical H3 condition is excluded because it used the original writer. Its corrected replacement contains ten completed and scored runs, all free of E1--E5 under the common checker. The cost and latency figure is omitted until a comparable timing construction is defined.

Rebuild from the repository root with:

```bash
python scripts/build_corrected_paper_artifacts.py
```
