# Corrected paper artifacts

This bundle uses the 941 accepted corrected writer replays, the 300 hash-verified independent baselines, and the 1,184 preserved pre-writer terminal outcomes. Writer-dependent exhibits are regenerated from `analysis_source/`. T3 is copied from the frozen bundle because it uses the independent C and C2 baselines. T5 preserves its upstream topic-fit values while separating deterministic input gates, substantive refusals, and structured failures.

H3 is excluded because it used the original writer. The cost and latency figure is omitted until a comparable timing construction is defined.

Rebuild from the repository root with:

```bash
python scripts/build_corrected_paper_artifacts.py
```
