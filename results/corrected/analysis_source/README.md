# Accepted paper analysis source

This directory contains the run-level analytical records used to generate the
accepted paper exhibits. The records account for 941 attempts that reached the
writer, 300 independent baselines, and 1,184 outcomes that ended before the
writer. The full join key for paired records is `(batch, condition, run)`.

`validation.json` records the count and historical-reproduction checks. The
supported exhibit builder reads this directory directly:

```bash
python scripts/build_corrected_paper_artifacts.py --out reproduced_artifacts
```

The original preparation paths retained in `manifest.json` document how this
analytical snapshot was produced. They are provenance fields and are not
required for reviewer verification or exhibit regeneration.
