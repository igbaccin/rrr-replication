# Corrected paper analysis source

This directory contains the paired July and corrected writer results used to map changes before manuscript revision. The full join key is `(batch, condition, run)`.

The deposited files are the accepted run-level analytical source. Regenerate the paper tables and figures from the repository root with:

```bash
python scripts/build_corrected_paper_artifacts.py --out reproduced_artifacts
```

`validation.json` records the required population, historical-reproduction, and writer-contract checks. The frozen writer inputs are deposited as `results/replay_inputs/writer_correction_260720.tar.gz`.
