# Reproducing the deposited RRR evaluation

[About RRR](../README.md) · [Installation modes](product_installation.md) ·
[Before running](before_running.md)

This repository contains the accepted vlm.2.1 analytical records, paper tables
and figures, raw generated reviews, frozen replay inputs, and external
comparison summaries.

## Fast verification

The deposited package can be checked without the copyrighted corpus or a
language model:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python scripts/verify_replication.py
python -m unittest discover -s tests/unit -v
```

Regenerate the accepted tables and figures from the deposited analytical
source:

```bash
python scripts/build_corrected_paper_artifacts_vlm21.py
```

The command writes `reproduced_artifacts/`. Its validation requires all 941
writer-stage attempts and all 1,184 pre-writer outcomes to be fully accounted
for. It also checks the 300 independent baselines, the 60 model-by-probe rows,
and the 33 writer failures in the model ladder.

## Deposited evaluation contents

| Path | Contents |
| --- | --- |
| `results/local_battery_vlm_2_1/` | Accepted battery counts, raw reviews, and normalization reproducer |
| `results/local_battery_vlm_2_1/exhibits/analysis_source/` | Validated run-level analytical records |
| `results/local_battery_vlm_2_1/exhibits/tables/` | Accepted tables in CSV and TeX formats |
| `results/local_battery_vlm_2_1/exhibits/figures/` | Accepted figures in PDF and PNG formats |
| `results/corrected/` | Historical v17 artifact bundle and the accepted corrected H3 record |
| `results/replay_inputs/` | Compressed frozen ledgers for writer replay |
| `results/external_comparisons/` | Claude Code summaries and provisional NotebookLM pilot |
| `results/rebuild_inputs/` | Frozen labels and writer-independent table inputs |
| `REPLICATION_MANIFEST.json` | Package status, populations, exclusions, and integrity hashes |

## Result population

The complete analytical population contains 2,425 attempts.

| Population | Attempts | Accepted treatment |
| --- | ---: | --- |
| Attempts that reached the writer | 941 | 887 success, 20 checker-flagged, 34 writer failures |
| Independent B2, C, and C2 baselines | 300 | Preserved with verified hashes |
| Attempts ending before the writer | 1,184 | Preserved with their recorded terminal states |
| Total | 2,425 | Fully classified |

Of the 941 writer-stage attempts, 907 produced review files. The main condition
contains 99 integrity-clean reviews in 100 attempts. Across
validation-enabled conditions, 553 reviews were released and all 553 are free
of E1 through E5 under the automated checker.

The model table and graph use 105 attempted runs per ladder model. They combine
the writer-stage outcomes with the earlier terminal outcomes and display writer
failures separately.

## Reproducing the normalization

Install the package and run:

```bash
python results/local_battery_vlm_2_1/reproduce_normalization.py
```

The script applies the deterministic vlm.2.1 citation normalization to all 907
raw generated reviews. It reports 49 changed review files. `per_run.csv`
records 41 checker-failed runs that become successful after normalization.

Complete E1 through E5 rescoring requires the source corpus. Most PDFs are
copyrighted and must be obtained legally by the replicator.

## Replaying the writer

Extract the frozen ledgers:

```bash
mkdir -p replay_inputs
tar -xzf results/replay_inputs/writer_correction_260720.tar.gz -C replay_inputs
```

List the smoke profile:

```bash
python scripts/run_corrected_writer_replay.py \
  --bundle replay_inputs/corrected_writer_v17 \
  --profile smoke \
  --list
```

The full replay requires the reconstructed 50-document corpus and the recorded
local models. Language-model generation is stochastic, so regenerated prose
and secondary metrics may vary.

## External comparisons

The completed direct Claude Code comparisons cover Haiku 4.5, Sonnet 4.5, and
Opus 4.8. Their summaries are under
`results/external_comparisons/claude_code/`.

The corrected `/rrr` skill condition is stored under
`results/corrected/external_comparisons/rrr_skill/`. All ten runs are clean
under E1 through E5. The NotebookLM directory contains one provisional pilot
and does not constitute a completed comparison arm.

## Reproducibility boundaries

- Corpus PDFs must be obtained by the replicator.
- Local model tags preserve the tested model identities.
- Language-model outputs are stochastic.
- Corrected writer timings are excluded because they are not comparable with
  the original full-pipeline timings.
- Historical commit identifiers embedded in replay records remain provenance
  references to the private development history.
