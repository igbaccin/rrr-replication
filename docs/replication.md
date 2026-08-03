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
python scripts/build_corrected_paper_artifacts.py --out reproduced_artifacts
```

The command writes `reproduced_artifacts/`. Its validation requires all 941
writer-stage attempts and all 1,184 pre-writer outcomes to be fully accounted
for. It also checks the 300 independent baselines, the 60 model-by-probe rows,
and the 33 writer failures in the model ladder.

## Deposited evaluation contents

| Path | Contents |
| --- | --- |
| `results/corrected/analysis_source/` | Run-level analytical records used to construct the exhibits |
| `results/corrected/tables/` | Corrected tables in CSV and TeX formats |
| `results/corrected/figures/` | Corrected figures in PDF and PNG formats |
| `results/corrected/external_comparisons/rrr_skill/` | Accepted corrected H3 result record |
| `results/replay_inputs/` | Compressed frozen ledgers for the corrected writer replays |
| `results/external_comparisons/` | Completed Claude Code summaries and five-run NotebookLM record |
| `results/rebuild_inputs/` | Frozen table inputs required by the corrected exhibit builder |
| `REPLICATION_MANIFEST.json` | Package status, population accounting, exclusions, and integrity hashes |
| `results/local_battery_vlm_2_1/` | Historical raw reviews, normalization records, and the first vlm.2.1 exhibit generation |

## Result population

The complete analytical population contains 2,425 attempts.

| Population | Attempts | Accepted treatment |
| --- | ---: | --- |
| Attempts that reached the writer | 941 | 907 produced reviews and 34 writer failures |
| Independent B2, C, and C2 baselines | 300 | Preserved with verified hashes |
| Attempts ending before the writer | 1,184 | Preserved with their recorded terminal states |
| Total | 2,425 | Fully classified |

The 941 writer-stage attempts are stochastic continuations from frozen upstream
ledgers. They produced 907 scored reviews and 34 writer failures. Retrieval,
document admission, retained passages, topic-fit decisions, and outline state
are held fixed for this correction. The replay identifies the corrected writer
boundary. It does not estimate a causal treatment effect between the two
generations.

Under the current accepted checker, 860 of the 907 produced reviews are free of
E1 through E5 and 47 contain at least one event. The as-released records under
`results/local_battery_vlm_2_1/` preserve the earlier checker outcomes for
provenance. Current tables, figures, and scores are under `results/corrected/`.

The main corrected RRR condition contains 100 completed and scored reviews.
Ninety-nine have zero E1 through E5 events under the deposited checker. One
review contains a conflated author-year attribution counted as E1. Across the
condition, the checker identified 4,767 citation surfaces. The complete
condition inventory and run counts are recorded in
`results/corrected/analysis_source/condition_changes.csv`.

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
records 41 checker-failed runs that became successful under the historical
checker. The accepted rescore also counts any unrendered internal evidence
identifier as E3.

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

## Re-executing the full pipeline

The 50 source PDFs are excluded because most are copyrighted.
`bibliography.bib` and `metadata_reference.csv` identify the required documents
and expected metadata. Place legally obtained copies under `corpus/` using the
filenames recorded in `metadata_reference.csv`.

These commands reproduce the local-model evaluation and require Ollama with
the recorded Mistral or Qwen model tags.

Preprocess the corpus:

```bash
chmod +x scripts/preprocess.sh
./scripts/preprocess.sh
```

Run the 1,010-attempt core block:

```bash
chmod +x scripts/run_battery.sh
./scripts/run_battery.sh
```

Run the model-ladder, topic, and corpus-size blocks:

```bash
PHASES="G,T,S" ./scripts/run_comprehensive_battery.sh
```

Together these blocks reproduce the 2,425-attempt design. Model generation is
stochastic, so regenerated prose and some secondary metrics will vary. The
reference summaries, accepted run counts, and checker outputs are deposited
under `results/corrected/`.

## External comparison arms

The completed off-the-shelf Claude Code comparisons cover Haiku 4.5, Sonnet
4.5, and Opus 4.8. Their run-level summaries are under
`results/external_comparisons/claude_code/`. The corrected `/rrr` skill
condition is deposited under
`results/corrected/external_comparisons/rrr_skill/`. Its ten runs produced ten
clean reviews across 642 parsed citations. `external_comparisons.md` records
both execution procedures.

The completed NotebookLM comparison contains five independent reviews produced
from the same 50 uploaded works and common prompt. The deposited record includes
the copied responses, interface chip counts, scoring sheet, and a separate
five-claim page check. NotebookLM did not export the stable document-page
mapping required by the common E1 through E5 checker. The diagnostic page check
found support for three of five sampled claims. The full record is under
`results/external_comparisons/notebooklm/`.

## Reproducibility boundaries

- Corpus PDFs must be obtained by the replicator.
- Local model tags preserve the tested model identities.
- Language-model outputs are stochastic.
- Corrected writer timings are excluded because they are not comparable with
  the original full-pipeline timings.
- Historical commit identifiers embedded in replay records remain provenance
  references to the private development history.
