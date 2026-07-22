# Replication Package for Retrieval-Restricted Reasoning

This repository accompanies the paper, "Retrieval-Restricted Reasoning: A Proof of Concept for Adapting Language Models to Economic History."

The package is synchronized with the corrected writer contract and the accepted results bundle dated 21 July 2026. It contains the current RRR implementation, the evaluation programs, run-level analytical summaries, generated exhibits, and the frozen input archive for the 941 corrected writer replays.

The NotebookLM comparison remains a documented pilot pending the final human-run protocol. The former RRR skill comparison is excluded because its recorded runs used the superseded writer.

## Fast verification

The deposited results can be checked without the copyrighted corpus or an installed language model.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python scripts/verify_replication.py
python -m unittest discover -s tests/unit -v
```

To regenerate the corrected tables and figures from the deposited analysis source:

```bash
python scripts/build_corrected_paper_artifacts.py --out reproduced_artifacts
```

The script validates the expected populations and writes the regenerated exhibits under `reproduced_artifacts/`.

## Package contents

| Path | Contents |
| --- | --- |
| `src/rrr/` | RRR pipeline source used by the corrected writer replay |
| `skills/rrr/SKILL.md` | Claude Code skill that exposes the RRR workflow as `/rrr` |
| `scripts/` | Corpus preparation, batteries, scoring, replay, and exhibit builders |
| `tests/unit/` | Regression tests, including the corrected writer evidence contract |
| `results/corrected/analysis_source/` | Run-level analytical records used to construct the paper exhibits |
| `results/corrected/tables/` | Corrected tables in CSV and TeX formats |
| `results/corrected/figures/` | Corrected figures in PDF and PNG formats |
| `results/replay_inputs/` | Compressed frozen ledgers for the 941 writer replays |
| `results/external_comparisons/` | Completed Claude Code summaries and the provisional NotebookLM pilot record |
| `results/rebuild_inputs/` | Clearly labelled frozen table inputs required by the corrected exhibit builder |
| `docs/` | Replay, RunPod, and external-comparison instructions |
| `REPLICATION_MANIFEST.json` | Package status, population accounting, exclusions, and integrity hashes |

## Using RRR as a Claude skill

The reviewer package includes the same `rrr` skill used for the skill
condition. Copy `skills/rrr/` to `.claude/skills/rrr/` within a
Claude Code workspace. The skill then exposes RRR as `/rrr` and can also
be selected automatically for a corpus-bounded literature review. It runs the
RRR implementation contained in this package and preserves the pipeline's
validation and audit artifacts.

## Result population

The corrected analytical source classifies 2,425 attempts.

| Population | Attempts | Treatment |
| --- | ---: | --- |
| Runs that reached the writer | 941 | Regenerated under the corrected evidence contract |
| Independent B2, C, and C2 baselines | 300 | Preserved with verified hashes because they do not use the RRR writer |
| Runs ending before the writer | 1,184 | Preserved with their original terminal states |
| Total | 2,425 | Complete classified source population |

The 941 regenerated reviews are stochastic continuations from frozen upstream ledgers. Retrieval, document admission, retained passages, topic-fit decisions, and outline state are held fixed for this correction. The replay identifies the corrected writer boundary; it does not estimate a causal treatment effect between the two generations.

The main corrected RRR condition contains 100 completed and scored reviews. All 100 have zero E1 through E5 events under the deposited checker. The complete condition inventory and run counts are recorded in `results/corrected/analysis_source/condition_changes.csv`.

## Verifying the writer replay inputs

The replay archive is compressed because its 941 ledgers expand to approximately 299 MiB.

```bash
mkdir -p replay_inputs
tar -xzf results/replay_inputs/writer_correction_260720.tar.gz -C replay_inputs
python scripts/run_corrected_writer_replay.py \
  --bundle replay_inputs/corrected_writer_v17 \
  --profile smoke \
  --list
```

Execute the smoke replay after starting Ollama and installing the required model:

```bash
python scripts/run_corrected_writer_replay.py \
  --bundle replay_inputs/corrected_writer_v17 \
  --profile smoke
```

The full replay can be divided across workers with `--shard-count` and `--shard-index`. See `docs/writer_correction_replay.md` and `docs/runpod_replay.md`.

## Re-executing the full pipeline

The 50 source PDFs are excluded because most are copyrighted. `bibliography.bib` and `metadata_reference.csv` identify the required documents and expected metadata. Place legally obtained copies under `corpus/` using the filenames recorded in `metadata_reference.csv`.

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

Together these blocks reproduce the 2,425-attempt design. Model generation is stochastic, so regenerated prose and some secondary metrics will vary. The reference summaries, exact accepted run counts, and checker outputs are deposited under `results/corrected/`.

## External comparison arms

The paper reports completed off-the-shelf Claude Code comparisons for Haiku 4.5, Sonnet 4.5, and Opus 4.8. Their run-level result summaries are under `results/external_comparisons/claude_code/`, and `docs/external_comparisons.md` records the execution procedure.

The NotebookLM directory currently preserves one pilot response, its runbook, and its scoring sheet. It is marked provisional throughout the package. The final NotebookLM exercise will be added after the human-run protocol is completed.

## Reproducibility boundaries

- Corpus PDFs must be obtained by the replicator.
- Local model tags identify the tested model tiers. Availability and upstream model packaging can change over time.
- Language-model outputs are stochastic.
- Corrected writer timings are excluded because replay timings are not comparable with the original full-pipeline timings.
- The RRR skill arm is excluded from the accepted results because it used the superseded writer.
- Original commit hashes embedded in replay records refer to the private archival history. They remain historical provenance identifiers.

## Citation and licence

Please cite the paper when using the package. The source code is distributed under the terms in `LICENSE.txt`.

Questions about replication can be directed to Igor Martins at `igor.martins@ekh.lu.se`.
