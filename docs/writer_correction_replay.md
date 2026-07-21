# Corrected writer replay

## Purpose

The correction addresses a writer-interface defect found after the July evaluation battery. The writer could cite retained evidence beyond the passages displayed in its call packet. The accepted contract now derives displayed passages, allowed evidence identifiers, rendered citations, accepted document-page pairs, coverage repair, and final validation from the same bounded packet.

The writer stages of every stored run that reached a ledger were regenerated. Upstream retrieval and reasoning state stayed fixed.

## Population

| Group | Conditions | Ledgers |
| --- | --- | ---: |
| Core | A, B, D1, F1, F2, S10, S25, S50, T1, T2 | 635 |
| Model ladder and refusal false accepts | All model runs that reached a ledger | 306 |
| Total | Complete writer replay population | 941 |

The analytical source also preserves 1,184 pre-writer terminal outcomes and 300 independent B2, C, and C2 baseline runs. These populations never enter the corrected RRR review namespace.

## Frozen boundary

Each replay begins with `review_ledger.json`. The archive retains retrieval, document admission, retained passages, working claims, clusters, outline selection, topic-fit decisions, and original run variation. It excludes the original generated review, writer metrics, citation output, quality output, checker results, and completion status from the corrected input tree.

Each corrected run writes a new review, citation record, writer-call contract, prompt and response capture, metrics, completion status, and run manifest. The original run manifest is copied under the explicit name `source_run_manifest.json`.

## Replay archive

The deposited archive is:

```text
results/replay_inputs/writer_correction_260720.tar.gz
```

Its SHA-256 digest and size are recorded in `REPLICATION_MANIFEST.json`. Extract it from the repository root:

```bash
mkdir -p replay_inputs
tar -xzf results/replay_inputs/writer_correction_260720.tar.gz -C replay_inputs
```

The extracted bundle is `replay_inputs/corrected_writer_v17/`.

Before executing a replay, complete `scripts/preprocess.sh` with the 50-document corpus. The frozen ledger supplies the writer packet, and the citation checker uses the reconstructed `metadata.csv` and `data/page_text/` files.

## Smoke replay

List the five-run smoke profile:

```bash
python scripts/run_corrected_writer_replay.py \
  --bundle replay_inputs/corrected_writer_v17 \
  --profile smoke \
  --list
```

Run and audit it:

```bash
python scripts/run_corrected_writer_replay.py \
  --bundle replay_inputs/corrected_writer_v17 \
  --profile smoke

python scripts/audit_corrected_writer_replay.py \
  --bundle replay_inputs/corrected_writer_v17
```

The smoke profile covers ordinary validation, the validation ablation, a weak writer, an adversarial topic, and a reduced corpus.

## Full replay

The replay can be divided into independent shards. For two workers:

```bash
python scripts/run_corrected_writer_replay.py --bundle replay_inputs/corrected_writer_v17 --profile core --shard-count 2 --shard-index 0
python scripts/run_corrected_writer_replay.py --bundle replay_inputs/corrected_writer_v17 --profile core --shard-count 2 --shard-index 1
python scripts/run_corrected_writer_replay.py --bundle replay_inputs/corrected_writer_v17 --profile ladder --shard-count 2 --shard-index 0
python scripts/run_corrected_writer_replay.py --bundle replay_inputs/corrected_writer_v17 --profile ladder --shard-count 2 --shard-index 1
```

Audit the complete merged output:

```bash
python scripts/audit_corrected_writer_replay.py \
  --bundle replay_inputs/corrected_writer_v17 \
  --require-complete
```

## Interpretation

The new reviews are stochastic regenerations from fixed upstream ledgers. Comparisons between original and corrected reviews map the consequences of the repair. They do not identify a causal effect because the writer output was regenerated stochastically.

The accepted replay eliminates every citation surface outside the selected call packets across all 941 replayed runs. `results/corrected/analysis_source/validation.json` records the complete automated validation.
