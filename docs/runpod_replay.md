# Two-GPU replay on RunPod

This guide executes the corrected writer replay on a two-GPU Linux host. Equivalent GPU providers can use the same commands.

## Prepare the repository and corpus

Clone the replication package under `/workspace/RRR`. Obtain the 50 corpus PDFs legally, place them under `corpus/`, and run the preprocessing step. The replay writer receives its evidence from the frozen ledgers, while the citation checker uses the reconstructed metadata and page-text corpus.

```bash
cd /workspace/RRR
bash scripts/preprocess.sh
```

Extract the deposited replay archive:

```bash
mkdir -p replay_inputs
tar -xzf results/replay_inputs/writer_correction_260720.tar.gz -C replay_inputs
test -f replay_inputs/corrected_writer_v17/manifest.json
```

## Bootstrap the host

The helper expects two visible NVIDIA GPUs.

```bash
cd /workspace/RRR
bash scripts/pod_prepare_corrected_writer_replay.sh
```

Pull the required model set recorded by the replay manifest:

```bash
bash scripts/pod_pull_models.sh
```

Start one Ollama service per GPU:

```bash
bash scripts/pod_start_dual_ollama.sh
```

The services listen on ports 11434 and 11435 and share the model store under `/workspace/ollama_models`.

## Smoke profile

```bash
REPLAY_BUNDLE=/workspace/RRR/replay_inputs/corrected_writer_v17 \
  bash scripts/pod_run_corrected_writer_profile.sh smoke
```

The smoke profile must produce five successful corrected outputs and pass the writer-contract audit.

## Full replay

Run the 635-item core block:

```bash
REPLAY_BUNDLE=/workspace/RRR/replay_inputs/corrected_writer_v17 \
  bash scripts/pod_run_corrected_writer_profile.sh core
```

Run the 306-item model-ladder and refusal false-accept block:

```bash
REPLAY_BUNDLE=/workspace/RRR/replay_inputs/corrected_writer_v17 \
  bash scripts/pod_run_corrected_writer_profile.sh ladder
```

The ladder controller ends with a complete audit of all 941 expected outputs.

## Manual audit

```bash
python scripts/audit_corrected_writer_replay.py \
  --bundle replay_inputs/corrected_writer_v17 \
  --outputs runs/corrected_writer_v17 \
  --require-complete
```

Keep `runs/corrected_writer_v17/` together with the controller logs. Every accepted output carries replay provenance and an artifact manifest.

## Resume behavior

Completed runs have a terminal `status.json` and are skipped on relaunch. Repeating the same profile resumes missing attempts. The two shards write disjoint output paths, so their files can be merged directly when separate hosts are used.
