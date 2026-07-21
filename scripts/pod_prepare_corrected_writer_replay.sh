#!/usr/bin/env bash
# Prepare the pinned RunPod PyTorch image for corrected_writer_v17.
#
# The repository must already exist at /workspace/RRR. The replay bundle may
# arrive later, which allows model pulls and bundle transfer to overlap.

set -euo pipefail

ROOT="${RRR_REPO_ROOT:-/workspace/RRR}"
PYTHON_BIN="${RRR_PYTHON:-${ROOT}/.venv/bin/python}"

if [[ ! -d "$ROOT/.git" ]]; then
  echo "[prepare] repository missing at ${ROOT}" >&2
  exit 1
fi

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
if [[ "$GPU_COUNT" -ne 2 ]]; then
  echo "[prepare] expected two GPUs and detected ${GPU_COUNT}" >&2
  exit 1
fi

cd "$ROOT"
bash scripts/pod_session_bootstrap.sh

if [[ ! -x "$PYTHON_BIN" ]]; then
  python3 -m venv "${ROOT}/.venv"
fi
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" -m pip check

for required in metadata.csv data indices src/rrr/writer.py; do
  if [[ ! -e "${ROOT}/${required}" ]]; then
    echo "[prepare] required repository path missing: ${required}" >&2
    exit 1
  fi
done

"$PYTHON_BIN" -m compileall -q src scripts

echo "[prepare] commit $(git rev-parse HEAD)"
echo "[prepare] Python $("$PYTHON_BIN" --version 2>&1)"
echo "[prepare] two-GPU environment ready"
df -h /workspace
