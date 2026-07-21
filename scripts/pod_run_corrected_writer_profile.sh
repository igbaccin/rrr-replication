#!/usr/bin/env bash
# Run one corrected-writer profile across two GPU-pinned Ollama services.
#
# Usage:
#   bash scripts/pod_run_corrected_writer_profile.sh smoke
#   bash scripts/pod_run_corrected_writer_profile.sh core
#   bash scripts/pod_run_corrected_writer_profile.sh ladder

set -euo pipefail

PROFILE="${1:-}"
ROOT="${RRR_REPO_ROOT:-/workspace/RRR}"
PYTHON_BIN="${RRR_PYTHON:-${ROOT}/.venv/bin/python}"
LOG_DIR="${REPLAY_LOG_DIR:-/workspace/replay_logs}"
BUNDLE="${REPLAY_BUNDLE:-${ROOT}/replay_inputs/corrected_writer_v17}"

case "$PROFILE" in
  smoke|core|ladder) ;;
  *)
    echo "usage: $0 smoke|core|ladder" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[replay] Python environment missing at ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${BUNDLE}/manifest.json" ]]; then
  echo "[replay] bundle manifest missing at ${BUNDLE}/manifest.json" >&2
  exit 1
fi
for port in 11434 11435; do
  if ! curl -fsS "http://127.0.0.1:${port}/api/version" >/dev/null; then
    echo "[replay] Ollama endpoint ${port} is unavailable" >&2
    exit 1
  fi
done

mkdir -p "$LOG_DIR"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[replay] profile ${PROFILE}"
"$PYTHON_BIN" scripts/run_corrected_writer_replay.py \
  --bundle "$BUNDLE" --profile "$PROFILE" --shard-count 2 --shard-index 0 --list \
  | tail -1
"$PYTHON_BIN" scripts/run_corrected_writer_replay.py \
  --bundle "$BUNDLE" --profile "$PROFILE" --shard-count 2 --shard-index 1 --list \
  | tail -1

set +e
OLLAMA_HOST=http://127.0.0.1:11434 \
  "$PYTHON_BIN" scripts/run_corrected_writer_replay.py \
  --bundle "$BUNDLE" --profile "$PROFILE" --shard-count 2 --shard-index 0 \
  >"${LOG_DIR}/${PROFILE}_gpu0.log" 2>&1 &
pid0=$!
OLLAMA_HOST=http://127.0.0.1:11435 \
  "$PYTHON_BIN" scripts/run_corrected_writer_replay.py \
  --bundle "$BUNDLE" --profile "$PROFILE" --shard-count 2 --shard-index 1 \
  >"${LOG_DIR}/${PROFILE}_gpu1.log" 2>&1 &
pid1=$!
echo "$pid0" > "${LOG_DIR}/${PROFILE}_gpu0.pid"
echo "$pid1" > "${LOG_DIR}/${PROFILE}_gpu1.pid"
echo "[replay] GPU 0 PID ${pid0}, GPU 1 PID ${pid1}"

wait "$pid0"
rc0=$?
wait "$pid1"
rc1=$?
set -e

if [[ "$rc0" -ne 0 || "$rc1" -ne 0 ]]; then
  echo "[replay] profile ${PROFILE} stopped, GPU 0 rc=${rc0}, GPU 1 rc=${rc1}" >&2
  tail -80 "${LOG_DIR}/${PROFILE}_gpu0.log" >&2 || true
  tail -80 "${LOG_DIR}/${PROFILE}_gpu1.log" >&2 || true
  exit 1
fi

echo "[replay] profile ${PROFILE} completed"
if [[ "$PROFILE" == "smoke" ]]; then
  "$PYTHON_BIN" scripts/audit_corrected_writer_replay.py \
    --bundle "$BUNDLE"
  "$PYTHON_BIN" - "$ROOT" "$BUNDLE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
bundle = Path(sys.argv[2])
sys.path.insert(0, str(root / "scripts"))
from run_corrected_writer_replay import SMOKE_KEYS

manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
smoke = [
    item for item in manifest["items"]
    if (item["batch"], item["condition"], item["run"]) in SMOKE_KEYS
]
failures = []
for item in smoke:
    status_path = (
        root / "runs" / "corrected_writer_v17" / item["batch"]
        / item["condition"] / item["run"] / "status.json"
    )
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.is_file()
        else {}
    )
    if not status.get("completed") or status.get("outcome") != "success":
        failures.append({
            "item_index": item["item_index"],
            "status": status,
        })
if failures or len(smoke) != 5:
    raise SystemExit(f"smoke acceptance failed: {failures}")
print("[replay] smoke acceptance passed: 5 successful corrected outputs")
PY
elif [[ "$PROFILE" == "ladder" ]]; then
  "$PYTHON_BIN" scripts/audit_corrected_writer_replay.py \
    --bundle "$BUNDLE" --require-complete
else
  "$PYTHON_BIN" scripts/audit_corrected_writer_replay.py \
    --bundle "$BUNDLE"
fi
