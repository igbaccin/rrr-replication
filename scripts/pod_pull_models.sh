#!/usr/bin/env bash
# Pull and verify every model needed by corrected_writer_v17.
#
# The model list comes from the replay manifest when it is available. TAGS can
# override it for diagnostics. The fallback list is the frozen v17 population.
#
# Start in the background:
#   bash scripts/pod_pull_models.sh
#
# Wait for completion and verify every tag:
#   WAIT=1 bash scripts/pod_pull_models.sh

set -euo pipefail

ROOT="${RRR_REPO_ROOT:-/workspace/RRR}"
MANIFEST="${REPLAY_MANIFEST:-${ROOT}/replay_inputs/corrected_writer_v17/manifest.json}"
LOG_DIR="${PULL_LOG_DIR:-/workspace/pull_logs}"
DRIVER="${LOG_DIR}/corrected_writer_v17_pull_driver.sh"
PID_FILE="${LOG_DIR}/corrected_writer_v17_pull.pid"
DONE_FILE="${LOG_DIR}/corrected_writer_v17_pull.done"
FAIL_FILE="${LOG_DIR}/corrected_writer_v17_pull.failed"
LOG_FILE="${LOG_DIR}/corrected_writer_v17_pull.log"
FALLBACK_TAGS="gemma3:12b gemma3:1b ministral-3:3b mistral-small:24b mistral:7b qwen3:0.6b qwen3:1.7b qwen3:14b qwen3:4b qwen3:8b"

mkdir -p "$LOG_DIR"
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

if [[ -n "${TAGS:-}" ]]; then
  TAGS_VALUE="$TAGS"
elif [[ -f "$MANIFEST" ]]; then
  TAGS_VALUE="$(
    python3 - "$MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
models = sorted({
    str(item.get("model", "")).strip()
    for item in manifest.get("items", [])
    if str(item.get("model", "")).strip()
})
print(" ".join(models))
PY
  )"
else
  TAGS_VALUE="$FALLBACK_TAGS"
fi

read -r -a MODEL_TAGS <<< "$TAGS_VALUE"
if [[ "${#MODEL_TAGS[@]}" -eq 0 ]]; then
  echo "[pulls] no model tags were found" >&2
  exit 1
fi

if ! curl -fsS "${OLLAMA_HOST}/api/version" >/dev/null; then
  echo "[pulls] Ollama is not reachable at ${OLLAMA_HOST}" >&2
  exit 1
fi

driver_is_running() {
  if [[ ! -f "$PID_FILE" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

write_driver() {
  cat > "$DRIVER" <<'HEAD'
#!/usr/bin/env bash
set -uo pipefail
HEAD
  printf 'export OLLAMA_HOST=%q\n' "$OLLAMA_HOST" >> "$DRIVER"
  printf 'LOG_FILE=%q\n' "$LOG_FILE" >> "$DRIVER"
  printf 'DONE_FILE=%q\n' "$DONE_FILE" >> "$DRIVER"
  printf 'FAIL_FILE=%q\n' "$FAIL_FILE" >> "$DRIVER"
  printf 'MODEL_TAGS=(' >> "$DRIVER"
  printf ' %q' "${MODEL_TAGS[@]}" >> "$DRIVER"
  printf ' )\n' >> "$DRIVER"
  cat >> "$DRIVER" <<'BODY'

failed=0
echo "===== $(date --iso-8601=seconds) corrected_writer_v17 pulls start =====" >> "$LOG_FILE"
for tag in "${MODEL_TAGS[@]}"; do
  echo "----- $(date --iso-8601=seconds) pulling ${tag} -----" >> "$LOG_FILE"
  if ollama pull "$tag" >> "$LOG_FILE" 2>&1; then
    echo "----- $(date --iso-8601=seconds) OK ${tag} -----" >> "$LOG_FILE"
  else
    echo "----- $(date --iso-8601=seconds) FAIL ${tag} -----" >> "$LOG_FILE"
    failed=1
  fi
done
ollama list >> "$LOG_FILE" 2>&1 || failed=1
df -h /workspace >> "$LOG_FILE" 2>&1 || true
if [[ "$failed" -eq 0 ]]; then
  rm -f "$FAIL_FILE"
  date --iso-8601=seconds > "$DONE_FILE"
else
  rm -f "$DONE_FILE"
  date --iso-8601=seconds > "$FAIL_FILE"
fi
echo "===== $(date --iso-8601=seconds) corrected_writer_v17 pulls end rc=${failed} =====" >> "$LOG_FILE"
exit "$failed"
BODY
  chmod +x "$DRIVER"
}

if driver_is_running; then
  echo "[pulls] driver already running with PID $(cat "$PID_FILE")"
elif [[ -f "$DONE_FILE" ]] && [[ "${RETRY:-0}" != "1" ]]; then
  echo "[pulls] a completed pull marker already exists"
elif [[ -f "$FAIL_FILE" ]] && [[ "${RETRY:-0}" != "1" ]]; then
  echo "[pulls] the previous pull attempt failed; inspect ${LOG_FILE}" >&2
  echo "[pulls] run RETRY=1 WAIT=1 bash scripts/pod_pull_models.sh after resolving the cause" >&2
else
  rm -f "$DONE_FILE" "$FAIL_FILE" "$PID_FILE"
  write_driver
  nohup "$DRIVER" >"${LOG_DIR}/corrected_writer_v17_pull.nohup.log" 2>&1 &
  pid=$!
  echo "$pid" > "$PID_FILE"
  disown 2>/dev/null || true
  sleep 1
  if ! kill -0 "$pid" 2>/dev/null && [[ ! -f "$DONE_FILE" ]] && [[ ! -f "$FAIL_FILE" ]]; then
    echo "[pulls] driver failed to start" >&2
    exit 1
  fi
  echo "[pulls] started driver with PID ${pid}"
fi

echo "[pulls] tags: ${MODEL_TAGS[*]}"
echo "[pulls] log: ${LOG_FILE}"

if [[ "${WAIT:-0}" != "1" ]]; then
  echo "[pulls] run WAIT=1 bash scripts/pod_pull_models.sh to wait and verify"
  exit 0
fi

if [[ -f "$FAIL_FILE" ]] && ! driver_is_running; then
  tail -80 "$LOG_FILE" >&2 || true
  exit 1
fi

echo "[pulls] waiting for completion"
while driver_is_running; do
  sleep 20
done

if [[ -f "$FAIL_FILE" ]]; then
  echo "[pulls] at least one pull failed; inspect ${LOG_FILE}" >&2
  tail -80 "$LOG_FILE" >&2 || true
  exit 1
fi
if [[ ! -f "$DONE_FILE" ]]; then
  echo "[pulls] driver ended without a completion marker" >&2
  tail -80 "$LOG_FILE" >&2 || true
  exit 1
fi

missing=0
for tag in "${MODEL_TAGS[@]}"; do
  if ollama show "$tag" >/dev/null 2>&1; then
    echo "[pulls] verified ${tag}"
  else
    echo "[pulls] missing ${tag}" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

echo "[pulls] all ${#MODEL_TAGS[@]} model tags are ready"
df -h /workspace
