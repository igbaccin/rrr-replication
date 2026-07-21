#!/usr/bin/env bash
# Start one Ollama service per GPU for the corrected writer replay.
#
# Pull and verify all models before running this script. Both services share
# the completed model store and use separate ports and log files.

set -euo pipefail

MODEL_STORE="${OLLAMA_MODELS:-/workspace/ollama_models}"
LOG_DIR="${OLLAMA_LOG_DIR:-/workspace/ollama_logs}"
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
mapfile -t GPU_UUIDS < <(
  nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null
)

if [[ "$GPU_COUNT" -lt 2 || "${#GPU_UUIDS[@]}" -lt 2 ]]; then
  echo "[ollama] two GPUs are required; detected ${GPU_COUNT}" >&2
  exit 1
fi

mkdir -p "$MODEL_STORE" "$LOG_DIR"

pkill -f '[o]llama serve' 2>/dev/null || true
sleep 2

start_service() {
  local gpu_index="$1"
  local gpu_uuid="$2"
  local port="$3"
  local log="${LOG_DIR}/gpu${gpu_index}.log"
  local pid_file="${LOG_DIR}/gpu${gpu_index}.pid"

  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu_uuid" \
    OLLAMA_HOST="127.0.0.1:${port}" \
    OLLAMA_MODELS="$MODEL_STORE" \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_FLASH_ATTENTION=1 \
    ollama serve >"$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$pid_file"
  disown 2>/dev/null || true

  for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${port}/api/version" >/dev/null 2>&1; then
      echo "[ollama] GPU ${gpu_index} (${gpu_uuid}) ready at http://127.0.0.1:${port}, PID ${pid}"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ollama] GPU ${gpu_index} service stopped during startup" >&2
      tail -80 "$log" >&2 || true
      return 1
    fi
    sleep 1
  done

  echo "[ollama] GPU ${gpu_index} service timed out during startup" >&2
  tail -80 "$log" >&2 || true
  return 1
}

start_service 0 "${GPU_UUIDS[0]}" 11434
start_service 1 "${GPU_UUIDS[1]}" 11435

curl -fsS http://127.0.0.1:11434/api/version
echo
curl -fsS http://127.0.0.1:11435/api/version
echo
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
