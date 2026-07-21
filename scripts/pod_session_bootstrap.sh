#!/usr/bin/env bash
# Bootstrap a fresh RunPod container for an RRR session.
#
# Operational pattern: pods are terminated at the end of every session to
# avoid dormancy cost; every new session starts from a clean container. This
# script is idempotent and safe to re-run.
#
# Usage on the pod (or via runpod_exec.py exec --script-file):
#   bash pod_session_bootstrap.sh

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# v15.16: point the RRR caches at /workspace so they survive pod restarts.
# The claim cache is content-keyed and safe to share; the stage cache is
# read-gated by RRR_STAGE_CACHE (the battery runs cold). Persisted via
# /etc/profile.d so LATER SSH sessions (where the battery actually runs)
# inherit them. An export here would end with this bootstrap shell.
mkdir -p /workspace/rrr_claim_cache /workspace/rrr_stage_cache 2>/dev/null || true
cat > /etc/profile.d/rrr_caches.sh <<'RRRENV'
export RRR_CLAIM_CACHE_DIR="${RRR_CLAIM_CACHE_DIR:-/workspace/rrr_claim_cache}"
export RRR_STAGE_CACHE_DIR="${RRR_STAGE_CACHE_DIR:-/workspace/rrr_stage_cache}"
RRRENV
export RRR_CLAIM_CACHE_DIR="${RRR_CLAIM_CACHE_DIR:-/workspace/rrr_claim_cache}"
export RRR_STAGE_CACHE_DIR="${RRR_STAGE_CACHE_DIR:-/workspace/rrr_stage_cache}"

echo "[bootstrap] apt update + base tools (curl wget git python3 venv pip zstd)"
apt-get update -qq
apt-get install -y -qq curl wget git python3 python3-venv python3-pip unzip zstd ca-certificates pciutils >/dev/null

if ! command -v ollama >/dev/null 2>&1; then
  echo "[bootstrap] install current Ollama"
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "[bootstrap] ollama CLI: $(ollama --version 2>&1)"
echo "[bootstrap] start the pull service with the shared model store"
mkdir -p /workspace/ollama_models
pkill -f '[o]llama serve' 2>/dev/null || true
sleep 2
nohup env OLLAMA_HOST=127.0.0.1:11434 \
  OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_FLASH_ATTENTION=1 OLLAMA_MODELS=/workspace/ollama_models \
  ollama serve >/tmp/ollama.log 2>&1 &
disown 2>/dev/null || true
for i in $(seq 1 30); do
  curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
  sleep 1
done

curl -fsS http://127.0.0.1:11434/api/version || { echo "ollama failed"; tail -30 /tmp/ollama.log 2>/dev/null; exit 1; }
echo ""
echo "[bootstrap] ollama version: $(curl -fsS http://127.0.0.1:11434/api/version)"
echo "[bootstrap] models dir env: $(cat /proc/$(pgrep -f 'ollama serve' | head -1)/environ 2>/dev/null | tr '\0' '\n' | grep -E '^OLLAMA_MODELS=' || echo 'OLLAMA_MODELS not set (will use default)')"
echo "[bootstrap] /workspace disk: $(df -h /workspace | tail -1)"
echo "[bootstrap] GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"
echo "[bootstrap] OK"
