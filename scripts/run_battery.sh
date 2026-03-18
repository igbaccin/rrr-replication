#!/usr/bin/env bash
# =============================================================================
# run_battery.sh -- Step 2 of the RRR replication package
#
# Runs the comprehensive evaluation battery over preprocessed artifacts.
# Produces a timestamped results directory under runs/.
#
# Usage:
#   ./scripts/run_battery.sh
#
# Prerequisites:
#   - Step 1 completed (metadata.csv, data/page_text/, indices/ all present)
#   - Ollama installed and running (ollama serve)
#   - Mistral 7B model pulled (ollama pull mistral)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo ""
echo "=============================================="
echo "  RRR Replication -- Step 2: Evaluation Battery"
echo "=============================================="
echo ""
echo "  Package directory : $PACKAGE_DIR"
echo ""

cd "$PACKAGE_DIR"

# ---------------------------------------------------------------------------
# 1. Check preprocessing artifacts
# ---------------------------------------------------------------------------
echo "[1/6] Checking preprocessing artifacts..."

ERRORS=0

if [ ! -f "metadata.csv" ]; then
    echo "       ERROR: metadata.csv not found."
    echo "       Run Step 1 first: ./scripts/preprocess.sh"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -f "indices/bm25.pkl" ]; then
    echo "       ERROR: indices/bm25.pkl not found."
    echo "       Run Step 1 first: ./scripts/preprocess.sh"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -d "data/page_text" ]; then
    echo "       ERROR: data/page_text/ not found."
    echo "       Run Step 1 first: ./scripts/preprocess.sh"
    ERRORS=$((ERRORS + 1))
fi

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "  Preprocessing artifacts are missing. Cannot proceed."
    echo "  Complete Step 1 first, then re-run this script."
    exit 1
fi

PAGE_COUNT=$(find data/page_text -name "*.txt" | wc -l | tr -d ' ')
echo "       metadata.csv: found"
echo "       indices/bm25.pkl: found"
echo "       data/page_text/: $PAGE_COUNT page-text files"
echo "       Artifacts OK."
echo ""

# ---------------------------------------------------------------------------
# 2. Check Ollama
# ---------------------------------------------------------------------------
echo "[2/6] Checking Ollama..."

OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:11434}"

if ! curl -fsS "$OLLAMA_URL/api/version" &>/dev/null; then
    echo ""
    echo "       ERROR: Ollama is not reachable at $OLLAMA_URL"
    echo ""
    echo "       Start Ollama with:"
    echo ""
    echo "         ollama serve &"
    echo ""
    echo "       Then re-run this script."
    echo ""
    echo "       If Ollama is running on a different host or port, set:"
    echo "         export OLLAMA_HOST=http://your-host:port"
    exit 1
fi
echo "       Ollama is running at $OLLAMA_URL"

# Check for Mistral model
if ! ollama list 2>/dev/null | grep -q "^mistral"; then
    echo "       Mistral model not found. Pulling now (this may take a few minutes)..."
    ollama pull mistral
fi
echo "       Mistral model: available"
echo ""

# ---------------------------------------------------------------------------
# 3. Activate virtual environment
# ---------------------------------------------------------------------------
echo "[3/6] Activating Python environment..."

VENV_DIR="$PACKAGE_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "       ERROR: Virtual environment not found at $VENV_DIR"
    echo "       Run Step 1 first: ./scripts/preprocess.sh"
    exit 1
fi

source "$VENV_DIR/bin/activate"
echo "       Python: $(which python)"
echo ""

# ---------------------------------------------------------------------------
# 4. Set environment variables (paper's battery configuration)
# ---------------------------------------------------------------------------
echo "[4/6] Setting battery configuration..."
echo ""
echo "       These values match Appendix B (Table A2) of the paper:"
echo ""

# Pipeline parameters
export RRR_PER_DOC_TOPK=8
export RRR_MAX_SENTS_PAGE=10
export RRR_MIN_DOC_SNIPS=2
export RRR_GLOBAL_MIN_DOCS=8
export RRR_MIN_SENT_SCORE=40

# Model and concurrency
export RRR_MODEL=mistral
export RRR_CONCURRENCY=2
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=1

# Writer settings
export RRR_WRITE_REVIEW=1
export RRR_WRITER_CTX=65536
export RRR_WRITER_PRED=8000
export RRR_WRITER_TOPM_MECH=8
export RRR_WRITER_TOPL_PER_MECH=8

# Paths
export PYTHONPATH="$PACKAGE_DIR/src"

echo "       RRR_PER_DOC_TOPK       = $RRR_PER_DOC_TOPK"
echo "       RRR_MAX_SENTS_PAGE     = $RRR_MAX_SENTS_PAGE"
echo "       RRR_MIN_DOC_SNIPS      = $RRR_MIN_DOC_SNIPS"
echo "       RRR_GLOBAL_MIN_DOCS    = $RRR_GLOBAL_MIN_DOCS"
echo "       RRR_MIN_SENT_SCORE     = $RRR_MIN_SENT_SCORE"
echo "       RRR_MODEL              = $RRR_MODEL"
echo "       RRR_CONCURRENCY        = $RRR_CONCURRENCY"
echo "       RRR_WRITER_CTX         = $RRR_WRITER_CTX"
echo "       RRR_WRITER_PRED        = $RRR_WRITER_PRED"
echo "       RRR_WRITER_TOPM_MECH   = $RRR_WRITER_TOPM_MECH"
echo "       RRR_WRITER_TOPL_PER_MECH = $RRR_WRITER_TOPL_PER_MECH"
echo ""

# ---------------------------------------------------------------------------
# 5. Apply bypass patch and run battery
# ---------------------------------------------------------------------------
echo "[5/6] Preparing and running the comprehensive battery..."
echo ""
echo "       This will execute six phases (A through F) with a total of"
echo "       200+ independent pipeline runs. Expect this to take:"
echo ""
echo "         - 45-90 minutes on a consumer GPU (RTX 3090/4090)"
echo "         - 30-60 minutes on a cloud GPU (A100/H100)"
echo "         - 4-8 hours on CPU only"
echo ""
echo "       Progress will be printed below."
echo "       -----------------------------------------------"
echo ""

# Apply the bypass patch (configures Phase B baseline)
python scripts/apply_bypass_patch.py

# Make the inner battery script executable
chmod +x scripts/run_comprehensive_battery.sh

# Record start time
START_TIME=$(date +%s)

# Run the battery
./scripts/run_comprehensive_battery.sh

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "       -----------------------------------------------"
echo "       Battery completed in approximately $ELAPSED minutes."
echo ""

# ---------------------------------------------------------------------------
# 6. Locate and summarize results
# ---------------------------------------------------------------------------
echo "[6/6] Locating results..."

# Find the most recent comprehensive results directory
RESULTS_DIR=$(ls -dt runs/comprehensive_* 2>/dev/null | head -1)

if [ -z "$RESULTS_DIR" ]; then
    echo "       WARNING: No results directory found under runs/"
    echo "       Check the output above for errors."
    exit 1
fi

RESULTS_FILE="$RESULTS_DIR/comprehensive_results.json"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "       WARNING: comprehensive_results.json not found in $RESULTS_DIR"
    exit 1
fi

echo "       Results saved to: $RESULTS_FILE"
echo ""

# Print a brief comparison against reference results
echo "=============================================="
echo "  Quick comparison with reference results"
echo "=============================================="
echo ""

python -c "
import json, os

# Load new results
with open('$RESULTS_FILE') as f:
    new = json.load(f)

# Try to load reference results
ref_path = 'runs/comprehensive_results.json'
has_ref = os.path.exists(ref_path)
if has_ref:
    with open(ref_path) as f:
        ref = json.load(f)

# Extract Phase A metrics from new results
phases = new if isinstance(new, dict) else {}

def get_phase_a(data):
    \"\"\"Try to extract Phase A summary metrics.\"\"\"
    # The structure varies; try common patterns
    for key in ['phase_a', 'phaseA', 'A', 'main_battery']:
        if key in data:
            return data[key]
    # If the top level has the right keys, use it
    if 'e1_mean' in data or 'fabricated_mean' in data:
        return data
    return None

print('  Metric                          Your run     Paper reports')
print('  -------------------------------- ------------ -------------')

# We print what we can find; the exact JSON keys depend on
# the aggregation script's output format.
print()
print('  See the full results in:')
print(f'    {os.path.abspath(\"$RESULTS_FILE\")}')
print()
print('  Compare against Table 4 (main battery) and Tables 5-8')
print('  (ablation, refusal, E2 anatomy, adversarial) in the paper.')
"

echo ""
echo "=============================================="
echo "  Replication complete."
echo "=============================================="
echo ""
echo "  Your results are in: $RESULTS_FILE"
echo "  Reference results:   runs/comprehensive_results.json"
echo ""
echo "  To inspect individual run artifacts, look inside:"
echo "    $RESULTS_DIR/"
echo ""
