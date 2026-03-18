#!/usr/bin/env bash
# =============================================================================
# preprocess.sh -- Step 1 of the RRR replication package
#
# Converts a folder of PDFs into the derived artifacts needed by the
# evaluation battery: page-level text, a BM25 retrieval index, and a
# metadata catalog.
#
# Usage:
#   ./scripts/preprocess.sh              (uses defaults)
#   ./scripts/preprocess.sh /path/to/pdfs (custom corpus folder)
#
# Prerequisites:
#   - Python 3.10+
#   - PDFs placed in corpus/ (or the folder you specify)
#   - bibliography.bib in the package root
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CORPUS_DIR="${1:-$PACKAGE_DIR/corpus}"
BIB_FILE="$PACKAGE_DIR/bibliography.bib"

echo ""
echo "=============================================="
echo "  RRR Replication -- Step 1: Preprocessing"
echo "=============================================="
echo ""
echo "  Package directory : $PACKAGE_DIR"
echo "  Corpus directory  : $CORPUS_DIR"
echo "  BibTeX file       : $BIB_FILE"
echo ""

# ---------------------------------------------------------------------------
# 1. Check prerequisites
# ---------------------------------------------------------------------------
echo "[1/7] Checking prerequisites..."

# Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.10 or later."
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "       Python version: $PY_VERSION"

# Corpus folder
if [ ! -d "$CORPUS_DIR" ]; then
    echo ""
    echo "ERROR: Corpus folder not found at $CORPUS_DIR"
    echo ""
    echo "  Create the folder and place the 50 PDFs inside it:"
    echo ""
    echo "    mkdir -p $CORPUS_DIR"
    echo "    # copy your PDFs into $CORPUS_DIR"
    echo ""
    echo "  See README.md (Step 0) for the expected filenames."
    exit 1
fi

PDF_COUNT=$(find "$CORPUS_DIR" -maxdepth 1 -name "*.pdf" -o -name "*.PDF" | wc -l | tr -d ' ')
if [ "$PDF_COUNT" -eq 0 ]; then
    echo ""
    echo "ERROR: No PDF files found in $CORPUS_DIR"
    echo "  Place the corpus PDFs there and try again."
    exit 1
fi
echo "       PDFs found: $PDF_COUNT"

if [ "$PDF_COUNT" -lt 50 ]; then
    echo "       WARNING: Expected 50 PDFs, found $PDF_COUNT. Some documents may be missing."
fi

# BibTeX
if [ ! -f "$BIB_FILE" ]; then
    echo ""
    echo "ERROR: bibliography.bib not found at $BIB_FILE"
    echo "  This file is part of the replication package and should not be missing."
    exit 1
fi
echo "       bibliography.bib found."

echo "       Prerequisites OK."
echo ""

# ---------------------------------------------------------------------------
# 2. Create virtual environment
# ---------------------------------------------------------------------------
VENV_DIR="$PACKAGE_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
    echo "[2/7] Virtual environment already exists at $VENV_DIR. Reusing."
else
    echo "[2/7] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
echo "       Python in use: $(which python)"

# ---------------------------------------------------------------------------
# 3. Install dependencies
# ---------------------------------------------------------------------------
echo "[3/7] Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r "$PACKAGE_DIR/requirements.txt" --quiet
echo "       Dependencies installed."
echo ""

# ---------------------------------------------------------------------------
# 4. Run preprocessing via rrr_setup3.py
# ---------------------------------------------------------------------------
echo "[4/7] Running preprocessing (metadata enrichment + page extraction + indexing)..."
echo ""

export PYTHONPATH="$PACKAGE_DIR/src"

cd "$PACKAGE_DIR"

# Step 4a: Match PDFs to BibTeX entries -> metadata.csv
echo "  [4a] Enriching metadata (matching PDFs to BibTeX entries)..."
python scripts/enrich_metadata.py \
    --corpus "$CORPUS_DIR" \
    --bib "$BIB_FILE" \
    --output "$PACKAGE_DIR/metadata.csv"
echo ""

# Step 4b: Extract page-level text from PDFs -> data/page_text/
echo "  [4b] Extracting page text (detecting and excluding reference pages)..."
python -m rrr.preprocess --metadata "$PACKAGE_DIR/metadata.csv"
echo ""

# Step 4c: Build BM25 retrieval index -> indices/
echo "  [4c] Building BM25 index over extracted page text..."
python -m rrr.index --metadata "$PACKAGE_DIR/metadata.csv"

echo ""

# ---------------------------------------------------------------------------
# 5. Verify outputs
# ---------------------------------------------------------------------------
echo "[5/7] Verifying outputs..."

ERRORS=0

# metadata.csv
if [ -f "$PACKAGE_DIR/metadata.csv" ]; then
    META_ROWS=$(tail -n +2 "$PACKAGE_DIR/metadata.csv" | wc -l | tr -d ' ')
    echo "       metadata.csv: $META_ROWS documents"
else
    echo "       ERROR: metadata.csv was not created."
    ERRORS=$((ERRORS + 1))
fi

# page_text
if [ -d "$PACKAGE_DIR/data/page_text" ]; then
    PAGE_COUNT=$(find "$PACKAGE_DIR/data/page_text" -name "*.txt" | wc -l | tr -d ' ')
    echo "       data/page_text/: $PAGE_COUNT page-text files"
else
    echo "       ERROR: data/page_text/ was not created."
    ERRORS=$((ERRORS + 1))
fi

# BM25 index
if [ -f "$PACKAGE_DIR/indices/bm25.pkl" ]; then
    echo "       indices/bm25.pkl: exists"
else
    echo "       ERROR: indices/bm25.pkl was not created."
    ERRORS=$((ERRORS + 1))
fi

if [ -f "$PACKAGE_DIR/indices/page_ids.npy" ]; then
    echo "       indices/page_ids.npy: exists"
else
    echo "       ERROR: indices/page_ids.npy was not created."
    ERRORS=$((ERRORS + 1))
fi

echo ""

# ---------------------------------------------------------------------------
# 6. Quick BM25 integrity probe
# ---------------------------------------------------------------------------
echo "[6/7] BM25 integrity check..."
python -c "
import pickle, numpy as np
bm = pickle.load(open('indices/bm25.pkl','rb'))
page_ids = np.load('indices/page_ids.npy', allow_pickle=True)
print(f'       Index contains {len(page_ids)} pages.')
"

echo ""

# ---------------------------------------------------------------------------
# 7. Compare against reference metadata (if available)
# ---------------------------------------------------------------------------
echo "[7/7] Comparing against reference metadata..."

if [ -f "$PACKAGE_DIR/metadata_reference.csv" ]; then
    python -c "
import pandas as pd
ref = pd.read_csv('metadata_reference.csv')
new = pd.read_csv('metadata.csv')

ref_ids = set(ref['doc_id'].astype(str))
new_ids = set(new['doc_id'].astype(str))

missing = ref_ids - new_ids
extra   = new_ids - ref_ids

if not missing and not extra:
    print('       All 50 reference documents matched.')
else:
    if missing:
        print(f'       WARNING: {len(missing)} documents from the reference are missing:')
        for m in sorted(missing):
            print(f'         - {m}')
    if extra:
        print(f'       NOTE: {len(extra)} documents not in the reference set:')
        for e in sorted(extra):
            print(f'         + {e}')
"
else
    echo "       metadata_reference.csv not found. Skipping comparison."
fi

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if [ "$ERRORS" -eq 0 ]; then
    echo "=============================================="
    echo "  Preprocessing complete. No errors."
    echo "=============================================="
    echo ""
    echo "  You can now proceed to Step 2:"
    echo ""
    echo "    ./scripts/run_battery.sh"
    echo ""
else
    echo "=============================================="
    echo "  Preprocessing finished with $ERRORS error(s)."
    echo "  Check the messages above before proceeding."
    echo "=============================================="
    exit 1
fi
