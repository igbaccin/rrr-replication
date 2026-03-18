# Replication Package: Retrieval-Restricted Reasoning (RRR)

**Paper:** "Retrieval-Restricted Reasoning: A Proof of Concept for Adapting Language Models to Economic History"

**Author:** Igor Martins, Department of Economic History, Lund University

---

## What this package contains

This package provides everything needed to reproduce the evaluation reported in the paper. It includes the complete pipeline source code, the evaluation battery, the reference results from the original runs, and two helper scripts that walk you through the process step by step.

The package does **not** include the 50 PDF documents that make up the demonstration corpus, because most of them are under copyright. You will need to obtain your own copies. The bibliography file (`bibliography.bib`) lists every work with full bibliographic metadata, and Section 1 below explains how to set up the corpus folder.


## How replication works (the two-step logic)

The paper describes a two-environment workflow. The same logic applies here:

**Step 1: Preprocessing (your machine, no GPU needed).** You place the 50 PDFs in a folder, run a script, and the system extracts page-level text, builds a retrieval index, and produces a metadata catalog. This step takes about one minute on a standard laptop.

**Step 2: Battery execution (any machine with a GPU, or a CPU if you are patient).** You run the evaluation battery over the preprocessed artifacts. The battery executes 200+ pipeline runs across six phases and produces a single JSON results file. On a consumer GPU (e.g. RTX 4090) the battery completes in roughly one hour. On CPU-only hardware, expect several hours.

Each step has its own shell script. You can run both steps on the same machine, or preprocess locally and run the battery on a cloud GPU.


## Before you start (prerequisites)

You need three things installed:

1. **Python 3.10 or later.** Check with `python3 --version`. The scripts create a virtual environment, so your system Python only needs to support `venv`.

2. **Ollama** (for local LLM inference). Install from [https://ollama.com](https://ollama.com). The battery uses the Mistral 7B model, which Ollama downloads automatically the first time it is needed. Mistral 7B requires roughly 4 GB of VRAM (GPU) or 8 GB of RAM (CPU-only, slower).

3. **The 50 corpus PDFs.** The complete list is in `bibliography.bib`. Every entry includes the DOI or URL where the work can be obtained. If you have institutional access to JSTOR, Cambridge Core, and similar platforms, most items are straightforward to download.

You do **not** need LaTeX, R, or any software beyond Python and Ollama.


---

## Step 0: Set up the corpus folder

Create a folder called `corpus/` inside this package directory and place the 50 PDFs in it. Each PDF must be named according to the convention used in the paper:

```
AuthorName_Year.pdf
```

For single-author works:
```
North_1989.pdf
Austin_2008.pdf
Mokyr_2005.pdf
```

For multi-author works, join surnames with `&`:
```
Broadberry&Gupta_2006.pdf
Sokoloff&Engerman_2000.pdf
North&Weingast_1989.pdf
```

For three or more authors, use the first author followed by `EtAl`:
```
AcemogluEtAl_2001.pdf
AcemogluEtAl_2002.pdf
BroadberryEtal_2015.pdf
AllenEtAl_2011.pdf
```

The full list of expected filenames, with document identifiers, is provided in the table at the end of this README (Appendix A). You can also find the mapping inside `bibliography.bib`.

**Important:** The filenames must match exactly, because the preprocessing script derives document identifiers (`doc_id`) from filenames. A file named `north_1989.pdf` or `North 1989.pdf` will not match the expected identifier `North_1989`.

Once you have placed all 50 PDFs in `corpus/`, your directory should look like this:

```
rrr-replication/
├── README.md              (this file)
├── bibliography.bib
├── metadata_reference.csv
├── requirements.txt
├── corpus/                 << you created this
│   ├── Abramovitz_1986.pdf
│   ├── AcemogluEtAl_2001.pdf
│   ├── ...
│   └── Williamson_1996.pdf
├── src/
│   └── rrr/
│       ├── __init__.py
│       └── ... (pipeline modules)
├── scripts/
│   ├── preprocess.sh
│   ├── run_battery.sh
│   ├── enrich_metadata.py
│   └── ... (battery and evaluation scripts)
└── runs/
    └── comprehensive_results.json  (reference results)
```


---

## Step 1: Preprocess the corpus

Open a terminal, navigate to the package directory, and run:

```bash
chmod +x scripts/preprocess.sh
./scripts/preprocess.sh
```

**What the script does, in order:**

1. Checks that `corpus/` exists and contains PDF files.
2. Checks that `bibliography.bib` is present.
3. Creates a Python virtual environment (`.venv/`) and installs dependencies from `requirements.txt`.
4. Runs metadata enrichment: matches each PDF filename to its BibTeX entry and writes `metadata.csv`.
5. Runs page-text extraction: converts each PDF into one plain-text file per content page, stored in `data/page_text/`. Trailing reference pages are automatically detected and excluded.
6. Builds the BM25 retrieval index over the extracted page text, stored in `indices/`.
7. Prints a summary of what was produced and whether any PDFs failed to process.

**How long it takes:** Under one minute for 50 documents on a standard laptop.

**What to check when it finishes:**

- `metadata.csv` should exist at the package root and contain 50 rows (one per document).
- `data/page_text/` should contain text files named like `North_1989_page_1.txt`, `North_1989_page_2.txt`, etc.
- `indices/bm25.pkl` and `indices/page_ids.npy` should exist.

You can compare your `metadata.csv` against `metadata_reference.csv` (the version produced during the original study). The content-page counts and document identifiers should match. Minor differences in PDF hashes are expected if your PDF copies differ in metadata or formatting from the ones used in the original study.

**If something goes wrong:**

- "No PDFs found in corpus/" -- check that the folder is named `corpus` (not `Corpus` or `pdfs`) and that it is inside the package directory.
- "0 PDFs matched to BibTeX entries" -- check that the filenames follow the naming convention described in Step 0.
- A specific PDF fails to extract -- the script will report it and continue. The most common cause is a scanned PDF without an embedded text layer. Replace it with a text-layer version if available.

**On Windows:** The preprocessing script is a bash script. On Windows, you have three options: (a) use WSL (Windows Subsystem for Linux), (b) use Git Bash, or (c) run the three Python commands manually. The manual commands are:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

set PYTHONPATH=src
python scripts\enrich_metadata.py --corpus corpus --bib bibliography.bib --output metadata.csv
python -m rrr.preprocess --metadata metadata.csv
python -m rrr.index --metadata metadata.csv
```


---

## Step 2: Run the evaluation battery

Make sure Ollama is running before you start:

```bash
ollama serve &          # start in background (skip if already running)
ollama pull mistral     # download the model (~4 GB, only needed once)
```

Then:

```bash
chmod +x scripts/run_battery.sh
./scripts/run_battery.sh
```

**What the script does, in order:**

1. Checks that preprocessing artifacts exist (`metadata.csv`, `indices/bm25.pkl`, `data/page_text/`).
2. Checks that Ollama is reachable and the Mistral model is available.
3. Activates the Python virtual environment created in Step 1.
4. Sets the environment-variable overrides reported in the paper (Appendix B of the paper: retrieval depth, evidence thresholds, writer settings).
5. Runs the comprehensive battery (`scripts/run_comprehensive_battery.sh`), which executes six phases:
   - **Phase A** (100 runs): Full pipeline, original prompt.
   - **Phase B** (20 runs): Retrieval-only baseline (validation bypassed).
   - **Phase C** (20 runs): Unrestricted LLM baseline (no pipeline).
   - **Phase D** (20 runs): High-threshold refusal test.
   - **Phase E** (20 runs): Narrow-prompt refusal test.
   - **Phase F** (20 runs): Adversarial prompt integrity test.
6. Aggregates all results into `runs/comprehensive_<timestamp>/comprehensive_results.json`.
7. Prints a summary comparing your results to the reference results.

**How long it takes:**

| Hardware | Approximate time |
|----------|-----------------|
| Consumer GPU (RTX 3090/4090) | 45--90 minutes |
| Cloud GPU (A100, H100) | 30--60 minutes |
| CPU only (modern laptop) | 4--8 hours |

**What to check when it finishes:**

The key metrics to compare against Table 4 in the paper:

| Metric | Paper reports | What you should see |
|--------|-------------|-------------------|
| E1 (fabricated documents), Phase A | 0.00 across 100 runs | 0.00 |
| E2 (invalid pages), Phase A | 0.02 mean (2 events in 100 runs) | Near zero (exact count may vary) |
| E1, unrestricted baseline (Phase C) | 2.50 mean | Similar magnitude |
| Refusal rate, high-threshold (Phase D) | 100% | 100% |

Because generation is non-deterministic, the exact prose, the set of cited documents, and some secondary metrics (E3 format violations, Jaccard similarity, word counts) will differ across replications. The integrity metrics (E1, E2, refusal rates) should be stable.

**If something goes wrong:**

- "Ollama not reachable" -- make sure `ollama serve` is running. Check with `curl http://127.0.0.1:11434/api/version`.
- "Model mistral not found" -- run `ollama pull mistral` and wait for the download to finish.
- A run hangs or is very slow -- if running on CPU, this is expected. Each run takes 2--5 minutes on CPU vs. 30--60 seconds on GPU.
- Out of memory -- Mistral 7B needs about 4 GB VRAM or 8 GB RAM. Close other applications or use a machine with more memory.


---

## What each file does

### Source code (`src/rrr/`)

| Module | Role |
|--------|------|
| `preprocess.py` | Extracts page-level text from PDFs, detects and excludes reference pages |
| `index.py` | Builds the BM25 retrieval index over extracted page text |
| `retrieve.py` | Queries the BM25 index and returns ranked pages per document |
| `query_planner.py` | Derives a keyword query from the topic prompt |
| `evidence_filter.py` | Scores candidate sentences for topical relevance, deduplicates, filters |
| `validate.py` | Enforces admissibility: corpus boundary, page existence, quote-on-page |
| `reasoner.py` | Orchestrates the full pipeline: retrieval, filtering, validation, composition |
| `writer.py` | Composes the literature review from admitted evidence and cluster maps |
| `stance.py` | Assigns stance labels (supports/complicates/critical) per document |
| `render.py` | Formats output for display |
| `e4_validate.py` | Post-hoc citation integrity checking (E1, E2, E3 metrics) |
| `cli.py` | Command-line interface for running the pipeline |
| `schemas.py` | Data structure definitions |
| `utils.py` | Shared utilities (hashing, JSON I/O, text normalization) |

### Scripts (`scripts/`)

| Script | Role |
|--------|------|
| `preprocess.sh` | Step 1 wrapper: corpus to derived artifacts |
| `run_battery.sh` | Step 2 wrapper: artifacts to evaluation results |
| `enrich_metadata.py` | Matches PDF filenames to BibTeX entries, writes metadata.csv |
| `run_comprehensive_battery.sh` | Inner battery loop: executes all six phases |
| `aggregate_comprehensive.py` | Aggregates per-run outputs into a single results JSON |
| `apply_bypass_patch.py` | Configures the retrieval-only baseline (Phase B) |
| `check_citations.py` | Standalone citation integrity checker |
| `t3_unrestricted.py` | Runs the unrestricted LLM baseline (Phase C) |

### Reference outputs (`runs/`)

| File | Contents |
|------|----------|
| `comprehensive_results.json` | Aggregated results from the original battery run reported in the paper |


---

## Adapting the pipeline to a different corpus

The architecture is not specific to the 50-document demonstration corpus. To use it with a different set of documents:

1. Place your PDFs in `corpus/`, following the `AuthorName_Year.pdf` naming convention.
2. Update `bibliography.bib` with entries for your documents. Each BibTeX key should allow the system to match filenames to metadata.
3. Run Step 1 (`preprocess.sh`). The system will build a new `metadata.csv`, extract page text, and create a fresh BM25 index.
4. Modify the topic prompt in `run_battery.sh` (the `--topic` argument) to match your research question.
5. Run the battery or a single pipeline run.

The evidence thresholds (minimum snippets per document, minimum admissible documents, minimum sentence score) can be adjusted via environment variables. The defaults are listed in Appendix B of the paper and in the `run_battery.sh` script.


---

## Appendix A: Expected corpus filenames

The table below lists all 50 documents in the demonstration corpus with the filename expected by the preprocessing script. The `doc_id` column shows the identifier used throughout the pipeline and in the paper's tables.

### Conceptual works (17 documents)

| Filename | doc_id |
|----------|--------|
| `Abramovitz_1986.pdf` | Abramovitz_1986 |
| `AcemogluEtAl_2001.pdf` | AcemogluEtAl_2001 |
| `Austin_2008.pdf` | Austin_2008 |
| `Braudel&Wallerstein_2009.pdf` | Braudel&Wallerstein_2009 |
| `Bryant_2006.pdf` | Bryant_2006 |
| `Domar_1970.pdf` | Domar_1970 |
| `Goldin_2006.pdf` | Goldin_2006 |
| `Kuznets_1973.pdf` | Kuznets_1973 |
| `Lindert&Williamson_1985.pdf` | Lindert&Williamson_1985 |
| `Mokyr_2005.pdf` | Mokyr_2005 |
| `McCloskey_2015.pdf` | McCloskey_2015 |
| `North&Weingast_1989.pdf` | North&Weingast_1989 |
| `North_1989.pdf` | North_1989 |
| `Ravallion_2001.pdf` | Ravallion_2001 |
| `Skocpol&Sommers_1980.pdf` | Skocpol&Sommers_1980 |
| `Sokoloff&Engerman_2000.pdf` | Sokoloff&Engerman_2000 |
| `Stern_2004.pdf` | Stern_2004 |

### Comparative / Great Divergence (17 documents)

| Filename | doc_id |
|----------|--------|
| `Allen_2001.pdf` | Allen_2001 |
| `AllenEtAl_2011.pdf` | AllenEtAl_2011 |
| `Bolt&vanZanden_2014.pdf` | Bolt&vanZanden_2014 |
| `Broadberry&Gupta_2006.pdf` | Broadberry&Gupta_2006 |
| `BroadberryEtal_2015.pdf` | BroadberryEtal_2015 |
| `Clark&Jacks_2007.pdf` | Clark&Jacks_2007 |
| `deVries_1994.pdf` | deVries_1994 |
| `Feinstein_1998.pdf` | Feinstein_1998 |
| `Humphries_2013.pdf` | Humphries_2013 |
| `Komlos_1998.pdf` | Komlos_1998 |
| `Koepke&Baten_2005.pdf` | Koepke&Baten_2005 |
| `Ogilvie_2007.pdf` | Ogilvie_2007 |
| `Pamuk_2007.pdf` | Pamuk_2007 |
| `Temin_2002.pdf` | Temin_2002 |
| `Williamson_1996.pdf` | Williamson_1996 |
| `vanZanden_1999.pdf` | vanZanden_1999 |
| `vanZanden_2009.pdf` | vanZanden_2009 |

### Empirical / African economic history (16 documents)

| Filename | doc_id |
|----------|--------|
| `AcemogluEtAl_2002.pdf` | AcemogluEtAl_2002 |
| `Akyeampong&Fofack_2014.pdf` | Akyeampong&Fofack_2014 |
| `Austin_2007.pdf` | Austin_2007 |
| `Bohannan_1959.pdf` | Bohannan_1959 |
| `Bryceson_2002.pdf` | Bryceson_2002 |
| `Broadberry&Gardner_2022.pdf` | Broadberry&Gardner_2022 |
| `Frankema_2012.pdf` | Frankema_2012 |
| `Frankema&vanWaijenburg_2012.pdf` | Frankema&vanWaijenburg_2012 |
| `Hopkins_2009.pdf` | Hopkins_2009 |
| `Hopkins_2019.pdf` | Hopkins_2019 |
| `Nunn_2008.pdf` | Nunn_2008 |
| `Nunn&Wantchekon_2011.pdf` | Nunn&Wantchekon_2011 |
| `Peters_2004.pdf` | Peters_2004 |
| `Simson_2019.pdf` | Simson_2019 |
| `Stoler_1989.pdf` | Stoler_1989 |
| `Whatley_2018.pdf` | Whatley_2018 |


---

## Appendix B: Environment variable reference

These variables control the pipeline's behavior. The values below match the battery configuration reported in the paper (Table A2).

| Variable | Default | Battery value | What it controls |
|----------|---------|---------------|-----------------|
| `RRR_MODEL` | mistral | mistral | Ollama model name |
| `RRR_PER_DOC_TOPK` | 4 | 8 | Pages retrieved per document |
| `RRR_MAX_SENTS_PAGE` | 5 | 10 | Max candidate sentences extracted per page |
| `RRR_MIN_DOC_SNIPS` | 1 | 2 | Min validated snippets to include a document |
| `RRR_GLOBAL_MIN_DOCS` | 5 | 8 | Min admissible documents to proceed |
| `RRR_MIN_SENT_SCORE` | 30 | 40 | Lexical overlap threshold (0--100) |
| `RRR_CONCURRENCY` | 1 | 2 | Parallel Ollama requests |
| `RRR_WRITE_REVIEW` | 1 | 1 | Enable composition step |
| `RRR_WRITER_CTX` | 32768 | 65536 | Writer context window (tokens) |
| `RRR_WRITER_PRED` | 4000 | 8000 | Writer max output tokens |
| `RRR_WRITER_TOPM_MECH` | 5 | 8 | Mechanisms included in composition |
| `RRR_WRITER_TOPL_PER_MECH` | 5 | 8 | Evidence items per mechanism |


---

## License and citation

If you use this code or architecture in your own work, please cite the paper:

> Martins, Igor. "Retrieval-Restricted Reasoning: A Proof of Concept for Adapting Language Models to Economic History." *Historical Methods* (forthcoming).

The source code is provided for replication purposes. See `LICENSE` for terms.


---

## Questions or problems

If you encounter issues replicating the results, please contact the author at igor.martins@ekh.lu.se.
