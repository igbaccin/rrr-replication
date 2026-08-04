# Reproducing the deposited RRR evaluation

[About RRR](../README.md) · [Installation modes](product_installation.md) ·
[Before running](before_running.md)

The reviewer package supports three tasks. Reviewers can install and run RRR,
check the integrity and population accounting of the deposited evaluation, and
regenerate the paper's tables and figures from the deposited analytical
records. Repeating thousands of stochastic model calls is outside this compact
package. The run-level records, generated reviews, checker outcomes, and exhibit
inputs needed to inspect the reported evaluation are included.

## Verify the deposit

The package can be checked without the copyrighted corpus or a language model:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/verify_replication.py
python -m unittest discover -s tests/unit -v
```

The verification script checks deposited hashes, the complete 2,425-attempt
accounting, the accepted Phase A results, the external-comparison summaries,
and package hygiene.

## Regenerate the paper exhibits

Run the accepted exhibit builder from the repository root:

```bash
python scripts/build_corrected_paper_artifacts.py --out reproduced_artifacts
```

The command writes `reproduced_artifacts/`. It validates all 941 writer-stage
attempts, 1,184 pre-writer outcomes, 300 independent baselines, 60
model-by-probe rows, and 33 writer failures in the model ladder before creating
the CSV, TeX, PDF, and PNG exhibits.

## Deposited records

| Path | Contents |
| --- | --- |
| `results/corrected/analysis_source/` | Run-level analytical records used to construct the accepted exhibits |
| `results/corrected/tables/` | Accepted tables in CSV and TeX formats |
| `results/corrected/figures/` | Accepted figures in PDF and PNG formats |
| `results/corrected/external_comparisons/rrr_skill/` | Accepted RRR skill comparison record |
| `results/external_comparisons/` | Completed Claude Code summaries and the five-run NotebookLM record |
| `results/rebuild_inputs/` | Frozen inputs required by the exhibit builder |
| `results/local_battery_vlm_2_1/` | Generated reviews, run records, and normalization records |
| `REPLICATION_MANIFEST.json` | Package status, population accounting, exclusions, and integrity hashes |

The analytical population contains 2,425 attempts. Of the 941 attempts that
reached the writer, 907 produced reviews and 34 ended in writer failure. A
further 300 attempts belong to independent baselines, while 1,184 reached a
recorded terminal state before the writer. The accepted checker classifies 860
of the 907 produced reviews as free of E1 through E5 and identifies at least
one event in 47 reviews.

The main RRR condition contains 100 completed and scored reviews. Ninety-nine
are free of E1 through E5. One contains a conflated author-year attribution
counted as E1. The checker parsed 4,767 citation surfaces across the condition.
The complete inventory and denominators are recorded in
`results/corrected/analysis_source/condition_changes.csv`.

## Inspect normalization and citations

The historical normalization can be reapplied to all 907 generated reviews:

```bash
python results/local_battery_vlm_2_1/reproduce_normalization.py
```

`per_run.csv` records the resulting changes. Complete E1 through E5 rescoring
uses the source corpus because page validity and quotation support depend on
the retained page text. Most source PDFs are copyrighted and must be obtained
legally by the reviewer. `bibliography.bib` and `metadata_reference.csv`
identify the collection.

The retained checking tools are:

- `scripts/check_citations.py`, the common citation-integrity checker used by
  the deposited evaluation;
- `scripts/rescore_paper_outputs.py`, which reapplies that checker to
  page-based output records when the reconstructed corpus is available.

## External comparisons

The completed Claude Code summaries cover Haiku 4.5, Sonnet 4.5, and Opus 4.8.
The accepted RRR skill condition contains ten completed and scored reviews,
all free of E1 through E5 under the common checker.

The NotebookLM deposit contains five independent reviews produced from the
same 50 works and common prompt. It includes copied responses, interface chip
counts, a scoring sheet, and a separate five-claim page check. NotebookLM's
export did not supply the stable document-page mapping required by the common
checker. The diagnostic page check found support for three of the five sampled
claims.

## Running RRR

Installation and first-run instructions are in
[`product_installation.md`](product_installation.md) and
[`before_running.md`](before_running.md). The plugin bundles support Codex and
Claude Code, and the Python package supports local Ollama models and configured
provider APIs.

## Reproducibility boundaries

- The copyrighted corpus is identified but cannot be redistributed.
- Local model tags preserve the tested model identities.
- Language-model outputs are stochastic, so a new run can differ in prose and
  secondary metrics.
- The deposit supports exact verification and exhibit regeneration from the
  reported run records.
