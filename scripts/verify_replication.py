#!/usr/bin/env python3
"""Verify the deposited RRR replication package without running an LLM."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "REPLICATION_MANIFEST.json"


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON file: {path.relative_to(ROOT)}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"expected JSON object: {path.relative_to(ROOT)}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file_record(record: dict) -> None:
    path = ROOT / record["path"]
    require(path.is_file(), f"missing deposited file: {record['path']}")
    require(path.stat().st_size == record["bytes"], f"size mismatch: {record['path']}")
    require(sha256(path) == record["sha256"], f"SHA-256 mismatch: {record['path']}")


def verify_corrected_results(package: dict) -> None:
    accepted = package["accepted_results"]
    for key in ("manifest", "analysis_manifest", "validation"):
        verify_file_record(accepted[key])

    artifact_manifest = load_json(ROOT / accepted["manifest"]["path"])
    expected_population = {
        "corrected_writer_replays": 941,
        "preserved_independent_baselines": 300,
        "preserved_pre_writer_terminals": 1184,
    }
    require(
        artifact_manifest.get("population") == expected_population,
        "corrected artifact population mismatch",
    )
    require(
        all(artifact_manifest.get("validation", {}).values()),
        "one or more corrected artifact validations failed",
    )
    require(
        artifact_manifest.get("validation", {}).get("corrected_h3_included") is True,
        "corrected H3 inclusion is not validated",
    )
    corrected_h3 = artifact_manifest.get("external_comparisons", {}).get("corrected_h3", {})
    require(corrected_h3.get("attempts") == 10, "corrected H3 attempt count mismatch")
    require(corrected_h3.get("scored_reviews") == 10, "corrected H3 scored count mismatch")
    require(corrected_h3.get("clean_reviews") == 10, "corrected H3 clean count mismatch")

    validation = load_json(ROOT / accepted["validation"]["path"])
    require(validation.get("all_passed") is True, "analysis-source validation is not complete")
    checks = {item["check"]: item for item in validation.get("checks", [])}
    expected_checks = {
        "replay item count": 941,
        "pre-writer terminal count": 1184,
        "independent baseline count": 300,
        "classified source population": 2425,
    }
    for name, expected in expected_checks.items():
        require(name in checks, f"missing validation check: {name}")
        require(checks[name].get("passed") is True, f"failed validation check: {name}")
        require(checks[name].get("detail") == expected, f"unexpected count for: {name}")
    require(
        checks.get("model outcome rows exhaust attempted runs", {}).get("passed") is True,
        "model outcome rows do not exhaust attempted runs",
    )
    require(
        checks.get("model ladder writer failures", {}).get("detail") == 33,
        "model ladder writer-failure count mismatch",
    )

    accepted_root = ROOT / accepted["path"]
    overall_path = accepted_root / "analysis_source/overall_writer_changes.csv"
    with overall_path.open(encoding="utf-8", newline="") as handle:
        overall_rows = list(csv.DictReader(handle))
    require(len(overall_rows) == 1, "overall writer summary must contain one row")
    overall = overall_rows[0]
    require(overall["new_scored_reviews"] == "907", "produced-review count mismatch")
    require(overall["new_clean_reviews"] == "860", "clean-review count mismatch")
    require(overall["new_reviews_with_error"] == "47", "error-bearing review count mismatch")
    require(overall["new_writer_failures"] == "34", "writer-failure count mismatch")

    condition_path = accepted_root / "analysis_source/condition_changes.csv"
    with condition_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, "condition change table is empty")
    require(sum(int(row["attempts"]) for row in rows) == 2425, "condition attempts do not total 2,425")
    require(
        sum(int(row["replayed_writer_runs"]) for row in rows) == 941,
        "condition replay counts do not total 941",
    )
    require(
        sum(int(row["preserved_independent_baselines"]) for row in rows) == 300,
        "condition independent baselines do not total 300",
    )
    require(
        sum(int(row["preserved_pre_writer_terminals"]) for row in rows) == 1184,
        "condition pre-writer terminals do not total 1,184",
    )
    main = next(row for row in rows if row["condition"] == "phase_a_main")
    require(main["attempts"] == "100", "Phase A attempt count mismatch")
    require(main["new_clean_reviews"] == "99", "Phase A clean-review count mismatch")
    require(main["new_e1_events"] == "1", "Phase A E1 count mismatch")
    require(
        round(float(main["new_n_citations_mean"]) * int(main["new_scored_reviews"]))
        == 4767,
        "Phase A parsed-citation count mismatch",
    )
    for metric in ("new_e2_events", "new_e3_events", "new_e4_events", "new_e5_events"):
        require(main[metric] == "0", f"Phase A contains nonzero {metric}")

    required_tables = (
        "T0_controlled_contract_staircase.csv",
        "T1_main_integrity.csv",
        "T2_conditions.csv",
        "T3_citation_taxonomy.csv",
        "T4_guardrail_activation.csv",
        "T5_refusal_calibration.csv",
        "T_workflow_comparison.csv",
        "F5_local_model_outcomes.csv",
        "TS_corpus_size.csv",
    )
    for name in required_tables:
        require((accepted_root / "tables" / name).is_file(), f"missing table: {name}")

    with (accepted_root / "tables/F5_local_model_outcomes.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        model_rows = list(csv.DictReader(handle))
    require(len(model_rows) == 60, "model table row count mismatch")
    require(
        sum(int(row["writer_failures"]) for row in model_rows) == 33,
        "model table writer-failure count mismatch",
    )
    for row in model_rows:
        accounted = sum(
            int(row[key])
            for key in (
                "clean_reviews",
                "reviews_with_error",
                "substantive_refusals",
                "execution_failures",
                "writer_failures",
            )
        )
        require(
            accounted == int(row["attempts"]),
            f"incomplete model outcome row: {row['model_tag']}/{row['probe']}",
        )

    flagship = {
        row["model_tag"]: row for row in model_rows if row["probe"] == "flagship"
    }
    require(
        flagship["qwen3:4b"]["clean_reviews"] == "24"
        and flagship["qwen3:4b"]["reviews_with_error"] == "1"
        and float(flagship["qwen3:4b"]["mean_e3"]) == 0.2,
        "Qwen 3 4B flagship score mismatch",
    )
    require(
        flagship["qwen3:8b"]["clean_reviews"] == "25"
        and flagship["qwen3:8b"]["reviews_with_error"] == "0",
        "Qwen 3 8B flagship score mismatch",
    )
    require(
        flagship["mistral:7b"]["clean_reviews"] == "12"
        and flagship["mistral:7b"]["reviews_with_error"] == "8"
        and float(flagship["mistral:7b"]["mean_e3"]) == 0.95,
        "Mistral 7B flagship score mismatch",
    )

    with (accepted_root / "tables/TS_corpus_size.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        corpus_rows = list(csv.DictReader(handle))
    corpus_scores = {row["Corpus docs"]: row["Zero-error %"] for row in corpus_rows}
    require(
        corpus_scores == {"10": "100", "25": "100", "50": "96"},
        "corpus-size score mismatch",
    )


def verify_replay_archive(package: dict) -> None:
    record = package["replay_archive"]
    verify_file_record(record)
    archive = ROOT / record["path"]
    manifest_member = f"{record['archive_root']}/manifest.json"
    with tarfile.open(archive, mode="r:gz") as handle:
        member = handle.getmember(manifest_member)
        extracted = handle.extractfile(member)
        require(extracted is not None, "replay manifest could not be read from archive")
        replay_manifest = json.load(io.TextIOWrapper(extracted, encoding="utf-8"))
    require(len(replay_manifest.get("items", [])) == 941, "replay archive item count mismatch")
    require(
        len(replay_manifest.get("independent_baselines", [])) == 300,
        "replay archive baseline count mismatch",
    )
    require(
        len(replay_manifest.get("terminal_before_writer", [])) == 1184,
        "replay archive terminal count mismatch",
    )


def verify_external_comparisons(package: dict) -> None:
    for record in package["external_comparisons"]:
        verify_file_record(record)
        payload = load_json(ROOT / record["path"])
        require(len(payload.get("per_run", [])) == record["attempts"], f"attempt count mismatch: {record['workflow']}")
        require(payload.get("summary", {}).get("n_scored") == record["scored_reviews"], f"scored count mismatch: {record['workflow']}")

    workflow_table = (
        ROOT / package["accepted_results"]["path"]
        / "tables/T_workflow_comparison.csv"
    )
    with workflow_table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    h3 = next(row for row in rows if row["workflow"] == "Claude Code with the /rrr skill")
    require(h3.get("attempts") == "10", "corrected H3 table attempt count mismatch")
    require(h3.get("scored") == "10", "corrected H3 table scored count mismatch")
    require(h3.get("clean_pct") == "100.0", "corrected H3 table clean rate mismatch")
    notebook = next(row for row in rows if row["workflow"] == "NotebookLM grounded chat")
    require(notebook.get("attempts") == "5", "NotebookLM attempt count mismatch")
    require(notebook.get("reviews") == "5", "NotebookLM review count mismatch")
    require(notebook.get("provisional") == "False", "NotebookLM row is still provisional")
    require(notebook.get("written_source_page_pairs") == "178", "NotebookLM page-pair count mismatch")
    require(notebook.get("notebooklm_chip_count") == "176", "NotebookLM chip count mismatch")

    notebook_record = package["notebooklm_comparison"]
    require(notebook_record.get("attempts") == 5, "NotebookLM manifest attempt count mismatch")
    require(notebook_record.get("reviews") == 5, "NotebookLM manifest review count mismatch")
    require(notebook_record.get("page_claims_checked") == 5, "NotebookLM page-check denominator mismatch")
    require(notebook_record.get("page_claims_supported") == 3, "NotebookLM page-check result mismatch")
    for record in notebook_record.get("files", []):
        verify_file_record(record)
    require(len(notebook_record.get("files", [])) == 8, "NotebookLM file inventory mismatch")


def verify_package_hygiene() -> None:
    require(not (ROOT / "runs").exists(), "obsolete or generated root runs directory is present")
    require(not (ROOT / "src/rrr/e4_validate.py").exists(), "obsolete e4_validate.py is present")
    require(not (ROOT / "src/rrr/schemas.py").exists(), "obsolete schemas.py is present")
    required_modules = (
        "writer.py",
        "reasoner.py",
        "render.py",
        "validate.py",
        "outline.py",
        "manifest.py",
    )
    for name in required_modules:
        require((ROOT / "src/rrr" / name).is_file(), f"missing RRR module: {name}")

    prohibited = []
    ignored_parts = {".git", "__pycache__", ".pytest_cache", "reproduced_artifacts"}
    for path in ROOT.rglob("*"):
        if ignored_parts.intersection(path.parts):
            continue
        allowed_figure_roots = (
            "results/corrected/figures",
            "results/local_battery_vlm_2_1/exhibits/figures",
        )
        if (
            path.is_file()
            and path.suffix.lower() == ".pdf"
            and not any(root in path.as_posix() for root in allowed_figure_roots)
        ):
            prohibited.append(path.relative_to(ROOT).as_posix())
    require(not prohibited, f"prohibited package artifacts: {prohibited[:10]}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    stale_phrases = (
        "200+ pipeline runs",
        "six phases and produces",
        "E2 (invalid pages), Phase A | 0.02",
    )
    for phrase in stale_phrases:
        require(phrase not in readme, f"stale README phrase remains: {phrase}")


def main() -> int:
    try:
        package = load_json(MANIFEST_PATH)
        require(package.get("schema_version") == "rrr-replication-v1", "unsupported package manifest")
        verify_corrected_results(package)
        verify_replay_archive(package)
        verify_external_comparisons(package)
        verify_package_hygiene()
    except (VerificationError, KeyError, OSError, tarfile.TarError, json.JSONDecodeError) as exc:
        print(f"[replication] FAILED: {exc}", file=sys.stderr)
        return 1

    print("[replication] PASS: deposited file hashes")
    print("[replication] PASS: 2,425-attempt population accounting")
    print("[replication] PASS: 941-item writer replay archive")
    print("[replication] PASS: corrected Phase A integrity")
    print("[replication] PASS: external comparison summaries")
    print("[replication] PASS: package hygiene")
    if package.get("source", {}).get("review_repository"):
        print("[replication] PASS: anonymous review repository recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
