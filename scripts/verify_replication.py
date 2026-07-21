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
    require("rrr_skill_h3" in artifact_manifest.get("excluded", {}), "H3 exclusion missing")

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

    condition_path = ROOT / "results/corrected/analysis_source/condition_changes.csv"
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
    require(main["new_clean_reviews"] == "100", "Phase A clean-review count mismatch")
    for metric in ("new_e1_events", "new_e2_events", "new_e3_events", "new_e4_events", "new_e5_events"):
        require(main[metric] == "0", f"Phase A contains nonzero {metric}")

    required_tables = (
        "T0_controlled_contract_staircase.csv",
        "T1_main_integrity.csv",
        "T2_conditions.csv",
        "T3_citation_taxonomy.csv",
        "T4_guardrail_activation.csv",
        "T5_refusal_calibration.csv",
        "T_workflow_comparison.csv",
    )
    for name in required_tables:
        require((ROOT / "results/corrected/tables" / name).is_file(), f"missing table: {name}")


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

    workflow_table = ROOT / "results/corrected/tables/T_workflow_comparison.csv"
    with workflow_table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    notebook = next(row for row in rows if row["workflow"] == "NotebookLM pilot")
    require(notebook.get("provisional") == "True", "NotebookLM pilot is not marked provisional")


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
    for path in ROOT.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_dir() and path.name == "__pycache__":
            prohibited.append(path.relative_to(ROOT).as_posix())
        elif path.is_file() and path.suffix.lower() == ".pdf" and "results/corrected/figures" not in path.as_posix():
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
    if package.get("source", {}).get("production_commit") is None:
        print("[replication] PENDING: cleaned production commit will be recorded at freeze")
    if package.get("pending", {}).get("notebooklm"):
        print("[replication] PENDING: final NotebookLM protocol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
