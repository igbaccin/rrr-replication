#!/usr/bin/env python3
"""Rescore the page-based conditions used by the paper.

The utility reads the canonical paper-artifact manifest, applies the current
citation checker only to an explicit allowlist of page-scored conditions, and
rebuilds their saved result summaries. It is a dry run unless ``--write`` is
supplied. Prompt-only C/C2 outputs are excluded by construction.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import statistics


EXACT_PAGE_PHASES = {
    "phase_a_main",
    "phase_b_validation_off",
    "phase_b2_rag_only",
    "phase_d1_high_threshold",
    "phase_f1_adversarial_irrelevant",
    "phase_f2_adversarial_culture",
    "phase_t1_gender",
    "phase_t2_colonial",
    "phase_s10_docs",
    "phase_s25_docs",
    "phase_s50_docs",
}
LOCAL_MODEL_BASE_PHASES = {
    "phase_g_gemma3_270m",
    "phase_g_gemma3_1b",
    "phase_g_gemma3_4b",
    "phase_g_gemma3_12b",
    "phase_g_qwen3_0_6b",
    "phase_g_qwen3_1_7b",
    "phase_g_qwen3_4b",
    "phase_g_qwen3_8b",
    "phase_g_qwen3_14b",
    "phase_g_ministral-3_3b",
    "phase_g_mistral_7b",
    "phase_g_mistral-small_24b",
}
LOCAL_MODEL_SUFFIXES = (
    "",
    "_refusal_wordsalad",
    "_refusal_far",
    "_refusal_mid",
    "_refusal_near",
)
LOCAL_MODEL_PAGE_PHASES = {
    base + suffix
    for base in LOCAL_MODEL_BASE_PHASES
    for suffix in LOCAL_MODEL_SUFFIXES
}
PAGE_PHASES = EXACT_PAGE_PHASES | LOCAL_MODEL_PAGE_PHASES
EXCLUDED_PHASES = {
    "phase_c_unrestricted",
    "phase_c2_prompt_constrained",
    "phase_d0_gibberish",
    "phase_d2_narrow_prompt",
    "phase_d3_marginal_fit",
    "phase_g_refusal_gibberish",
}
SCORE_KEYS = ("e1", "e2", "e3", "e4", "e5")
RUN_COMPARE_KEYS = SCORE_KEYS + (
    "quotes_checked",
    "quotes_verified",
    "n_citations",
    "n_docs_cited",
)


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def sha256_lf_text(path: Path):
    """Hash UTF-8 text after normalizing line endings to LF."""
    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def resolve_repo_path(repo_root: Path, value: str | Path):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def portable_path(path: Path, repo_root: Path):
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def is_page_phase(name: str):
    return name in PAGE_PHASES


def result_file_for(phase_dir: Path):
    files = sorted(phase_dir.glob("*_results.json"))
    if len(files) != 1:
        raise RuntimeError(
            f"expected one result file in {phase_dir}, found {len(files)}")
    return files[0]


def run_name(row):
    if isinstance(row.get("run_id"), str):
        return row["run_id"]
    if isinstance(row.get("run"), int):
        return f"run_{row['run']:03d}"
    raise RuntimeError(f"result row has no usable run identifier: {row}")


def is_completed(row):
    return (
        not row.get("refusal", False)
        and not row.get("emission_failure", False)
        and "error" not in row
    )


def is_scored(row):
    return is_completed(row) and isinstance(row.get("e1"), (int, float))


def stat(rows, key):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return {"mean": 0, "sd": 0, "min": 0, "max": 0}
    return {
        "mean": round(statistics.mean(values), 2),
        "sd": round(statistics.stdev(values), 2) if len(values) > 1 else 0,
        "min": min(values),
        "max": max(values),
    }


def clean_count(rows):
    return sum(
        1 for row in rows
        if all(row.get(key, 1) == 0 for key in SCORE_KEYS)
    )


def metric_totals(rows):
    scored = [row for row in rows if is_scored(row)]
    return {
        "n_scored": len(scored),
        "clean_reviews": clean_count(scored),
        **{
            key: sum(row.get(key, 0) for row in scored)
            for key in SCORE_KEYS
        },
        "quotes_checked": sum(row.get("quotes_checked", 0) for row in scored),
        "quotes_verified": sum(row.get("quotes_verified", 0) for row in scored),
    }


def rebuild_summary(payload):
    rows = payload.get("per_run", [])
    summary = copy.deepcopy(payload.get("summary", {}))
    completed = [row for row in rows if is_completed(row)]
    scored = [row for row in rows if is_scored(row)]
    refused = [row for row in rows if row.get("refusal", False)]
    emission = [row for row in rows if row.get("emission_failure", False)]

    summary["n_runs"] = len(rows)
    summary["n_completed"] = len(completed)
    summary["n_scored"] = len(scored)
    summary["n_checker_failed"] = len(completed) - len(scored)
    summary["n_refused"] = len(refused)
    summary["refusal_rate_pct"] = round(
        100 * len(refused) / (len(rows) or 1), 1)
    if "n_emission_failed" in summary:
        summary["n_emission_failed"] = len(emission)
        summary["emission_failure_pct"] = round(
            100 * len(emission) / (len(rows) or 1), 1)

    for key in SCORE_KEYS:
        summary[key] = stat(scored, key)
    summary["n_citations"] = stat(scored, "n_citations")
    summary["e1_loose_advisory"] = stat(scored, "e1_loose_advisory_count")
    summary["words"] = stat(scored, "word_count")
    summary["docs_cited"] = stat(scored, "n_docs_cited")

    clean = clean_count(scored)
    summary["zero_err_pct"] = round(100 * clean / (len(scored) or 1), 1)
    if "zero_fab_pct" in summary:
        zero_e1 = sum(1 for row in scored if row.get("e1", 1) == 0)
        summary["zero_fab_pct"] = round(
            100 * zero_e1 / (len(scored) or 1), 1)

    if "jaccard" in summary:
        doc_sets = [
            set(row.get("docs_cited", []))
            for row in scored if row.get("docs_cited")
        ]
        jaccards = []
        for left, right in itertools.combinations(doc_sets, 2):
            union = len(left | right)
            jaccards.append(len(left & right) / union if union else 0)
        summary["jaccard"] = {
            "mean": round(statistics.mean(jaccards), 3) if jaccards else 0,
            "min": round(min(jaccards), 3) if jaccards else 0,
            "max": round(max(jaccards), 3) if jaccards else 0,
        }
    if "core_docs" in summary:
        frequencies = Counter()
        for row in scored:
            frequencies.update(row.get("docs_cited", []))
        summary["core_docs"] = sorted(
            doc_id for doc_id, count in frequencies.items()
            if count >= len(scored) * 0.8
        )
    return summary


def load_checker(checker_path: Path):
    spec = importlib.util.spec_from_file_location("paper_checker", checker_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def subset_metadata(phase_dir: Path, default_metadata: Path):
    for size in (10, 25, 50):
        if phase_dir.name == f"phase_s{size}_docs":
            path = phase_dir.parent / f"subsample_ws_{size}" / "metadata.csv"
            if not path.is_file():
                raise FileNotFoundError(f"subset metadata missing: {path}")
            return path
    return default_metadata


def discover_targets(manifest, repo_root: Path):
    source_dirs = manifest["source_directories"]
    battery_dirs = [source_dirs["primary_battery"]] + source_dirs.get(
        "additional_batteries", [])
    targets = []
    seen = set()
    for value in battery_dirs:
        battery_dir = resolve_repo_path(repo_root, value)
        for phase_dir in sorted(path for path in battery_dir.iterdir() if path.is_dir()):
            if not is_page_phase(phase_dir.name):
                continue
            result_file = result_file_for(phase_dir)
            key = result_file.resolve()
            if key not in seen:
                seen.add(key)
                targets.append((phase_dir, result_file, False))

    for value in manifest.get("selected_claude_results", {}).values():
        result_file = resolve_repo_path(repo_root, value)
        key = result_file.resolve()
        if key not in seen:
            seen.add(key)
            targets.append((result_file.parent, result_file, True))
    return targets


def rescore_target(
    checker,
    phase_dir: Path,
    result_file: Path,
    nested_runs: bool,
    metadata_path: Path,
    data_dir: Path,
    repo_root: Path,
    write: bool,
):
    payload = read_json(result_file)
    original_payload = copy.deepcopy(payload)
    rows = payload.get("per_run", [])
    before = metric_totals(rows)
    changes = []
    citation_files_changed = 0
    citation_files_written = 0
    reviews_dir = phase_dir / "runs" if nested_runs else phase_dir

    for row in rows:
        if not is_scored(row):
            continue
        name = run_name(row)
        review_file = reviews_dir / name / "review_composed.md"
        if not review_file.is_file():
            raise FileNotFoundError(f"scored review missing: {review_file}")
        old_row = copy.deepcopy(row)
        old = {key: old_row.get(key) for key in RUN_COMPARE_KEYS}
        result = checker.check_file(
            str(review_file),
            metadata_path=str(metadata_path),
            data_dir=str(data_dir),
        )
        if result.get("refusal"):
            raise RuntimeError(f"saved scored review became unscorable: {review_file}")
        new = {key: result.get(key) for key in RUN_COMPARE_KEYS}
        row.update(result)
        changed_fields = sorted(
            key for key, value in result.items()
            if key not in old_row or old_row[key] != value
        )
        if changed_fields:
            changes.append({
                "run": name,
                "changed_fields": changed_fields,
                "before": old,
                "after": new,
            })

        citations_path = reviews_dir / name / "citations.json"
        citations_changed = (
            not citations_path.is_file()
            or read_json(citations_path) != result
        )
        if citations_changed:
            citation_files_changed += 1
            if write:
                write_json(citations_path, result)
                citation_files_written += 1

    if any(is_scored(row) for row in rows):
        payload["summary"] = rebuild_summary(payload)
    after = metric_totals(rows)
    result_file_changed = payload != original_payload
    result_file_written = False
    if write and result_file_changed:
        write_json(result_file, payload)
        result_file_written = True
    return {
        "phase": payload["summary"].get("phase", phase_dir.name),
        "phase_directory": portable_path(phase_dir, repo_root),
        "result_file": portable_path(result_file, repo_root),
        "metadata": portable_path(metadata_path, repo_root),
        "before": before,
        "after": after,
        "changed_run_count": len(changes),
        "changed_runs": changes,
        "result_file_changed": result_file_changed,
        "citation_files_changed": citation_files_changed,
        "result_file_written": result_file_written,
        "citation_files_written": citation_files_written,
    }


def sum_totals(records, side):
    keys = (
        "n_scored",
        "clean_reviews",
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "quotes_checked",
        "quotes_verified",
    )
    return {
        key: sum(record[side][key] for record in records)
        for key in keys
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        required=True,
        help="Canonical paper-artifact manifest",
    )
    parser.add_argument("--metadata", default="metadata.csv")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checker", default="scripts/check_citations.py")
    parser.add_argument("--audit-out", default=None)
    parser.add_argument(
        "--show-changes",
        action="store_true",
        help="Print the run-level metric changes during a dry run or write",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Overwrite citations.json and phase result files",
    )
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    manifest_path = resolve_repo_path(repo_root, args.manifest)
    metadata_path = resolve_repo_path(repo_root, args.metadata)
    data_dir = resolve_repo_path(repo_root, args.data_dir)
    checker_path = resolve_repo_path(repo_root, args.checker)
    manifest = read_json(manifest_path)
    checker = load_checker(checker_path)

    records = []
    for phase_dir, result_file, nested_runs in discover_targets(manifest, repo_root):
        phase_metadata = subset_metadata(phase_dir, metadata_path)
        record = rescore_target(
            checker,
            phase_dir,
            result_file,
            nested_runs,
            phase_metadata,
            data_dir,
            repo_root,
            args.write,
        )
        records.append(record)
        print(
            f"[{record['phase']}] scored={record['after']['n_scored']} "
            f"clean {record['before']['clean_reviews']}"
            f"->{record['after']['clean_reviews']} "
            f"E4 {record['before']['e4']}->{record['after']['e4']} "
            f"E5 {record['before']['e5']}->{record['after']['e5']} "
            f"changed={record['changed_run_count']}"
        )
        if args.show_changes:
            for change in record["changed_runs"]:
                before_metrics = ", ".join(
                    f"{key}={change['before'][key]}" for key in RUN_COMPARE_KEYS
                )
                after_metrics = ", ".join(
                    f"{key}={change['after'][key]}" for key in RUN_COMPARE_KEYS
                )
                print(
                    f"  {change['run']} "
                    f"({', '.join(change['changed_fields'])}): "
                    f"[{before_metrics}] -> [{after_metrics}]"
                )

    audit = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "write" if args.write else "dry_run",
        "manifest": portable_path(manifest_path, repo_root),
        "checker": portable_path(checker_path, repo_root),
        "checker_sha256": sha256_lf_text(checker_path),
        "checker_hash_basis": "UTF-8 text with LF line endings",
        "policy": {
            "page_phase_allowlist": sorted(EXACT_PAGE_PHASES),
            "local_model_phase_allowlist": sorted(LOCAL_MODEL_PAGE_PHASES),
            "excluded_phases": sorted(EXCLUDED_PHASES),
            "prompt_only_conditions": [
                "phase_c_unrestricted",
                "phase_c2_prompt_constrained",
            ],
            "quote_association": (
                "all actionable citations in the quote's sentence; "
                "when that sentence has none, the nearest citation-bearing "
                "sentence supplies all actionable citations"
            ),
            "quote_normalisation": (
                "Unicode NFKC, soft hyphens, dash variants, terminal quote "
                "punctuation, OCR pound markers, and ordered ellipsis or "
                "editorial-bracket fragments"
            ),
        },
        "before": sum_totals(records, "before"),
        "after": sum_totals(records, "after"),
        "changed_run_count": sum(record["changed_run_count"] for record in records),
        "conditions": records,
    }
    print(json.dumps({
        "mode": audit["mode"],
        "before": audit["before"],
        "after": audit["after"],
        "changed_run_count": audit["changed_run_count"],
    }, indent=2))
    if args.audit_out:
        audit_path = resolve_repo_path(repo_root, args.audit_out)
        if not args.write:
            raise SystemExit("--audit-out requires --write")
        write_json(audit_path, audit)
        print(f"[audit] {audit_path}")


if __name__ == "__main__":
    main()
