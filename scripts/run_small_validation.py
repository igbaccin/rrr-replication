#!/usr/bin/env python3
"""
Run a small multi-topic RRR validation pass and collect run artifacts.

This is a pre-battery smoke runner. It keeps the normal `runs/` contract for
the pipeline, then copies each topic's artifacts into a timestamped folder.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_TOPICS = [
    "Institutions are the fundamental cause of long-run economic growth",
    "Gender and development shaped long-run economic outcomes",
    "Colonial institutions explain African economic growth trajectories",
]

RUN_FILES = [
    "admission_rejections.json",
    "citations.json",
    "citations_writer_final.json",
    "plan.json",
    "review_cited_docs.json",
    "review_composed.md",
    "review_ledger.json",
    "review_narrative.md",
    "review_references.txt",
    "run_manifest.json",
    "run_metrics.json",
    "topic_fit.json",
    # v15.1.0: outline_plan.json is written even on refusal (before the
    # ledger). Copy it per-run so off-topic refusal smokes can be
    # inspected without re-running.
    "outline_plan.json",
    # v15.7: quality_manifest.json — writer-side observability bundle
    # (attribution_mismatches/retries, unknown_eids, display_leaks,
    # coverage_patches_shipped, etc.). Single-source-of-truth for the
    # 9-battery quality cut.
    "quality_manifest.json",
]

RUN_DIRS = ["cache", "layered_docs"]


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return slug[:max_len] or "topic"


def copy_artifacts(runs_dir: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        src = runs_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)
    for name in RUN_DIRS:
        src = runs_dir / name
        if src.exists():
            dst = dest / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def load_json(path: Path):
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def summarize_topic(topic: str, dest: Path):
    metrics = load_json(dest / "run_metrics.json")
    citations = load_json(dest / "citations.json")
    topic_fit = load_json(dest / "topic_fit.json")
    counters = metrics.get("counters", {})
    values = metrics.get("values", {})
    return {
        "topic": topic,
        "folder": str(dest),
        "duration_s": metrics.get("duration_s"),
        "llm_calls_total": counters.get("llm_calls_total"),
        "docs_total": values.get("docs_total"),
        "docs_admitted_total": values.get("docs_admitted_total"),
        "docs_selected_for_llm": values.get("docs_selected_for_llm"),
        "docs_represented": values.get("docs_represented"),
        "evidence_ids_assigned": values.get("evidence_ids_assigned"),
        "topic_fit_warnings": ";".join(topic_fit.get("warnings", [])),
        "admission_share": topic_fit.get("admission_share"),
        "selected_probe_coverage": topic_fit.get("selected_probe_coverage"),
        "n_citations": citations.get("n_citations"),
        "e1": citations.get("e1"),
        "e2": citations.get("e2"),
        "e3": citations.get("e3"),
        "n_docs_cited": citations.get("n_docs_cited"),
    }


def bounded_env():
    # v15.14: pruned. RRR_DOC_ADMIT_CACHE / RRR_WRITE_REVIEW /
    # RRR_WRITER_ENFORCE_COVERAGE were retired as knobs in v13 (frozen on in
    # source) — setting them was cargo-cult. RRR_WRITER_PARALLEL=1 is GONE
    # on purpose: the production default is the sequential writer with
    # claims-so-far context; forcing parallel here made every smoke measure
    # a different writer path (no claims_so_far) than headline runs use.
    env = os.environ.copy()
    defaults = {
        "RRR_DOC_BUDGET": "24",
        "RRR_DOC_ADMIT_REPLAY": "0",
        "RRR_WRITER_MIN_SECTION_CITED_DOCS": "2",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    return env


def main():
    ap = argparse.ArgumentParser(description="Run a small multi-topic RRR validation pass")
    ap.add_argument("--metadata", default="metadata.csv")
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--topic", action="append", default=None, help="Topic to run. May be repeated.")
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    root = Path.cwd()
    runs_dir = root / "runs"
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root) if args.output_root else root / "runs" / f"small_validation_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    env = bounded_env()
    env["PYTHONPATH"] = str(root / "src")
    topics = args.topic or DEFAULT_TOPICS
    rows = []

    for idx, topic in enumerate(topics, start=1):
        slug = f"{idx:02d}_{slugify(topic)}"
        dest = output_root / slug
        # v15.9: pin RRR_RUN_ID so the child writes its artifacts to
        # runs/<slug>/*, which we then copy from directly (no more
        # clean_runs_dir wipe-between-topics dance).
        topic_env = dict(env)
        topic_env["RRR_RUN_ID"] = slug
        per_run_dir = runs_dir / slug
        # Best effort: clear only this topic's per-run dir if it exists
        # (avoids stale results if the harness re-runs the same slug).
        if per_run_dir.exists():
            shutil.rmtree(per_run_dir)
        cmd = [
            sys.executable,
            "-m",
            "rrr.cli",
            "t2",
            "--metadata",
            args.metadata,
            "--topic",
            topic,
            "--multi",
            "--narrative-only",
        ]
        print(f"[small-validation] {idx}/{len(topics)} {topic}")
        res = subprocess.run(cmd, cwd=str(root), env=topic_env, text=True, capture_output=True, timeout=args.timeout)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "stdout.log").write_text(res.stdout, encoding="utf-8")
        (dest / "stderr.log").write_text(res.stderr, encoding="utf-8")
        (dest / "returncode.txt").write_text(str(res.returncode), encoding="ascii")
        # v15.9: artifacts now live under runs/<slug>/ thanks to RRR_RUN_ID.
        copy_artifacts(per_run_dir, dest)

        review = dest / "review_composed.md"
        if review.is_file():
            citation_json = dest / "citations.json"
            check_cmd = [
                sys.executable,
                "scripts/check_citations.py",
                str(review),
                "--metadata",
                args.metadata,
                "--json",
                str(citation_json),
            ]
            subprocess.run(check_cmd, cwd=str(root), env=env, text=True, capture_output=True, timeout=120)

        row = summarize_topic(topic, dest)
        row["returncode"] = res.returncode
        rows.append(row)
        print(f"[small-validation] returncode={res.returncode} folder={dest}")

    summary_json = output_root / "summary.json"
    summary_csv = output_root / "summary.csv"
    summary_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["topic"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[small-validation] summary={summary_json}")
    print(f"[small-validation] summary_csv={summary_csv}")


if __name__ == "__main__":
    main()
