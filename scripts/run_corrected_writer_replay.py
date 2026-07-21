#!/usr/bin/env python3
"""Run corrected writer calls from a prepared replay bundle."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "replay_inputs" / "corrected_writer_v17"
OUTPUT_ROOT = ROOT / "runs" / "corrected_writer_v17"

SMOKE_KEYS = {
    ("runpod_20260708_072545", "phase_a_main", "run_001"),
    ("runpod_20260708_072545", "phase_b_validation_off", "run_001"),
    ("runpod_20260709_031835", "phase_g_qwen3_0_6b", "run_001"),
    ("runpod_20260709_202003", "phase_f2_adversarial_culture", "run_001"),
    ("runpod_20260710_140421", "phase_s10_docs", "run_001"),
}


def load_manifest(bundle: Path) -> dict:
    path = bundle / "manifest.json"
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema_version") != "corrected-writer-replay-v17":
        raise ValueError(f"unsupported replay manifest: {path}")
    return value


def selected_items(items: list[dict], args) -> list[dict]:
    selected = []
    for item in items:
        key = (item["batch"], item["condition"], item["run"])
        if args.profile == "smoke" and key not in SMOKE_KEYS:
            continue
        if args.profile in {"core", "ladder"} and item["group"] != args.profile:
            continue
        if args.condition and item["condition"] not in args.condition:
            continue
        if args.model and item.get("model") not in args.model:
            continue
        if item["item_index"] % args.shard_count != args.shard_index:
            continue
        selected.append(item)
    selected.sort(
        key=lambda item: (
            item.get("model") or "",
            item["batch"],
            item["condition"],
            item["run"],
        )
    )
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def git_state() -> dict:
    def run(*parts):
        result = subprocess.run(
            ["git", *parts],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short").splitlines(),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_corrected_manifest(
    out_dir: Path,
    item: dict,
    started_at: str,
    finished_at: str,
    model: str,
    outcome: str,
) -> dict:
    artifacts = []
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
        if path.name in {"status.json", "run_manifest.json"}:
            continue
        artifacts.append({
            "path": path.relative_to(out_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "schema_version": "corrected-writer-run-v17",
        "started_at": started_at,
        "finished_at": finished_at,
        "outcome": outcome,
        "source": item,
        "corrected_git": git_state(),
        "writer_model": model,
        "writer_quotes_per_doc": 2,
        "bypass_validation": bool(item.get("bypass_validation")),
        "artifacts": artifacts,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def execute_item(bundle: Path, item: dict) -> int:
    out_dir = (
        OUTPUT_ROOT
        / item["batch"]
        / item["condition"]
        / item["run"]
    )
    status_path = out_dir / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("completed"):
            print(f"skip completed {out_dir.relative_to(ROOT)}")
            return 0
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"incomplete output directory exists: {out_dir}. "
            "Move it aside before retrying."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    model = item.get("model")
    if not model:
        raise ValueError(f"item {item['item_index']} has no writer model")

    controlled_defaults = {
        "RRR_TOPIC_LANG": "en",
        "RRR_CORPUS_LANG": "en",
        "RRR_WRITER_PARALLEL": "0",
        "RRR_WRITER_STITCH_SENTENCES": "1",
        "RRR_QUOTE_VERIFY": "1",
        "RRR_LINKIFY": "0",
    }
    for key, value in controlled_defaults.items():
        os.environ[key] = str(value)
    for key, value in (item.get("writer_env") or {}).items():
        os.environ[key] = str(value)
    if "RRR_RUNTIME" not in (item.get("writer_env") or {}):
        os.environ.pop("RRR_RUNTIME", None)

    os.environ["RRR_PROJECT_ROOT"] = str(ROOT)
    os.environ["RRR_RUN_ID"] = (
        f"corrected_writer_v17/{item['batch']}/"
        f"{item['condition']}/{item['run']}"
    )
    os.environ["RRR_MODEL"] = model
    os.environ["RRR_WRITER_MODEL"] = model
    os.environ["RRR_WRITER_QUOTES_PER_DOC"] = "2"
    os.environ["RRR_BYPASS_VALIDATION"] = (
        "1" if item.get("bypass_validation") else "0"
    )
    os.environ["RRR_WRITER_PROMPT_DUMP_DIR"] = str(out_dir / "prompts")

    sys.path.insert(0, str(ROOT / "src"))
    from rrr.metrics import RunMetrics
    from rrr.paths import set_default_run_id
    from rrr.writer import compose_review

    set_default_run_id(
        f"corrected_writer_v17/{item['batch']}/"
        f"{item['condition']}/{item['run']}"
    )

    source_ledger = bundle / item["bundle_ledger"]
    target_ledger = out_dir / "review_ledger.json"
    shutil.copy2(source_ledger, target_ledger)
    source_manifest = source_ledger.parent / "run_manifest.json"
    if source_manifest.is_file():
        shutil.copy2(source_manifest, out_dir / "source_run_manifest.json")

    ledger = json.loads(target_ledger.read_text(encoding="utf-8"))
    topic = ledger.get("topic") or "(no topic)"
    metrics = RunMetrics("CORRECTED_WRITER_REPLAY", topic)
    started_at = datetime.now(timezone.utc).isoformat()
    failure_stage = "writer"

    try:
        with (
            (out_dir / "stdout.txt").open("w", encoding="utf-8") as stdout,
            (out_dir / "stderr.txt").open("w", encoding="utf-8") as stderr,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            with metrics.stage("corrected_writer_replay"):
                compose_review(str(target_ledger), metrics=metrics)
        metrics.save()

        writer_citations = out_dir / "citations.json"
        if writer_citations.is_file():
            shutil.copy2(writer_citations, out_dir / "citations_manifest.json")

        failure_stage = "checker"
        checker = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_citations.py"),
                str(out_dir / "review_composed.md"),
                "--metadata",
                str(ROOT / "metadata.csv"),
                "--json",
                str(out_dir / "citations.json"),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        (out_dir / "checker_stdout.log").write_text(
            checker.stdout, encoding="utf-8"
        )
        (out_dir / "checker_stderr.log").write_text(
            checker.stderr, encoding="utf-8"
        )
        if checker.returncode != 0 or not (out_dir / "citations.json").is_file():
            raise RuntimeError(
                f"citation checker failed with return code {checker.returncode}"
            )

        finished_at = datetime.now(timezone.utc).isoformat()
        provenance = write_corrected_manifest(
            out_dir, item, started_at, finished_at, model, "success"
        )
        provenance["schema_version"] = "corrected-writer-replay-v17"
        (out_dir / "replay_provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_corrected_manifest(
            out_dir, item, started_at, finished_at, model, "success"
        )
        status_path.write_text(
            json.dumps(
                {
                    "completed": True,
                    "outcome": "success",
                    "checker_returncode": checker.returncode,
                    "finished_at": finished_at,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"completed {out_dir.relative_to(ROOT)}")
        return 0
    except Exception as exc:
        metrics.save()
        failed_at = datetime.now(timezone.utc).isoformat()
        outcome = f"{failure_stage}_failed"
        (out_dir / "failure.json").write_text(
            json.dumps(
                {
                    "completed": True,
                    "outcome": outcome,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "failed_at": failed_at,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        provenance = write_corrected_manifest(
            out_dir, item, started_at, failed_at, model, outcome
        )
        provenance["schema_version"] = "corrected-writer-replay-v17"
        (out_dir / "replay_provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_corrected_manifest(
            out_dir, item, started_at, failed_at, model, outcome
        )
        status_path.write_text(
            json.dumps(
                {
                    "completed": True,
                    "outcome": outcome,
                    "finished_at": failed_at,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"completed with {outcome}: {out_dir.relative_to(ROOT)}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the corrected writer replay battery"
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--profile",
        choices=("smoke", "core", "ladder", "full"),
        default="smoke",
    )
    parser.add_argument("--condition", action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--execute-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.shard_count < 1:
        parser.error("--shard-count must be at least 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        parser.error("--shard-index must be between 0 and shard-count minus 1")

    manifest = load_manifest(args.bundle)
    items = manifest["items"]
    if args.execute_index is not None:
        item = next(
            item for item in items if item["item_index"] == args.execute_index
        )
        return execute_item(args.bundle, item)

    selected = selected_items(items, args)
    if args.list:
        for item in selected:
            print(
                f"{item['item_index']:04d} {item.get('model') or '(unknown)'} "
                f"{item['batch']}/{item['condition']}/{item['run']}"
            )
        print(f"selected={len(selected)}")
        return 0

    for position, item in enumerate(selected, 1):
        print(
            f"[{position}/{len(selected)}] {item.get('model')} "
            f"{item['batch']}/{item['condition']}/{item['run']}"
        )
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--bundle",
                str(args.bundle),
                "--execute-index",
                str(item["item_index"]),
            ],
            cwd=ROOT,
            check=False,
        )
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
