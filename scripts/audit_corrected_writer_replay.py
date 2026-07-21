#!/usr/bin/env python3
"""Audit corrected writer replay outputs against their call contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "replay_inputs" / "corrected_writer_v17"
DEFAULT_OUTPUTS = ROOT / "runs" / "corrected_writer_v17"
sys.path.insert(0, str(ROOT / "src"))

from rrr.writer import _citation_fingerprints, _classify_sentence_violations


STYLE_NUMBERED_SENTENCE_RE = re.compile(r"^(\d+)\. (.*)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def style_rewrite_drift_survives(run_dir: Path, review: str) -> bool:
    """Detect a style rewrite that changed citations and reached the review."""
    prompt_dir = run_dir / "prompts"
    for prompt_path in sorted(prompt_dir.glob("style_rewrite_*.txt")):
        if prompt_path.name.endswith("_response.txt"):
            continue
        response_path = prompt_path.with_name(
            prompt_path.stem + "_response.txt"
        )
        if not response_path.is_file():
            continue
        prompt = prompt_path.read_text(encoding="utf-8", errors="replace")
        originals = []
        in_sentences = False
        for line in prompt.splitlines():
            if line == "Sentences:":
                in_sentences = True
                continue
            if not in_sentences:
                continue
            match = STYLE_NUMBERED_SENTENCE_RE.match(line)
            if match:
                originals.append(match.group(2))

        raw = response_path.read_text(
            encoding="utf-8", errors="replace"
        ).strip()
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            response = json.loads(raw[start:end + 1])
            rewritten = response.get("rewritten") or []
        except Exception:
            continue
        if not isinstance(rewritten, list) or len(rewritten) != len(originals):
            continue

        for original, new_text in zip(originals, rewritten):
            new_text = str(new_text or "").strip()
            if not new_text or _classify_sentence_violations(new_text):
                continue
            if _citation_fingerprints(original) == _citation_fingerprints(new_text):
                continue
            if new_text in review:
                return True
    return False


def audit_run(item: dict, bundle: Path, outputs: Path) -> list[str]:
    run_dir = (
        outputs / item["batch"] / item["condition"] / item["run"]
    )
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        return ["incomplete"]
    status = load_json(status_path)
    if not status.get("completed"):
        return ["status_not_completed"]
    outcome = status.get("outcome") or "success"

    if outcome != "success":
        required_failure = (
            "review_ledger.json",
            "run_metrics.json",
            "replay_provenance.json",
            "run_manifest.json",
            "failure.json",
        )
        missing = [
            f"missing_{name}"
            for name in required_failure
            if not (run_dir / name).is_file()
        ]
        if missing:
            return missing
        return [f"recorded_outcome:{outcome}"]

    errors = []
    ledger_path = bundle / item["bundle_ledger"]
    if sha256(ledger_path) != item["source_ledger_sha256"]:
        errors.append("source_ledger_hash_mismatch")
    if (run_dir / "review_ledger.json").is_file():
        if sha256(run_dir / "review_ledger.json") != item["source_ledger_sha256"]:
            errors.append("corrected_run_ledger_hash_mismatch")

    required = (
        "review_composed.md",
        "review_ledger.json",
        "writer_call_contracts.json",
        "citations_manifest.json",
        "citations.json",
        "quality_manifest.json",
        "run_metrics.json",
        "replay_provenance.json",
        "run_manifest.json",
        "stdout.txt",
        "stderr.txt",
    )
    for name in required:
        if not (run_dir / name).is_file():
            errors.append(f"missing_{name}")
    if errors:
        return errors

    contracts = load_json(run_dir / "writer_call_contracts.json")
    calls = contracts.get("calls") or []
    if not calls:
        errors.append("no_call_contracts")
    allowed_pairs = set()
    for call in calls:
        if not all((call.get("invariants") or {}).values()):
            errors.append("call_invariant_failed")
        for pair in call.get("allowed_doc_page_pairs") or []:
            allowed_pairs.add((pair.get("doc_id"), int(pair.get("page", 0))))

    review = (run_dir / "review_composed.md").read_text(encoding="utf-8")
    if (
        item.get("source_review_sha256")
        and sha256(run_dir / "review_composed.md") == item["source_review_sha256"]
    ):
        errors.append("corrected_review_matches_original_review_hash")
    if re.search(r"\[[Ee]\d{1,5}\]", review):
        errors.append("raw_evidence_id_in_review")

    citation_manifest = load_json(run_dir / "citations_manifest.json")
    for citation in citation_manifest.get("citations") or []:
        did = citation.get("doc_id")
        page = int(citation.get("page", 0) or 0)
        if did and (did, page) not in allowed_pairs:
            errors.append("final_citation_outside_call_union")
            break

    prompt_dir = run_dir / "prompts"
    prompts = sorted(
        p for p in prompt_dir.glob("*.txt")
        if not p.name.endswith("_response.txt")
    )
    responses = sorted(prompt_dir.glob("*_response.txt"))
    response_stems = {
        p.name.removesuffix("_response.txt") for p in responses
    }
    missing_responses = [
        p.name for p in prompts if p.stem not in response_stems
    ]
    if missing_responses:
        errors.append("prompt_without_response")

    metrics = load_json(run_dir / "run_metrics.json")
    style_metrics = (
        (metrics.get("values") or {}).get("writer_style_enforcement") or {}
    )
    if (
        style_metrics.get("citation_fingerprint_guard") != "all-surfaces-v1"
        and style_rewrite_drift_survives(run_dir, review)
    ):
        errors.append("style_rewrite_citation_drift_in_review")
    llm_calls = metrics.get("llm_calls") or []
    if not llm_calls:
        errors.append("no_recorded_writer_calls")
    if len(prompts) != len(llm_calls):
        errors.append("prompt_count_differs_from_llm_call_count")
    if len(responses) != len(llm_calls):
        errors.append("response_count_differs_from_llm_call_count")

    corrected_manifest = load_json(run_dir / "run_manifest.json")
    if corrected_manifest.get("schema_version") != "corrected-writer-run-v17":
        errors.append("invalid_corrected_run_manifest")
    for artifact in corrected_manifest.get("artifacts") or []:
        artifact_path = run_dir / artifact.get("path", "")
        if not artifact_path.is_file():
            errors.append("manifest_artifact_missing")
            break
        if sha256(artifact_path) != artifact.get("sha256"):
            errors.append("manifest_artifact_hash_mismatch")
            break

    return sorted(set(errors))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit corrected writer replay outputs"
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    manifest = load_json(args.bundle / "manifest.json")
    outcomes = Counter()
    failures = []
    for item in manifest.get("items") or []:
        errors = audit_run(item, args.bundle, args.outputs)
        if errors == ["incomplete"]:
            outcomes["incomplete"] += 1
            continue
        if len(errors) == 1 and errors[0].startswith("recorded_outcome:"):
            outcomes[errors[0].split(":", 1)[1]] += 1
            continue
        if errors:
            outcomes["failed"] += 1
            failures.append({
                "item_index": item["item_index"],
                "condition": item["condition"],
                "run": item["run"],
                "errors": errors,
            })
        else:
            outcomes["passed"] += 1

    result = {
        "schema_version": "corrected-writer-replay-audit-v17",
        "total": len(manifest.get("items") or []),
        "outcomes": dict(outcomes),
        "failures": failures[:100],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        return 1
    if args.require_complete and outcomes["incomplete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
