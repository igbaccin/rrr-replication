#!/usr/bin/env python3
"""Build the corrected exhibit bundle from the accepted replay analysis.

The script treats ``results/rebuild_inputs/pre_correction_snapshot`` as a
frozen July source. Only writer-independent tables are copied from it. Every
writer-dependent table and the local-model outcome figure are rebuilt from
``results/corrected/analysis_source``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PDF_METADATA = {
    "Creator": "RRR replication package",
    "CreationDate": None,
    "ModDate": None,
}
MODEL_ORDER = [
    "gemma3:270m",
    "gemma3:1b",
    "gemma3:4b",
    "gemma3:12b",
    "ministral-3:3b",
    "mistral:7b",
    "mistral-small:24b",
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen3:4b",
    "qwen3:8b",
    "qwen3:14b",
]
PROBE_ORDER = ["flagship", "wordsalad", "far", "mid", "near"]
PROBE_LABELS = {
    "flagship": "Institutional-growth task",
    "wordsalad": "Word salad",
    "far": "Far-domain physics",
    "mid": "Literary modernism",
    "near": "Soviet-planning near miss",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    replacements = {
        "\u00b1": "+/-",
        "\u2011": "-",
        "\u2013": "-",
        "\u2014": "--",
        "\u00c2\u00b1": "+/-",
        "\u00e2\u20ac\u201d": "--",
    }
    for row in rows:
        for key, value in row.items():
            if not isinstance(value, str):
                continue
            for old, new in replacements.items():
                value = value.replace(old, new)
            row[key] = value
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def write_generic_tex(path: Path, rows: list[dict], caption: str) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty TeX table: {path}")
    headers = list(rows[0])
    columns = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{tex_escape(caption)}}}",
        r"\scriptsize",
        f"\\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(tex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(row.get(header, "")) for header in headers) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def number(value):
    if value in (None, ""):
        return None
    result = float(value)
    return int(result) if result.is_integer() else result


def fmt(value, digits=2):
    value = number(value)
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def mean_sd(row, prefix, digits=2):
    mean = row.get(f"new_{prefix}_mean")
    sd = row.get(f"new_{prefix}_sd")
    if mean in (None, ""):
        return "n/a"
    return f"{fmt(mean, digits)} +/- {fmt(sd, digits)}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_independent_table(frozen_tables: Path, tables: Path, stem: str) -> None:
    for extension in ("csv", "tex"):
        source = frozen_tables / f"{stem}.{extension}"
        if source.exists():
            shutil.copy2(source, tables / source.name)


def paired_by_condition(paired: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for row in paired:
        result.setdefault(row["condition"], []).append(row)
    return result


def build_t0(condition_map, current_rows, paired_groups):
    condition_keys = {
        "Unrestricted generation": "phase_c_unrestricted",
        "Corpus instruction": "phase_c2_prompt_constrained",
        "Single-pass retrieval": "phase_b2_rag_only",
        "Selected-check ablation": "phase_b_validation_off",
        "Full RRR": "phase_a_main",
    }
    rows = []
    for old in current_rows:
        row = dict(old)
        key = condition_keys[row["condition"]]
        if key in {"phase_a_main", "phase_b_validation_off"}:
            source = condition_map[key]
            paired = paired_groups[key]
            scored = int(source["new_scored_reviews"])
            row.update({
                "reviews": scored,
                "scored": scored,
                "mean_e1": fmt(source["new_e1_mean"]),
                "zero_e1_pct": round(
                    100 * sum(float(item["new_e1"]) == 0 for item in paired) / scored, 1
                ),
                "mean_e2": fmt(source["new_e2_mean"]),
                "mean_e3": fmt(source["new_e3_mean"]),
                "mean_e4": fmt(source["new_e4_mean"]),
                "mean_e5": fmt(source["new_e5_mean"]),
                "zero_e1_e5_pct": fmt(source["new_clean_pct"], 1),
                "mean_citations": fmt(source["new_n_citations_mean"]),
                "mean_words": fmt(source["new_body_words_mean"]),
                "sd_words": fmt(source["new_body_words_sd"]),
                "archived_mean_words": fmt(source["new_body_words_mean"]),
                "word_metric_basis": "corrected_review_body",
                "raw_reviews_for_words": scored,
                "reference_headings_removed": sum(
                    item["new_reference_heading_removed"] == "True" for item in paired
                ),
                "mean_documents": fmt(source["new_n_docs_cited_mean"]),
            })
        rows.append(row)
    return rows


def build_t1(condition_map, stability_rows):
    row = condition_map["phase_a_main"]
    stability = {item["metric"]: item for item in stability_rows}
    return [
        {"Measure": "Runs completed (of attempted)", "Value": "100 of 100"},
        {"Measure": "Refusals", "Value": "0 (0.0%)"},
        {"Measure": "E1: out-of-corpus citations / run", "Value": mean_sd(row, "e1")},
        {"Measure": "E2: invalid-page citations / run", "Value": mean_sd(row, "e2")},
        {"Measure": "E3: hard format violations / run", "Value": mean_sd(row, "e3")},
        {"Measure": "E4: unverified quotations / run", "Value": mean_sd(row, "e4")},
        {"Measure": "E5: mis-attributed quotes / run", "Value": mean_sd(row, "e5")},
        {"Measure": "Runs with zero E1", "Value": "100.0%"},
        {"Measure": "Runs with zero E1-E5", "Value": f"{fmt(row['new_clean_pct'], 1)}%"},
        {"Measure": "Citations / run", "Value": mean_sd(row, "n_citations")},
        {"Measure": "Runs scored (checker succeeded)", "Value": "100 of 100 completed"},
        {"Measure": "Distinct documents cited / run", "Value": mean_sd(row, "n_docs_cited")},
        {"Measure": "Body words / run", "Value": mean_sd(row, "body_words")},
        {"Measure": "Word-count basis", "Value": "corrected_review_body"},
        {"Measure": "Pairwise citation-set Jaccard", "Value": fmt(stability["mean"]["new"], 3)},
        {"Measure": "Core documents (>=80% of runs)", "Value": "10"},
    ]


def update_t2_row(row, condition):
    result = dict(row)
    result.update({
        "N (scored/run)": f"{condition['new_scored_reviews']}/{condition['attempts']}",
        "Refusal %": "0.0",
        "No-output %": "0",
        "Citations": mean_sd(condition, "n_citations"),
        "E1 (out-of-corpus)": mean_sd(condition, "e1"),
        "E2 (invalid page)": mean_sd(condition, "e2"),
        "E3 (format)": mean_sd(condition, "e3"),
        "E4 (unverified quotation)": mean_sd(condition, "e4"),
        "E5 (misattr. quote)": mean_sd(condition, "e5"),
        "Docs cited": mean_sd(condition, "n_docs_cited"),
        "Body words": mean_sd(condition, "body_words"),
        "Word metric": "corrected_review_body",
        "Zero-error runs %": fmt(condition["new_clean_pct"], 1),
    })
    return result


def build_t2(condition_map, current_rows):
    rows = []
    for row in current_rows:
        label = row["Condition"]
        if label == "Full RRR (A)":
            rows.append(update_t2_row(row, condition_map["phase_a_main"]))
        elif label.startswith("Validation-gate ablation"):
            rows.append(update_t2_row(row, condition_map["phase_b_validation_off"]))
        elif label.startswith("RRR as a Claude skill"):
            continue
        elif "frontier model" in label or "unpinned" in label:
            continue
        else:
            rows.append(dict(row))
    return rows


def build_t4(guardrails):
    return [{
        "Guardrail layer": row["guardrail"],
        "Runs where fired": f"{fmt(row['new_runs_fired_pct'], 1)}%",
        "Total events": row["new_events"],
        "Mean/run": fmt(row["new_mean_per_run"]),
        "Max/run": row["new_max_per_run"],
    } for row in guardrails]


def build_body_words(condition_map, current_rows, paired_groups):
    keys = {
        "Full RRR": "phase_a_main",
        "Selected-check ablation": "phase_b_validation_off",
    }
    rows = []
    for old in current_rows:
        row = dict(old)
        key = keys.get(row["Condition"])
        if key:
            source = condition_map[key]
            paired = paired_groups[key]
            row.update({
                "Raw files": source["new_scored_reviews"],
                "Raw reviews": source["new_scored_reviews"],
                "Empty files ignored": "0",
                "Reference headings removed": sum(
                    item["new_reference_heading_removed"] == "True" for item in paired
                ),
                "Archived mean": source["new_body_words_mean"],
                "Archived SD": source["new_body_words_sd"],
                "Body-only mean": source["new_body_words_mean"],
                "Body-only SD": source["new_body_words_sd"],
                "Body min": source["new_body_words_min"],
                "Body max": source["new_body_words_max"],
                "Metric basis": "corrected_review_body",
            })
        rows.append(row)
    return rows


def build_t5(condition_map, current_rows):
    keys = {
        "Flagship (well-supported)": "phase_a_main",
        "Gibberish topic (D0, deterministic gate)": "phase_d0_gibberish",
        "High threshold (D1)": "phase_d1_high_threshold",
        "Narrow prompt (D2)": "phase_d2_narrow_prompt",
        "Marginal fit (D3)": "phase_d3_marginal_fit",
        "Nuanced topic: gender (T1)": "phase_t1_gender",
        "Nuanced topic: colonial (T2)": "phase_t2_colonial",
    }
    output = []
    for old in current_rows:
        source = condition_map[keys[old["Topic condition"]]]
        attempts = int(source["attempts"])
        row = {
            "Topic condition": old["Topic condition"],
            "N": attempts,
            "Invalid input %": round(
                100 * int(source["new_deterministic_input_gates"]) / attempts, 1
            ),
            "Substantive refusal %": round(
                100 * int(source["new_substantive_refusals"]) / attempts, 1
            ),
            "Structured failure %": round(
                100 * int(source["new_structured_failures"]) / attempts, 1
            ),
        }
        for key in (
            "Admission share",
            "Probe coverage",
            "Represented share",
            "Mean evidence score",
            "Warning rate %",
            "Top refusal reasons",
        ):
            row[key] = old[key]
        output.append(row)
    return output


def build_ts(condition_map):
    rows = []
    for documents, key in ((10, "phase_s10_docs"), (25, "phase_s25_docs"), (50, "phase_s50_docs")):
        source = condition_map[key]
        rows.append({
            "Corpus docs": documents,
            "N": source["attempts"],
            "Refusal %": "0.0",
            "E1": mean_sd(source, "e1"),
            "E4": mean_sd(source, "e4"),
            "Zero-error %": fmt(source["new_clean_pct"], 1),
            "Docs cited": mean_sd(source, "n_docs_cited"),
            "Body words": mean_sd(source, "body_words"),
            "Word metric": "corrected_review_body",
        })
    return rows


def build_stability(stability_rows):
    values = {row["metric"]: row["new"] for row in stability_rows}
    return [{
        "pairs": int(float(values["pairs"])),
        "mean": fmt(values["mean"], 3),
        "sd": fmt(values["sd"], 3),
        "min": fmt(values["min"], 3),
        "p05": fmt(values["p05"], 3),
        "p25": fmt(values["p25"], 3),
        "median": fmt(values["median"], 3),
        "p75": fmt(values["p75"], 3),
        "p95": fmt(values["p95"], 3),
        "max": fmt(values["max"], 3),
    }]


def build_core(core_rows, label_map):
    selected = [row for row in core_rows if row["generation"] == "new" and row["core_at_80_pct"] == "True"]
    selected.sort(key=lambda row: (-int(row["reviews"]), label_map.get(row["doc_id"], row["doc_id"])))
    return [{
        "doc_id": row["doc_id"],
        "Document": label_map.get(row["doc_id"], row["doc_id"]),
        "Reviews": row["reviews"],
        "Share %": row["share_pct"],
    } for row in selected]


def pivot_inclusion(source_rows, current_rows, conditions, columns):
    labels = {row["doc_id"]: row["document"] for row in current_rows}
    values = {(row["condition"], row["doc_id"]): row for row in source_rows}
    docs = sorted(
        {row["doc_id"] for row in source_rows if row["condition"] in conditions},
        key=lambda doc: labels.get(doc, doc).casefold(),
    )
    output = []
    for doc in docs:
        if not any(int(values.get((condition, doc), {}).get("new_reviews", 0)) for condition in conditions):
            continue
        row = {"doc_id": doc, "document": labels.get(doc, doc)}
        for condition, prefix in zip(conditions, columns):
            item = values.get((condition, doc), {})
            row[f"{prefix}_runs"] = item.get("new_reviews", 0)
            row[f"{prefix}_share_pct"] = item.get("new_share_pct", 0.0)
        output.append(row)
    return output


def build_accumulation(rows):
    return [{key: value for key, value in row.items() if key != "generation"}
            for row in rows if row["generation"] == "new"]


def build_model_outcomes(rows, current_f5, paired):
    current_lookup = {(row["model_tag"], row["probe"]): row for row in current_f5}
    reference_counts = {}
    for row in paired:
        reference_counts.setdefault((row["model"], row["condition"]), 0)
        reference_counts[(row["model"], row["condition"])] += (
            row["new_reference_heading_removed"] == "True"
        )
    output = []
    for row in rows:
        current = current_lookup[(row["model_tag"], row["probe"])]
        output.append({
            "model_tag": row["model_tag"],
            "model": row["model"],
            "parameters_b": current["parameters_b"],
            "probe": row["probe"],
            "probe_label": PROBE_LABELS[row["probe"]],
            "attempts": row["attempts"],
            "clean_reviews": row["new_clean_reviews"],
            "reviews_with_error": row["new_reviews_with_error"],
            "substantive_refusals": row["new_substantive_refusals"],
            "structured_failures": row["new_structured_failures"],
            "postcomposition_failures": row["new_postcomposition_failures"],
            "execution_failures": (
                int(row["new_structured_failures"])
                + int(row["new_postcomposition_failures"])
            ),
            "scored_reviews": int(row["new_clean_reviews"]) + int(row["new_reviews_with_error"]),
            "mean_e1": row["new_e1_mean"],
            "mean_e2": row["new_e2_mean"],
            "mean_e3": row["new_e3_mean"],
            "mean_e4": row["new_e4_mean"],
            "mean_e5": row["new_e5_mean"],
            "mean_words": row["new_body_words_mean"],
            "sd_words": row["new_body_words_sd"],
            "word_metric_basis": "corrected_review_body" if row["new_body_words_mean"] else "",
            "mean_documents": row["new_n_docs_cited_mean"],
            "mean_citations": row["new_n_citations_mean"],
        })
    order = {model: index for index, model in enumerate(MODEL_ORDER)}
    probe_order = {probe: index for index, probe in enumerate(PROBE_ORDER)}
    output.sort(key=lambda row: (order[row["model_tag"]], probe_order[row["probe"]]))
    return output


def build_model_figure(rows, figures: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Patch
    from matplotlib.textpath import TextPath

    colors = {
        "clean_reviews": "#2E7D32",
        "reviews_with_error": "#E69F00",
        "substantive_refusals": "#4C78A8",
        "execution_failures": "#B44E52",
    }
    labels = {
        "clean_reviews": "Clean review",
        "reviews_with_error": "Review with error",
        "substantive_refusals": "Substantive refusal",
        "execution_failures": "Execution failure",
    }
    by_cell = {(row["model_tag"], row["probe"]): row for row in rows}
    model_labels = [by_cell[(tag, "flagship")]["model"] for tag in MODEL_ORDER]
    panels = [
        ("flagship", "A. Institutional-growth task (n=25)"),
        ("wordsalad", "B. Word salad (n=20)"),
        ("far", "C. Far-domain physics (n=20)"),
        ("mid", "D. Literary modernism (n=20)"),
        ("near", "E. Soviet-planning near miss (n=20)"),
    ]
    y = list(range(len(MODEL_ORDER)))
    rc = {
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with matplotlib.rc_context(rc):
        figure = plt.figure(figsize=(6.25, 8.0), facecolor="white")
        grid = figure.add_gridspec(
            3, 2, left=0.205, right=0.985, top=0.885, bottom=0.070,
            hspace=0.28, wspace=0.11,
        )
        axes = [
            figure.add_subplot(grid[0, :]),
            figure.add_subplot(grid[1, 0]),
            figure.add_subplot(grid[1, 1]),
            figure.add_subplot(grid[2, 0]),
            figure.add_subplot(grid[2, 1]),
        ]
        pending = []
        for panel_index, (axis, (probe, title)) in enumerate(zip(axes, panels)):
            left = [0.0] * len(MODEL_ORDER)
            for outcome, color in colors.items():
                counts = [int(by_cell[(tag, probe)][outcome]) for tag in MODEL_ORDER]
                attempts = [int(by_cell[(tag, probe)]["attempts"]) for tag in MODEL_ORDER]
                widths = [100.0 * count / total for count, total in zip(counts, attempts)]
                bars = axis.barh(y, widths, left=left, color=color, height=0.74, linewidth=0)
                for bar, count, width in zip(bars, counts, widths):
                    if count:
                        pending.append((axis, bar, count, width, outcome))
                left = [first + second for first, second in zip(left, widths)]
            axis.set_xlim(0, 100)
            axis.set_ylim(-0.7, len(MODEL_ORDER) - 0.3)
            axis.invert_yaxis()
            axis.set_title(title, loc="left", fontsize=8.0, fontweight="bold", pad=4.5)
            axis.set_xticks([0, 25, 50, 75, 100])
            axis.tick_params(axis="x", labelsize=8.0, length=2.6, pad=2.2)
            axis.tick_params(axis="y", length=0, pad=3.0)
            axis.grid(axis="x", color="#D8D8D8", linewidth=0.55)
            axis.set_axisbelow(True)
            for boundary in (3.5, 6.5):
                axis.axhline(boundary, color="#777777", linewidth=0.55)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_yticks(y)
            axis.set_yticklabels(model_labels if panel_index in (0, 1, 3) else [], fontsize=8.1)
        figure.canvas.draw()
        count_font = FontProperties(family="DejaVu Sans", weight="bold", size=8.0)
        for axis, bar, count, width, outcome in pending:
            segment_width = (width / 100.0) * axis.bbox.width * 72.0 / figure.dpi
            text_width = TextPath((0, 0), str(count), prop=count_font).get_extents().width
            if segment_width >= text_width + 5.5:
                color = "#1F1F1F" if outcome == "reviews_with_error" else "white"
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(count), ha="center", va="center", color=color,
                    fontsize=8.0, fontweight="bold", clip_on=True,
                )
        handles = [Patch(facecolor=colors[key], edgecolor="none", label=labels[key]) for key in colors]
        figure.suptitle(
            "Local-model outcomes by attempted-run denominator",
            x=0.595, y=0.978, fontsize=10.6, fontweight="bold",
        )
        figure.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.595, 0.947),
            ncol=4, frameon=False, fontsize=8.2, handlelength=1.05,
            handleheight=0.75, handletextpad=0.38, columnspacing=0.82,
            borderaxespad=0.0,
        )
        figure.supxlabel("Share of attempted runs (%)", x=0.595, y=0.014, fontsize=8.7)
        figures.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            figures / "F5_local_model_outcomes.pdf",
            facecolor="white",
            metadata=PDF_METADATA,
        )
        figure.savefig(figures / "F5_local_model_outcomes.png", dpi=300, facecolor="white")
        plt.close(figure)


def build_corpus_figure(ts_rows, figures: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    documents = [int(row["Corpus docs"]) for row in ts_rows]
    cited = [float(row["Docs cited"].split()[0]) for row in ts_rows]
    clean = [float(row["Zero-error %"]) for row in ts_rows]
    figure, first = plt.subplots(figsize=(6.5, 4.2))
    second = first.twinx()
    first.plot(documents, cited, "o-", color="#4C78A8", label="Documents cited")
    second.plot(documents, clean, "s--", color="#2E7D32", label="Clean reviews")
    first.set_xlabel("Documents in declared corpus")
    first.set_ylabel("Mean documents cited")
    second.set_ylabel("Reviews free of E1-E5 (%)")
    first.set_xticks(documents)
    first.grid(axis="y", color="#D8D8D8", linewidth=0.55)
    handles = first.get_lines() + second.get_lines()
    first.legend(handles, [line.get_label() for line in handles], frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(figures / "FS_corpus_size.pdf", metadata=PDF_METADATA)
    figure.savefig(figures / "FS_corpus_size.png", dpi=300)
    plt.close(figure)


def build_stability_figure(accumulation, figures: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [int(row["runs_pooled"]) for row in accumulation]
    y = [float(row["mean_core_recovery_pct"]) for row in accumulation]
    low = [float(row["p05_core_recovery_pct"]) for row in accumulation]
    high = [float(row["p95_core_recovery_pct"]) for row in accumulation]
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot(x, y, "o-", color="#4C78A8")
    axis.fill_between(x, low, high, color="#4C78A8", alpha=0.2)
    axis.set_xlabel("Runs pooled")
    axis.set_ylabel("Core documents recovered (%)")
    axis.set_xticks(x)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.55)
    figure.tight_layout()
    figure.savefig(figures / "F3_stability_vs_k.pdf", metadata=PDF_METADATA)
    figure.savefig(figures / "F3_stability_vs_k.png", dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-source",
        default=ROOT / "results" / "corrected" / "analysis_source",
        type=Path,
    )
    parser.add_argument(
        "--frozen-bundle",
        default=ROOT / "results" / "rebuild_inputs" / "pre_correction_snapshot",
        type=Path,
    )
    parser.add_argument(
        "--out",
        default=ROOT / "results" / "corrected",
        type=Path,
    )
    args = parser.parse_args()

    analysis = args.analysis_source.resolve()
    frozen = args.frozen_bundle.resolve()
    out = args.out.resolve()
    tables = out / "tables"
    figures = out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    conditions = read_csv(analysis / "condition_changes.csv")
    condition_map = {row["condition"]: row for row in conditions}
    paired = read_csv(analysis / "paired_writer_runs.csv")
    paired_groups = paired_by_condition(paired)
    guardrails = read_csv(analysis / "guardrail_changes.csv")
    stability_changes = read_csv(analysis / "stability_distribution_changes.csv")
    core_changes = read_csv(analysis / "core_document_changes.csv")
    inclusion_changes = read_csv(analysis / "source_inclusion_changes.csv")
    accumulation_changes = read_csv(analysis / "source_accumulation_changes.csv")
    model_changes = read_csv(analysis / "model_outcome_changes.csv")

    frozen_tables = frozen / "tables"
    current_t0 = read_csv(frozen_tables / "T0_controlled_contract_staircase.csv")
    current_t2 = read_csv(frozen_tables / "T2_conditions.csv")
    current_words = read_csv(frozen_tables / "T_body_only_word_counts.csv")
    current_t5 = read_csv(frozen_tables / "T5_refusal_calibration.csv")
    current_topic_inclusion = read_csv(frozen_tables / "T_source_inclusion_topics.csv")
    current_prompt_inclusion = read_csv(frozen_tables / "F_document_inclusion_by_prompt.csv")
    current_f5 = read_csv(frozen_tables / "F5_local_model_outcomes.csv")

    t0 = build_t0(condition_map, current_t0, paired_groups)
    t1 = build_t1(condition_map, stability_changes)
    t2 = build_t2(condition_map, current_t2)
    t4 = build_t4(guardrails)
    body_words = build_body_words(condition_map, current_words, paired_groups)
    t5 = build_t5(condition_map, current_t5)
    ts = build_ts(condition_map)
    stability = build_stability(stability_changes)
    labels = {row["doc_id"]: row["document"] for row in current_topic_inclusion}
    core = build_core(core_changes, labels)
    topic_inclusion = pivot_inclusion(
        inclusion_changes,
        current_topic_inclusion,
        ["phase_a_main", "phase_t1_gender", "phase_t2_colonial"],
        ["flagship", "gender", "colonial"],
    )
    prompt_inclusion = pivot_inclusion(
        inclusion_changes,
        current_prompt_inclusion,
        [
            "phase_a_main",
            "phase_t1_gender",
            "phase_t2_colonial",
            "phase_f1_adversarial_irrelevant",
            "phase_f2_adversarial_culture",
        ],
        ["flagship", "gender", "colonial", "irrelevance", "culture"],
    )
    accumulation = build_accumulation(accumulation_changes)
    model_outcomes = build_model_outcomes(model_changes, current_f5, paired)
    workflow = [
        row for row in read_csv(frozen_tables / "T_workflow_comparison.csv")
        if not row["workflow"].startswith("RRR skill")
    ]

    generated_tables = {
        "T0_controlled_contract_staircase": (t0, "Performance under successively stronger evidence contracts"),
        "T1_main_integrity": (t1, "Main corrected RRR condition"),
        "T2_conditions": (t2, "Corrected architecture and workflow conditions"),
        "T4_guardrail_activation": (t4, "Guardrail activation in 100 corrected full-RRR runs"),
        "T_body_only_word_counts": (body_words, "Corrected body-only word counts"),
        "T5_refusal_calibration": (t5, "Refusal calibration with separated run states"),
        "TS_corpus_size": (ts, "Corrected nested-corpus results"),
        "T_stability_distribution": (stability, "Pairwise stability of corrected cited-document sets"),
        "T_stability_core": (core, "Core documents in corrected full-RRR reviews"),
        "T_source_inclusion_topics": (topic_inclusion, "Corrected source inclusion across historical questions"),
        "F_document_inclusion_by_prompt": (prompt_inclusion, "Corrected document inclusion by prompt"),
        "T_source_accumulation_data": (accumulation, "Corrected source accumulation data"),
        "T_workflow_comparison": (workflow, "Eligible writer-independent workflow comparison"),
        "F5_local_model_outcomes": (model_outcomes, "Corrected local-model outcomes"),
    }
    for stem, (rows, caption) in generated_tables.items():
        write_csv(tables / f"{stem}.csv", rows)
        write_generic_tex(tables / f"{stem}.tex", rows, caption)

    presentation_accumulation = [{
        "Runs pooled": row["runs_pooled"],
        "Mean distinct works": row["mean_distinct_works"],
        "P5--P95": f"{row['p05_distinct_works']}--{row['p95_distinct_works']}",
        "Mean core recovery %": row["mean_core_recovery_pct"],
        "P5--P95 %": f"{row['p05_core_recovery_pct']}--{row['p95_core_recovery_pct']}",
    } for row in accumulation]
    write_csv(tables / "T_source_accumulation.csv", presentation_accumulation)
    write_generic_tex(
        tables / "T_source_accumulation.tex",
        presentation_accumulation,
        "Source accumulation as corrected full-RRR runs are pooled",
    )

    for stem in ("T3_citation_taxonomy", "T3_citation_taxonomy_detail"):
        copy_independent_table(frozen_tables, tables, stem)

    build_model_figure(model_outcomes, figures)
    build_corpus_figure(ts, figures)
    build_stability_figure(accumulation, figures)

    expected_model_cells = len(MODEL_ORDER) * len(PROBE_ORDER)
    checks = {
        "phase_a_clean_reviews": int(condition_map["phase_a_main"]["new_clean_reviews"]) == 100,
        "phase_a_parsed_citations": round(
            float(condition_map["phase_a_main"]["new_n_citations_mean"]) * 100
        ) == 3184,
        "rag_baseline_unchanged": next(
            row for row in t0 if row["condition"] == "Single-pass retrieval"
        )["zero_e1_e5_pct"] == "87.0",
        "h3_excluded": all("RRR skill" not in row["workflow"] for row in workflow),
        "model_cells": len(model_outcomes) == expected_model_cells,
        "stability_core_count": len(core) == 10,
        "topic_inclusion_rows": len(topic_inclusion) == 35,
        "contract_validation_passed": json.loads(
            (analysis / "validation.json").read_text(encoding="utf-8")
        )["all_passed"],
    }
    if not all(checks.values()):
        raise ValueError(f"corrected artifact validation failed: {checks}")

    sources = sorted(path for path in analysis.rglob("*") if path.is_file())
    generated = sorted(
        path.relative_to(out).as_posix()
        for directory in (tables, figures)
        for path in directory.rglob("*") if path.is_file()
    )
    expected_generated = {
        f"tables/{stem}.{extension}"
        for stem in generated_tables
        for extension in ("csv", "tex")
    } | {
        "tables/T_source_accumulation.csv",
        "tables/T_source_accumulation.tex",
        "tables/T3_citation_taxonomy.csv",
        "tables/T3_citation_taxonomy.tex",
        "tables/T3_citation_taxonomy_detail.csv",
        "figures/F3_stability_vs_k.pdf",
        "figures/F3_stability_vs_k.png",
        "figures/F5_local_model_outcomes.pdf",
        "figures/F5_local_model_outcomes.png",
        "figures/FS_corpus_size.pdf",
        "figures/FS_corpus_size.png",
    }
    unexpected = set(generated) - expected_generated
    missing = expected_generated - set(generated)
    if unexpected or missing:
        raise ValueError(
            f"corrected bundle file set mismatch: unexpected={sorted(unexpected)}, "
            f"missing={sorted(missing)}"
        )
    manifest = {
        "schema_version": "corrected-paper-artifacts-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
        "population": {
            "corrected_writer_replays": 941,
            "preserved_independent_baselines": 300,
            "preserved_pre_writer_terminals": 1184,
        },
        "excluded": {
            "rrr_skill_h3": "Used the original writer outside the accepted replay population",
            "cost_latency_figure": "Corrected writer timings are not comparable with July full-pipeline timings",
        },
        "copied_writer_independent_tables": [
            "T3_citation_taxonomy",
            "T3_citation_taxonomy_detail",
        ],
        "validation": checks,
        "analysis_sources": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
            }
            for path in sources
        ],
        "generated_files": generated,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (out / "README.md").write_text(
        "# Corrected paper artifacts\n\n"
        "This bundle uses the 941 accepted corrected writer replays, the 300 "
        "hash-verified independent baselines, and the 1,184 preserved pre-writer "
        "terminal outcomes. Writer-dependent exhibits are regenerated from "
        "`analysis_source/`. T3 is copied from the frozen bundle because it uses "
        "the independent C and C2 baselines. T5 preserves its upstream topic-fit "
        "values while separating deterministic input gates, substantive refusals, "
        "and structured failures.\n\n"
        "H3 is excluded because it used the original writer. The cost and latency "
        "figure is omitted until a comparable timing construction is defined.\n\n"
        "Rebuild from the repository root with:\n\n"
        "```bash\n"
        "python scripts/build_corrected_paper_artifacts.py\n"
        "```\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"[corrected-artifacts] tables: {len(list(tables.glob('*.csv')))} CSV")
    print(f"[corrected-artifacts] figures: {len(list(figures.glob('*')))} files")
    print(f"[corrected-artifacts] validation: {len(checks)} passed")
    print(f"[corrected-artifacts] output: {out}")


if __name__ == "__main__":
    main()
