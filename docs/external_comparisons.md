# External workflow comparisons

## Completed Claude Code arms

The paper reports off-the-shelf Claude Code comparisons using the same extracted corpus text and citation instruction. These arms receive the evidence as files and written instructions. They do not receive the enforced RRR evidence contract.

The accepted summaries cover:

| Model | Attempts | Scored reviews |
| --- | ---: | ---: |
| Claude Haiku 4.5 | 50 | 49 |
| Claude Sonnet 4.5 | 50 | 50 |
| Claude Opus 4.8 | 51 | 51 |

Their run-level summaries are deposited under `results/external_comparisons/claude_code/`.

## Requirements

1. Python 3.10 or later.
2. Claude Code installed and authenticated.
3. The corpus preprocessing step completed, producing `data/page_text/`.

The arm reads extracted page text. It does not send the 50 source PDFs directly to Claude Code.

## Execution

Run a one-attempt smoke test:

```bash
N_ARM=1 ./scripts/run_claude_code_arm.sh
```

Run a pinned model arm:

```bash
N_ARM=50 ARM_MODEL=claude-haiku-4-5 ./scripts/run_claude_code_arm.sh
N_ARM=50 ARM_MODEL=claude-sonnet-4-5 ./scripts/run_claude_code_arm.sh
N_ARM=51 ARM_MODEL=claude-opus-4-8 ./scripts/run_claude_code_arm.sh
```

Model identifiers should be checked against current provider availability before rerunning. The deposited JSON files preserve the identifiers and outcomes used by the paper.

Each attempt is scored with `scripts/check_citations.py`, the same checker used for the RRR conditions. The scored dimensions are E1, out-of-corpus citations; E2, invalid pages; E3, citation-format violations; E4, unverified quotations; and E5, misattributed quotations.

## Corrected RRR skill arm

The corrected skill condition installs RRR as `/rrr` inside Claude Code. Claude
Code invokes the RRR pipeline, whose internal model calls are routed to Opus
4.8 through the API. All ten attempts completed and were scored. The 642 parsed
citations contained no E1 through E5 event, giving ten clean reviews. The two
eligible direct quotations both verified. The accepted summary is deposited at
`results/corrected/external_comparisons/rrr_skill/phase_h3_results.json` and
enters both corrected comparison tables.

The released skill can also select local Ollama or the subscription-backed
Claude and Codex host adapters. Those routes do not change the H3 result, which
records the API configuration used for the ten reported attempts.

Run the condition after obtaining the corpus PDFs and preprocessing them:

```bash
N_ARM=10 ARM_MODEL=claude-opus-4-8 ./scripts/run_claude_skill_arm.sh
```

The earlier skill runs remain historical because they used the superseded
writer. They do not enter the accepted analytical bundle.

## NotebookLM status

`results/external_comparisons/notebooklm_pilot/` preserves a single provisional pilot, its runbook, and its scoring sheet. The final human-run NotebookLM protocol remains pending. The pilot is labelled provisional in `results/corrected/tables/T_workflow_comparison.csv` and should not be interpreted as a completed comparison arm.
