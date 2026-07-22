# Replication Package for Retrieval-Restricted Reasoning

This repository accompanies the paper, "Retrieval-Restricted Reasoning: A Proof of Concept for Adapting Language Models to Economic History."

The package is synchronized with the corrected writer contract and the accepted results bundle dated 22 July 2026. It contains the current RRR implementation, the evaluation programs, run-level analytical summaries, generated exhibits, and the frozen input archive for the 941 corrected writer replays.

The corrected RRR skill comparison is included in the accepted tables. Its ten completed runs were all free of E1 through E5 under the common checker. The NotebookLM comparison remains a documented pilot pending the final human-run protocol.

## Fast verification

The deposited results can be checked without the copyrighted corpus or an installed language model.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python scripts/verify_replication.py
python -m unittest discover -s tests/unit -v
```

To regenerate the corrected tables and figures from the deposited analysis source:

```bash
python scripts/build_corrected_paper_artifacts.py --out reproduced_artifacts
```

The script validates the expected populations and writes the regenerated exhibits under `reproduced_artifacts/`.

## Package contents

| Path | Contents |
| --- | --- |
| `src/rrr/` | RRR pipeline source used by the corrected writer replay |
| `.agents/skills/rrr/` | Repository-scoped Codex skill, invoked explicitly as `$rrr` |
| `.agents/plugins/marketplace.json` | Local marketplace entry for the installable Codex plugin |
| `plugins/rrr/` | Codex plugin that distributes the `$rrr` skill |
| `skills/rrr/` | Portable skill source for Codex and Claude Code |
| `dist/` | Standalone universal wheel and SHA-256 checksum |
| `scripts/` | Corpus preparation, batteries, scoring, replay, and exhibit builders |
| `tests/unit/` | Regression tests for the evidence contract and all three model runtimes |
| `results/corrected/analysis_source/` | Run-level analytical records used to construct the paper exhibits |
| `results/corrected/tables/` | Corrected tables in CSV and TeX formats |
| `results/corrected/figures/` | Corrected figures in PDF and PNG formats |
| `results/corrected/external_comparisons/rrr_skill/` | Accepted corrected H3 result record |
| `results/replay_inputs/` | Compressed frozen ledgers for the 941 writer replays |
| `results/external_comparisons/` | Completed Claude Code summaries and the provisional NotebookLM pilot record |
| `results/rebuild_inputs/` | Clearly labelled frozen table inputs required by the corrected exhibit builder |
| `docs/` | Replay, RunPod, and external-comparison instructions |
| `REPLICATION_MANIFEST.json` | Package status, population accounting, exclusions, and integrity hashes |

## Using RRR from Codex or Claude Code

The casual workflow begins by installing the RRR plugin and opening a folder
containing PDFs. The folder can have any name. The user then types `$rrr t2`
followed by a topic. If RRR cannot identify one PDF collection, Codex asks the
user to select it. The plugin installs its verified bundled wheel in a private
RRR runtime on first use.

The skill defaults to the product in which it is invoked. `$rrr` uses Codex,
while `/rrr` uses Claude Code. The selected runtime determines where every
internal RRR model call is executed.

| Runtime | Model execution | Account or hardware |
| --- | --- | --- |
| Local | Mistral or Qwen through Ollama | The user's machine or a user-controlled Ollama server |
| Provider API | Anthropic or OpenAI API | A separate provider API account |
| Native host | Codex or Claude Code subprocesses | The user's ChatGPT or Claude subscription allowance |

When this checkout is open, the skill installs the local package if necessary.
An installed plugin uses its bundled wheel, so the user does not need to run a
package-installation command. Manual installations remain available for direct
CLI use.

```bash
python -m pip install -e .
python -m pip install -e ".[api]"
```

The same wheel can be downloaded from `dist/` for publication on the project
website. Installation details are in `docs/product_installation.md`.

Codex discovers `.agents/skills/rrr/` while this checkout is open. Use
`$rrr t2 <topic>` for a literature review or `$rrr t1 <claim>` for a claim
evaluation. `$rrr <topic>` defaults to T2. These are prompts, so the topic or
claim does not need shell quotation marks. To install the skill for use outside
this checkout, register the bundled marketplace and install the plugin.

```bash
codex plugin marketplace add igbaccin/rrr-replication --ref main
codex plugin add rrr@rrr-replication
```

This is also the reviewer installation route. The final revision archive will
identify a release tag that can replace `main` when an exact deposited version
is required. Start a new Codex task after installation. Claude Code users can copy
`skills/rrr/` to `.claude/skills/rrr/` in a workspace and use the corresponding
`/rrr` forms. When the current project contains an existing RRR index, the skill
uses it automatically. New collections are prepared under the user's RRR data
directory, while the PDFs remain in their original folder.

An explicit skill invocation selects the native host and needs no second
runtime confirmation. Ask for `local` or `ollama` to use a local model. Ask for
`api anthropic` or `api openai` to use a separate provider account. A direct
Python or `rrr` CLI invocation with `RRR_RUNTIME` unset continues to use local
Ollama.

Provider API mode is enabled with `RRR_RUNTIME=api` and
`RRR_API_PROVIDER=anthropic` or `openai`. The optional `RRR_API_MODEL` selects a
different Anthropic model or a standard GPT-5.5 or GPT-5.6 model. The provider
SDK reads `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; RRR never accepts a key as a
command argument. OpenAI Responses use `store=false` unless `RRR_API_STORE=1`
is set. This disables application-state response storage. Provider retention
and abuse-monitoring policies still apply.

Native host mode is enabled with `RRR_RUNTIME=host` and `RRR_HOST=codex` or
`claude`. Run `rrr host-doctor --host codex` or
`rrr host-doctor --host claude` before the review. This checks the executable
and subscription authentication. Add `--smoke` to spend one subscription call
on an end-to-end JSON check. The doctor accepts a ChatGPT product login for
Codex and a Claude.ai Pro, Max, Team, or Enterprise login for Claude. Claude
host mode requires Claude Code 2.1.169 or newer. Provider API authentication is
rejected. Each internal model call starts in a fresh tool-restricted subprocess
with a narrow environment allowlist. An already indexed T2 review usually
requires about 35 to 40 such calls, all of which draw on the selected product
allowance. The outer Codex or Claude task also uses its normal subscription
allowance while it coordinates the command.

The skill sets native host mode automatically. `$rrr` selects Codex and
`/rrr` selects Claude Code. An explicit runtime request overrides this default.

In provider API and native host modes, RRR sends the instructions and admitted
evidence passages required for each model call. The PDFs and local retrieval
index remain in the corpus workspace. Local mode keeps the model calls there as
well. During corpus ingestion, add `--no-llm` in API or host mode unless the
optional metadata-extraction calls have been approved. Host requests and
responses are recorded locally by default. If an enabled audit cannot be
written, RRR does not deliver the model output.

After obtaining the corpus PDFs and installing Claude Code, the corrected H3
condition can be repeated with `scripts/run_claude_skill_arm.sh`. The runner
installs the package, invokes `/rrr`, and scores each released review with the
deposited checker. That experiment used the Anthropic API runtime. The native
host adapter is a new delivery route and is not part of the H3 evidence.

## Result population

The corrected analytical source classifies 2,425 attempts.

| Population | Attempts | Treatment |
| --- | ---: | --- |
| Runs that reached the writer | 941 | Regenerated under the corrected evidence contract |
| Independent B2, C, and C2 baselines | 300 | Preserved with verified hashes because they do not use the RRR writer |
| Runs ending before the writer | 1,184 | Preserved with their original terminal states |
| Total | 2,425 | Complete classified source population |

The 941 regenerated reviews are stochastic continuations from frozen upstream ledgers. Retrieval, document admission, retained passages, topic-fit decisions, and outline state are held fixed for this correction. The replay identifies the corrected writer boundary; it does not estimate a causal treatment effect between the two generations.

The main corrected RRR condition contains 100 completed and scored reviews. All 100 have zero E1 through E5 events under the deposited checker. The complete condition inventory and run counts are recorded in `results/corrected/analysis_source/condition_changes.csv`.

## Verifying the writer replay inputs

The replay archive is compressed because its 941 ledgers expand to approximately 299 MiB.

```bash
mkdir -p replay_inputs
tar -xzf results/replay_inputs/writer_correction_260720.tar.gz -C replay_inputs
python scripts/run_corrected_writer_replay.py \
  --bundle replay_inputs/corrected_writer_v17 \
  --profile smoke \
  --list
```

Execute the smoke replay after starting Ollama and installing the required model:

```bash
python scripts/run_corrected_writer_replay.py \
  --bundle replay_inputs/corrected_writer_v17 \
  --profile smoke
```

The full replay can be divided across workers with `--shard-count` and `--shard-index`. See `docs/writer_correction_replay.md` and `docs/runpod_replay.md`.

## Re-executing the full pipeline

The 50 source PDFs are excluded because most are copyrighted. `bibliography.bib` and `metadata_reference.csv` identify the required documents and expected metadata. Place legally obtained copies under `corpus/` using the filenames recorded in `metadata_reference.csv`.

The commands in this section reproduce the paper's local-model evaluation and
therefore require Ollama and the recorded Mistral or Qwen model tags. The API
and native host runtimes can run the interactive RRR pipeline without an
Ollama server.

Preprocess the corpus:

```bash
chmod +x scripts/preprocess.sh
./scripts/preprocess.sh
```

Run the 1,010-attempt core block:

```bash
chmod +x scripts/run_battery.sh
./scripts/run_battery.sh
```

Run the model-ladder, topic, and corpus-size blocks:

```bash
PHASES="G,T,S" ./scripts/run_comprehensive_battery.sh
```

Together these blocks reproduce the 2,425-attempt design. Model generation is stochastic, so regenerated prose and some secondary metrics will vary. The reference summaries, exact accepted run counts, and checker outputs are deposited under `results/corrected/`.

## External comparison arms

The paper reports completed off-the-shelf Claude Code comparisons for Haiku 4.5, Sonnet 4.5, and Opus 4.8. Their run-level result summaries are under `results/external_comparisons/claude_code/`. The corrected `/rrr` skill condition is deposited under `results/corrected/external_comparisons/rrr_skill/`. Its ten runs produced ten clean reviews across 642 parsed citations. `docs/external_comparisons.md` records both execution procedures.

The NotebookLM directory currently preserves one pilot response, its runbook, and its scoring sheet. It is marked provisional throughout the package. The final NotebookLM exercise will be added after the human-run protocol is completed.

## Reproducibility boundaries

- Corpus PDFs must be obtained by the replicator.
- Local model tags identify the tested model tiers. Availability and upstream model packaging can change over time.
- Language-model outputs are stochastic.
- Corrected writer timings are excluded because replay timings are not comparable with the original full-pipeline timings.
- The historical RRR skill arm is excluded because it used the superseded writer. The corrected replacement has a separate accepted run record.
- Original commit hashes embedded in replay records refer to the private archival history. They remain historical provenance identifiers.

## Citation and licence

Please cite the paper when using the package. The source code is distributed under the terms in `LICENSE.txt`.

Questions about replication can be directed to Igor Martins at `igor.martins@ekh.lu.se`.
