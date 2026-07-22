# Installing RRR

## Codex plugin

The RRR plugin contains the workflow and a verified universal Python wheel. On
its first use, it creates a private Python environment under the user's RRR data
directory and installs that wheel with its dependencies. It does not modify the
system Python installation.

After installing the plugin, start a new Codex task. Open a folder containing
the PDFs to analyse and enter a prompt such as:

```text
$rrr t2 How did institutions shape long-run economic growth?
```

The folder does not need to be named `corpus`. If Codex cannot identify one PDF
collection, it asks the user to select the folder. RRR keeps the original PDFs
in place and stores extracted text, indices, and run records in its private
user-data directory.

## Reviewer installation from the replication repository

The public replication repository is also a Codex marketplace. A reviewer can
register it and install RRR with:

```bash
codex plugin marketplace add igbaccin/rrr-replication --ref main
codex plugin add rrr@rrr-replication
```

After starting a new Codex task, the reviewer can open any folder containing
OCR-readable PDFs and enter `$rrr t2 <topic>` or `$rrr t1 <claim>`. The plugin
contains the verified RRR wheel and prepares its own private runtime on first
use. Reviewing the workflow therefore requires a signed-in Codex installation
and Python 3.10 or later. Ollama and provider API credentials are unnecessary
for native Codex mode.

The final revision archive should identify the replication commit used for the
submission. Reviewers who want that exact version can replace `main` in the
first command with the corresponding release tag.

## Standalone Python package

The release artifact is a platform-independent wheel under `dist/`, accompanied
by a SHA-256 checksum. It can be offered as a download from the project website
and installed with:

```bash
python -m pip install rrr_poc-15.17.0-py3-none-any.whl
```

Provider API users can install the optional clients with:

```bash
python -m pip install "rrr_poc-15.17.0-py3-none-any.whl[api]"
```

The installed command supports local Ollama, provider APIs, and native Codex or
Claude host mode. The website should offer the wheel and its `.sha256` file
together. Once the final download address is known, it can also provide a
single direct-URL `pip install` command.

## Requirements

RRR requires Python 3.10 or later and PDFs containing usable text or OCR. Native
Codex use requires a signed-in Codex installation. Local inference additionally
requires Ollama and the selected open-weight model.
