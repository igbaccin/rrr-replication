# NotebookLM pilot runbook

## Current status

One pilot response was collected on 10 July 2026. It contains 1,275 body words, 44 model-written source-page pairs, and 11 distinct cited works. The user-interface citation-chip count was not recorded, so this pilot does not satisfy the complete comparison protocol.

At least four additional independent runs and the missing user-interface observations remain pending. The deposited pilot is therefore provisional.

## One-time setup

1. Open NotebookLM and create a notebook named `RRR-H2`.
2. Upload the 50 legally obtained corpus PDFs listed in `bibliography.bib`.
3. Wait until every source has completed processing.

## Per-run protocol

1. Start a fresh chat. Record whether a new notebook or a cleared chat was used.
2. Paste the following prompt exactly, in one message, with no follow-up:

   > Write a literature review of 900-1500 words on the topic: 'Institutions are the fundamental cause of long-run economic growth.' Support every claim with citations to the uploaded sources, including the specific page where the supporting passage appears. Do not use any knowledge beyond the uploaded sources. If the sources cannot support the review, say so.

3. Before copying the response, record the number of citation chips shown in the interface.
4. Record whether the interface citations display page numbers.
5. Open five citation chips selected across the response and record whether each highlighted source passage supports the associated claim.
6. Save the unedited response under `arm_outputs/notebooklm/run_NN.md`.
7. Add the run to `arm_outputs/notebooklm/scoring_sheet.csv` with the following fields:
   - run identifier;
   - body words;
   - interface citation-chip count;
   - model-written page-pair count;
   - distinct sources;
   - out-of-corpus references;
   - five-chip support tally;
   - refusal or hedging behavior;
   - audit-trail note.

## Interface limitation

Citation chips can disappear when a response is copied. The chip count and five-chip support audit must therefore be recorded in the interface before saving the response text.

## Completion rule

The arm becomes eligible for the final workflow comparison after at least five independent runs have complete interface observations, saved response text, and a filled scoring sheet. Until then, every table and narrative reference must identify it as a pilot.
