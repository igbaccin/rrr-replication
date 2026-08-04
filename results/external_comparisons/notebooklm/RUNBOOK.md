# Completion update, 2026-07-28

The protocol is complete. Five independent NotebookLM runs are preserved as
`run_01.md` through `run_05.md`. The scoring sheet records 176 interface
citation chips and 178 plain page-looking pairs across the five runs. The
common body-word counter gives a mean of 1,286.6 words with a sample standard
deviation of 175.13. A separate five-claim page check found support for three
claims. See `verification_sample.md` for the claim-level record.

# NotebookLM (Phase H2) — runbook

## Original checkpoint from 2026-07-10

- Notebook exists, all 50 PDFs uploaded. Run 01 done and saved (`run_01.md`).
- Run 01 mechanical: 1,275 words; 44 model-written citation pairs `[N, printed-page]`;
  11 distinct works cited; out-of-corpus 0.
- KEY FINDING (screenshot-confirmed): two disconnected citation layers —
  clickable chips link to WHOLE papers (no page, no passage); the `[N, page]`
  pairs are UNLINKED plain text the model wrote. Chips vanish on copy.
  Neither layer gives a checkable claim→source→page unit. Three pairs look
  mis-anchored: [384, 3221] (Nunn-2008 claim → Nunn&Wantchekon first page),
  [220, 917] (page outside Austin 2008's range), [370, 1328] (N&W claim →
  North 1989's range).
- REVISED DIVISION OF LABOR: human does ONLY the UI loop per run (~3 min:
  fresh chat → paste prompt → rough chip count → copy answer to run_NN.md).
  Claude verifies the claimed printed pages offline against
  `data/page_text/` (no UI needed), fills the scoring sheet, aggregates.
- This checkpoint was superseded by the completion update above.

Protocol: `scripts/claude_arm/NOTEBOOKLM_PROTOCOL.md` (frozen). N = 5–10 runs,
~2h total. One-time setup, then ~10 min per run.

## One-time setup

1. Go to notebooklm.google.com, sign in with a Google account (free tier is
   enough: 50 sources per notebook = exactly our corpus).
2. "Create new" notebook → name it `RRR-H2`.
3. Upload all 50 PDFs from `<path-to-corpus>` (drag-and-drop the whole
   selection). Wait until every source shows as processed (no spinner).

## Per run (run 01 … run NN)

1. Start a FRESH chat: three-dot menu (or the chat panel) → clear/refresh
   chat history. Do not reuse a chat that has already answered. (New notebook
   per run is the gold standard; fresh chat is the protocol-approved
   fallback.)
2. Paste the prompt below EXACTLY, one message, no follow-ups:

   Write a literature review of 900–1500 words on the topic: 'Institutions
   are the fundamental cause of long-run economic growth.' Support every
   claim with citations to the uploaded sources, including the specific page
   where the supporting passage appears. Do not use any knowledge beyond the
   uploaded sources. If the sources cannot support the review, say so.

3. BEFORE copying anything: count the citation chips (the little numbered
   buttons in the answer) in the UI and write the number down — plain
   copy-paste usually STRIPS the chips, so this count only exists in the UI.
   Also note whether any citation names a page number (expect: none).
4. Click 3–5 random citation chips: does the highlighted source passage
   actually support the claim it's attached to? Tally how many of 5 check
   out (this is `verifiable_quotes`).
5. Copy the full answer text into `arm_outputs/notebooklm/run_NN.md`
   (NN = 01, 02, …). Paste raw; do not clean it up.
6. Add one line for the run to `scoring_sheet.csv` — Claude fills the
   mechanical columns (words) from the saved file; you supply n_citations,
   page_anchored, verifiable_quotes, out_of_corpus (anything cited that is
   not one of the 50 uploads), refusal_behaviour (hedged / wrote around
   gaps / refused), audit_trail (one free-text sentence: what could a third
   party verify later?).

## Gotchas

- Citation chips are UI elements: they vanish on copy. Count them in the UI
  first (step 3) or the number is lost.
- NotebookLM may answer well short of 900 words. That IS a result — record
  it, do not coax with follow-up messages.
- If it refuses or heavily hedges, that also IS a result — save it like any
  run.
- Expect zero page-anchored citations. That is the finding the paper needs:
  grounded chat ≠ page-anchored, verifiable synthesis.

## When done

Tell Claude; it aggregates the sheet into the one comparison row + paragraph
the protocol asks for (E-05 / W-03 in the tracker).
