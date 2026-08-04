# NotebookLM comparison record

Five independent NotebookLM runs are preserved as `run_01.md` through
`run_05.md`. Each notebook received the same 50 works and the following
request:

> Write a literature review of 900–1500 words on the topic: 'Institutions are
> the fundamental cause of long-run economic growth.' Support every claim with
> citations to the uploaded sources, including the specific page where the
> supporting passage appears. Do not use any knowledge beyond the uploaded
> sources. If the sources cannot support the review, say so.

Each run began in a fresh chat. The answer was copied without editing, and the
number of citation chips visible in the interface was recorded before export.
The scoring sheet records 176 interface citation chips and 178 page-looking
pairs written in the copied text. The common body-word counter gives a mean of
1,286.6 words, with a sample standard deviation of 175.13.

NotebookLM's copied output did not preserve the link between its interface
citation chips and the page-looking pairs written in the response. The common
E1 through E5 checker therefore could not evaluate these outputs on the same
basis as the page-bound conditions. A separate deterministic check evaluated
five page-attributed claims from one run and found support for three. The
claim-level record is in `verification_sample.md`.

The deposited files are:

- `run_01.md` through `run_05.md`, the copied answers;
- `scoring_sheet.csv`, the run-level counts;
- `verification_sample.md`, the five-claim page check.
