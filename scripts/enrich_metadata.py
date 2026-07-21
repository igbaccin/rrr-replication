#!/usr/bin/env python3
"""
enrich_metadata.py -- Match corpus PDFs to BibTeX entries and write metadata.csv

Extracted from rrr_setup3.py for the replication package. This script takes a
folder of PDFs and a BibTeX file, matches each PDF filename to its bibliography
entry using author-year parsing and fuzzy surname matching, and writes an
enriched metadata.csv that the rest of the pipeline consumes.

Usage:
    python scripts/enrich_metadata.py --corpus corpus/ --bib bibliography.bib --output metadata.csv

If --bib is omitted, the script falls back to filename-only metadata (no title,
venue, or DOI enrichment).
"""

import argparse
import csv
import os
import re
import sys

def pjoin(*parts):
    return os.path.normpath(os.path.join(*parts))


# ---------------------------------------------------------------------------
# BibTeX loading
# ---------------------------------------------------------------------------

def load_bib(bib_path):
    """Load BibTeX file and build an author-year lookup dictionary."""
    try:
        from pybtex.database import parse_file
        from rapidfuzz import fuzz  # noqa: F401 (checked at match time)
        have_deps = True
    except ImportError:
        print("[WARN] pybtex or rapidfuzz not installed. Proceeding with filename-only metadata.")
        return None, False

    if not os.path.exists(bib_path):
        print(f"[WARN] BibTeX file not found at {bib_path}. Proceeding with filename-only metadata.")
        return None, have_deps

    bib_data = parse_file(str(bib_path))
    bib_by_author_year = {}

    for key, entry in bib_data.entries.items():
        year = entry.fields.get("year", "")
        authors = entry.persons.get("author", [])
        if authors and year:
            first_author = " ".join(authors[0].last_names) if authors[0].last_names else ""
            all_surnames = [" ".join(p.last_names) for p in authors]

            bib_entry_data = {
                "title": str(entry.fields.get("title", "")).strip("{} "),
                "authors": " & ".join(all_surnames),
                "year": year,
                "venue": entry.fields.get("journal", ""),
                "volume": entry.fields.get("volume", ""),
                "number": entry.fields.get("number", ""),
                "pages": entry.fields.get("pages", ""),
                "doi_or_url": entry.fields.get("doi", "") or entry.fields.get("url", ""),
                "all_surnames": all_surnames,
                "author_full": "; ".join(
                    [f"{' '.join(p.last_names)}, {' '.join(p.first_names)}" for p in authors]
                ),
            }

            # v16.11: store a LIST per (first_author, year) key. Two distinct
            # papers can share that key — North_1989 "Institutions..." vs
            # North&Weingast_1989 "Constitutions...", Frankema_2012 vs
            # Frankema&vanWaijenburg_2012. The old dict ASSIGNMENT overwrote on
            # collision, so both PDFs matched the surviving entry and one
            # paper's metadata was cloned onto the other. match_bib_entry()
            # now disambiguates the list by full-author-set overlap.
            lookup_key_spaced = (first_author.lower(), str(year))
            bib_by_author_year.setdefault(lookup_key_spaced, []).append(bib_entry_data)

            # Also index without spaces (for filenames like "deVries_1994"),
            # unless identical to the spaced key (avoid double-listing).
            lookup_key_nospace = (first_author.lower().replace(" ", ""), str(year))
            if lookup_key_nospace != lookup_key_spaced:
                bib_by_author_year.setdefault(lookup_key_nospace, []).append(bib_entry_data)

    return bib_by_author_year, have_deps


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename_to_author_year(filename):
    """
    Parse PDF filename into author, year, and author list.

    Examples:
        AcemogluEtAl_2001.pdf  -> ('Acemoglu', '2001', ['Acemoglu'])
        North&Weingast_1989.pdf -> ('North', '1989', ['North', 'Weingast'])
        Austin_2008.pdf         -> ('Austin', '2008', ['Austin'])
    """
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    if len(parts) < 2:
        return None, None, []

    author_part = parts[0]
    year_part = parts[-1]
    year_match = re.search(r'\d{4}', year_part)
    if not year_match:
        return None, None, []
    year = year_match.group()

    if "EtAl" in author_part or "etal" in author_part.lower():
        first_author = re.sub(r'(EtAl|etal).*$', '', author_part, flags=re.IGNORECASE)
        authors = [first_author]
    elif "&" in author_part:
        authors = [a.strip() for a in author_part.split("&")]
    else:
        authors = [author_part]

    first_author = authors[0] if authors else None
    return first_author, year, authors


def _surname_matches(fa, bib_surnames, have_deps):
    """True if filename surname `fa` corresponds to some bib surname."""
    if have_deps:
        from rapidfuzz import fuzz
        return any(fuzz.partial_ratio(fa, bs) > 80 for bs in bib_surnames)
    return any(fa == bs or fa in bs or bs in fa for bs in bib_surnames)


def _authors_overlap(entry, filename_authors, have_deps):
    """v14 rule: at least half the filename's authors appear in the entry."""
    bib_surnames = [s.lower() for s in entry["all_surnames"]]
    fn = [a.lower() for a in filename_authors]
    matches = sum(1 for fa in fn if _surname_matches(fa, bib_surnames, have_deps))
    return matches >= len(fn) * 0.5


def _author_match_score(entry, filename_authors, have_deps):
    """Rank a candidate entry against the filename's authors when several
    share a (first_author, year) key. Compared as a tuple, higher wins:
    (all filename authors present, author-count equal, number matched).
    The author-count term is the tiebreaker that sends North_1989 (1 author)
    to the 1-author entry and North&Weingast_1989 (2 authors) to the 2-author
    entry."""
    bib_surnames = [s.lower() for s in entry["all_surnames"]]
    fn = [a.lower() for a in filename_authors]
    matched = sum(1 for fa in fn if _surname_matches(fa, bib_surnames, have_deps))
    return (matched == len(fn), len(bib_surnames) == len(fn), matched)


def match_bib_entry(first_author, year, filename_authors, bib_lookup, have_deps):
    """Try to find a matching BibTeX entry for the given author/year."""
    if not bib_lookup:
        return None

    # Try lookup with spaces removed for compound surnames
    lookup_key = (first_author.lower().replace(" ", ""), str(year))
    candidates = bib_lookup.get(lookup_key)
    if not candidates:
        return None

    # v16.11: `candidates` is a list. Usually one entry — keep the v14
    # behaviour (accept, but reject a multi-author filename with no overlap).
    # Several entries means a shared (first_author, year) key: pick the one
    # whose full author set best matches the filename, so the collision
    # resolves to the RIGHT paper instead of cloning whichever was last.
    if len(candidates) == 1:
        entry = candidates[0]
        if len(filename_authors) > 1 and not _authors_overlap(entry, filename_authors, have_deps):
            return None
        return entry

    return max(candidates, key=lambda e: _author_match_score(e, filename_authors, have_deps))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Match corpus PDFs to BibTeX entries and write metadata.csv"
    )
    parser.add_argument(
        "--corpus", required=True,
        help="Path to the folder containing corpus PDFs"
    )
    parser.add_argument(
        "--bib", default=None,
        help="Path to the BibTeX file (default: bibliography.bib in corpus folder)"
    )
    parser.add_argument(
        "--output", default="metadata.csv",
        help="Path for the output metadata CSV (default: metadata.csv)"
    )
    args = parser.parse_args()

    corpus_dir = os.path.abspath(args.corpus)
    output_path = os.path.abspath(args.output)

    # Resolve BibTeX path
    if args.bib:
        bib_path = os.path.abspath(args.bib)
    else:
        # Try bibliography.bib in the corpus folder, then in the working
        # directory. (v16.11: dropped the legacy corpus-side "data_bib.bib"
        # name — bibliography.bib is the single source of truth.)
        for candidate in [
            os.path.join(corpus_dir, "bibliography.bib"),
            "bibliography.bib",
        ]:
            if os.path.exists(candidate):
                bib_path = os.path.abspath(candidate)
                break
        else:
            bib_path = None
            print("[WARN] No BibTeX file found. Using filename-only metadata.")

    # Check corpus folder
    if not os.path.isdir(corpus_dir):
        print(f"ERROR: Corpus folder not found: {corpus_dir}")
        sys.exit(1)

    pdfs = [f for f in sorted(os.listdir(corpus_dir)) if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"ERROR: No PDFs found in {corpus_dir}")
        sys.exit(1)

    print(f"[enrich] Corpus folder : {corpus_dir}")
    print(f"[enrich] PDFs found    : {len(pdfs)}")
    print(f"[enrich] BibTeX file   : {bib_path or '(none)'}")
    print(f"[enrich] Output        : {output_path}")
    print()

    # Load BibTeX
    bib_lookup, have_deps = (None, False)
    if bib_path:
        bib_lookup, have_deps = load_bib(bib_path)

    # Write metadata.csv
    matched = 0
    unmatched = 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "doc_id", "title", "authors", "author_full", "year",
            "venue", "volume", "number", "pages", "doi_or_url", "pdf_path"
        ])

        for fn in pdfs:
            doc_id = os.path.splitext(fn)[0].replace(" ", "_")
            first_author, year, filename_authors = parse_filename_to_author_year(fn)

            if first_author and year and bib_lookup:
                entry = match_bib_entry(first_author, year, filename_authors, bib_lookup, have_deps)
                if entry:
                    w.writerow([
                        doc_id,
                        entry["title"],
                        entry["authors"],
                        entry["author_full"],
                        entry["year"],
                        entry["venue"],
                        entry["volume"],
                        entry["number"],
                        entry["pages"],
                        entry["doi_or_url"],
                        pjoin(corpus_dir, fn),
                    ])
                    matched += 1
                    print(f"  [match]  {doc_id}")
                    continue

            unmatched += 1
            w.writerow([
                doc_id,
                f"[Title unknown - {doc_id}]",
                first_author or "[Unknown author]",
                "",
                year or "",
                "",
                "",
                "",
                "",
                "",
                pjoin(corpus_dir, fn),
            ])
            print(f"  [no match] {doc_id}")

    print()
    print(f"[enrich] {matched} PDFs matched to BibTeX entries")
    print(f"[enrich] {unmatched} PDFs using filename-only metadata")
    print(f"[enrich] Wrote {output_path}")


if __name__ == "__main__":
    main()
