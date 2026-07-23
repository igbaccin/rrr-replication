import os
import json
import re
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from rrr.utils import ensure_dir, env_int
from rrr.paths import runs_path
from rrr.render import (
    CITE_RE,
    DISPLAY_CITE_RE,
    DISPLAY_PAREN_CITE_RE,
    parse_citations,
    render_citation,
    _build_author_year_lookup,
    _build_display_lookup,
    _collect_cited_docs as _shared_collect_cited_docs,
    _doc_id_to_author_label,
)
# v15.14: install the central ollama.chat shim here too (idempotent). The
# reasoner installs it at pipeline entry, but the writer can also be driven
# standalone (__main__ / compose_review from a script), in which case the
# thinking-mode suppression, API and host routing, and request timeout
# would silently not apply without this call.
from rrr.llm import install as _install_llm_shim
_install_llm_shim()
# The writer emits only [E####] evidence markers. The renderer below resolves
# those markers directly to the final display citation.

# v11.2 lever 2: per-stage model selection. The writer needs prose quality;
# the reasoner needs JSON-schema obedience. Splitting lets the writer stay on
# a prose-tier model while the reasoner runs on a smaller, cheaper model.
# Falls back to RRR_MODEL if unset (preserves v11.1 behaviour).
_MODEL = os.environ.get("RRR_WRITER_MODEL", os.environ.get("RRR_MODEL", "mistral-small:24b"))
_KEEP_ALIVE = "30m"

# v13: RRR_WRITER_T/CTX/PRED/TOPP/TAIL_CHARS retired (lever pruning). Per-stage
# tuning was never overridden in the wild; v7-v8 values are frozen below.
# v7 raised temperature from 0.30 to 0.50; v8 R12 dropped num_ctx from 32768
# to 12288 (largest observed writer prompt is ~2900 tokens); v8 dropped
# num_predict to 1500 (largest observed output ~620 tokens, 2.4x slack).
_DEFAULT_CHAT_OPTIONS = {
    "temperature": 0.50,
    "num_ctx": 12288,
    "num_predict": 1500,
    "top_p": 0.9,
}

# v15.4.0 (Bug 4 half-fix): bump the previous-chunk tail from 250 to 600
# chars so the next writer call sees enough prior prose to recognise an
# author who was just introduced. The cross-paragraph "Ogilvie introduced
# twice" echo in the v15.2.3 INST review traced to this slice being too
# narrow to include the previous paragraph's citation surnames.
_TAIL_CHARS = 600

# v13: _PAGE_ONLY_RE / _AUTHOR_NAME_RE / _DOC_WITHOUT_PAGE_RE / _AUTHOR_YEAR_*
# / _MULTIPAGE_CITE_RE removed. These regexes fed only the loose-pattern arm
# of _remove_invalid_citations, which is now retired per architecture item
# A-030 (evidence-id rendering + display_lookup gating supersede the
# loose-pattern scrub). The v12 smoke had writer_removed_citations=12 — those
# came from the loose arm catching display-form variants outside the
# canonical-pair gating, but with the v13 display/canonical pipeline the model
# only emits forms reachable via the lookup, so the loose arm is dead.
_GENERIC_STYLE_RE = re.compile(
    r"\b("
    r"complex interplay|valuable insights|policy-making|future research|further research|"
    r"further investigation|nuanced perspective|the stakes are high|this analysis will|"
    r"delving deeper|underscores? the need|further exploration|ongoing research|"
    r"shed light|complex and influenced by various factors|"
    # v8: NGO/consulting register that leaks through current directives
    r"holds significance|potential implications|strategies aimed at|"
    r"sustainable development|alleviating poverty|this investigation|"
    r"this analysis|this study|this examination|plays a pivotal role|"
    r"contributing significantly to|fundamental factor|crucial factor|"
    r"essential factor|interplay between|in this regard|in this context|"
    r"by acknowledging these variables|presents a complex picture|"
    r"thereby contributing|"
    # v13: near-paraphrases confirmed by the post-v12 prose audit. Six surfaces
    # survived v12: "complex interactions between" (paraphrase of interplay),
    # "played/playing a significant role" and "plays a significant role" (the
    # pivotal-role family with 'significant' substituted), "significantly
    # influenced" (intensifier + impact-verb pattern), "a more nuanced
    # understanding" (review wishlist), "It is crucial to examine" (paired
    # filler intent), and "are pivotal in" (predicate-adjective form of plays a
    # pivotal role).
    r"complex interactions between|"
    r"play(?:s|ed)?\s+a\s+significant\s+role|"
    r"significantly\s+influenc(?:e|ed|es)|"
    r"a\s+more\s+nuanced\s+understanding|"
    r"it\s+is\s+crucial\s+to\s+examine|"
    r"are\s+pivotal\s+in|"
    # v13.1 FIX-C: widen the pivotal family. The v13 smoke had openers like
    # "stands as a pivotal question" (topic 1) and "is pivotal for understanding"
    # (topic 4) that "are pivotal in" did not catch. Two alternation arms:
    # (1) "pivotal <noun>" surfaces (question/role/issue/factor/moment/area/
    # importance/insight), (2) modal + optional article + "pivotal" predicate
    # forms ("is/are/stands as/remains/becomes a pivotal..."). Routed to the
    # existing batched LLM style-rewriter, same as the other entries.
    r"pivotal\s+(?:question|role|issue|factor|moment|area|importance|insight)|"
    r"(?:is|are|stands\s+as|remains|becomes)\s+(?:a\s+|the\s+)?pivotal"
    r")\b",
    re.IGNORECASE,
)

# v8 (R5) / v9.1: catch nested citation wraps that escape every other cleanup
# because CITE_RE matches only the single-paren form. Two forms observed:
#   1. ((Doc: p.N))              — simple double-wrap (v8)
#   2. ((Doc: p.X): p.Y)         — doubled page-suffix; happens when the LLM
#      writes ([E0001]: p.X) and _render_evidence_id_citations turns [E0001]
#      into (Doc: p.X), producing the nested form. v9 introduced two of these
#      because the fused-call writer prompt provides richer evidence context
#      that the model wraps in parens. (v9.1 fix.)
_DOUBLE_PAREN_CITE_RE = re.compile(
    r"\(\(([A-Za-z0-9_&.\-]+:\s*p\.\d+)\)\)"
)
_NESTED_PAGE_SUFFIX_RE = re.compile(
    r"\(\(([A-Za-z0-9_&.\-]+):\s*p\.\d+\):\s*p\.\d+\)"
)
# v15.7: display-form double-paren patterns the audit found 50 times in the
# v15.6 smoke shipped to users. Display surfaces:
#   ((Author Year, p.N))          — single paren-display cite wrapped
#   ((Author Year, p.N); (Author2 Year2, p.M)) — group of paren-display cites
#   ((Author Year, p.N) (Author2 Year2, p.M))  — same, space-joined
_DOUBLE_PAREN_DISPLAY_RE = re.compile(
    r"\(\(((?:(?i:van|von|de|del|der)\s+[A-Z][A-Za-z\-]+|[A-Z][A-Za-z\-]+)"
    r"(?:\s+(?:and\s+(?:(?i:van|von|de|del|der)\s+[A-Z][A-Za-z\-]+|[A-Z][A-Za-z\-]+)"
    r"|et\s+al\.))?"
    r"\s+\d{4}[a-z]?(?:,\s*|\s+)p\.\s*\d+)\)\)"
)
# Outer paren wrapping a group of inner display-paren cites separated by
# whitespace or semicolons. Conservative: inner must be two or more cite
# units wrapped together with no other prose in the outer paren.
_DOUBLE_PAREN_DISPLAY_GROUP_RE = re.compile(
    r"\(((?:\([^()]+?\)[\s;,]*){2,})\)"
)
# v15.7.1: asymmetric outer-paren residue — the model writes '((cite).' (open
# wrap, single close, sentence terminator after). The symmetric collapser
# above misses these because it requires '))'. The v15.7 smoke showed two of
# these shipped in GD 04 ('((Broadberry and Gupta 2006, p.17).',
# '((North 1989, p.6) (Ogilvie 2007, p.2).'). Match '((' + non-paren content
# + ')' NOT followed by ')' — strip the leading '(' (capture the content +
# close as one group). When a SECOND cite follows ('((cite1) (cite2).'), the
# outer-wrap was over the FIRST cite only; the regex catches the leading
# half and the second cite stays as-is.
_ASYM_OUTER_PAREN_RE = re.compile(
    r"\(\(([^()]+\))(?!\))"
)

# v10: detect-then-LLM-rewrite style enforcement. We deliberately do NOT
# substitute forbidden words mechanically because the right replacement is
# sense-dependent (e.g. "sharp distinction" vs "sharply declined"). Sentences
# containing any of these patterns are routed to a single batched LLM rewrite
# call per writer section.
_FORBIDDEN_LEXEME_RE = re.compile(r"\b(?:legible|sharp(?:ly)?)\b", re.IGNORECASE)
_ADVERSATIVE_RE = re.compile(
    r"(?:\b("
    # v13: widen 'not X but Y' to span 0-15 tokens between not and but. The v12
    # smoke had 'not merely the identification of factors that drive economic
    # progress, but also' which is 11 tokens — the v10 6-token cap missed it.
    r"not\s+(?:\S+\s+){0,15}but(?:\s+(?:rather|also))?\s+"
    r"|rather than\b"
    r"|in (?:sharp\s+)?contrast(?:\s+to)?\b"
    r"|instead of\b"
    r"|unlike\b"
    # v13: sentence-initial Conversely/Yet/However acting as adversative pivots.
    # The (?<=^|[.!?]\s)|(?<=\n) lookbehind anchors to a sentence boundary so we
    # don't match mid-sentence 'however' (which is rare anyway). Comma after the
    # pivot is the model's tell — these always sit at sentence start with comma.
    r")\b)|(?:(?<=[.!?]\s)(?:Conversely|Yet|However),)|(?:^(?:Conversely|Yet|However),)",
    re.IGNORECASE | re.MULTILINE,
)
# v10 (rule 6): trailing significance clauses — these ARE safely strippable
# because the head of the sentence carries the substantive claim.
_TRAILING_SIGNIFICANCE_RE = re.compile(
    r"[,.;]?\s+"
    r"(?:[Aa]nd that matters"
    r"|[Ww]hich matters because[^.!?]*"
    r"|[Tt]his is important because[^.!?]*"
    r"|[Tt]his matters because[^.!?]*)"
    r"[.!?]?",
)
# v10 (rule 8): colon-heavy academic setup ("A long stem clause: A new clause").
_COLON_SETUP_RE = re.compile(r"[A-Z][^.!?:\n]{20,}:\s+[A-Z]")
# v10 (rule 1): em dash + en dash. Both go through the LLM rewriter because the
# right replacement varies (comma, period, parens, semicolon).
_TYPOGRAPHIC_DASH_RE = re.compile("[–—]")
# v10 options for the short style-rewrite LLM call.
# v13: RRR_STYLE_CTX/PRED retired; tuning frozen.
_OPTIONS_STYLE_REWRITE = {"temperature": 0.1, "num_ctx": 4096, "num_predict": 1500}

# v8 (R5): detect "Author argues / posits / emphasizes ..." openings that the
# system prompt forbids but the model keeps producing. Used to flag, then
# optionally rewrite, sections that violate.
_AUTHOR_VERB_RE = re.compile(
    r"\b((?:[A-Z][A-Za-z&.\-]+|(?:van|von|de|del|der)\s*[A-Z][A-Za-z&.\-]+)(?:\s+et\s+al\.?)?(?:\s*\(?[A-Za-z_]*\d{4}[A-Za-z_]*(?:,\s*p\.\d+)?\)?)?)\s+"
    r"(argues|emphasizes|emphasises|demonstrates|highlights|posits|suggests|"
    r"claims|notes|observes|maintains|asserts|contends|shows|writes|states|"
    r"finds|associates|associate|situates|situate|extends|extend|points out|"
    r"underscores|argues that|notes that)\b",
    re.IGNORECASE,
)
# v15.14: env_int — a malformed value here used to kill the module IMPORT.
_MIN_SECTION_CITED_DOCS = env_int("RRR_WRITER_MIN_SECTION_CITED_DOCS", 2)
# A closing shorter than this has lost too much of its generated synthesis to
# serve as the review's conclusion. The generation prompt asks for 140-180
# words; 120 leaves room for tokenization variation while still rejecting the
# 10-40 word stubs observed in the v15.17 battery.
_MIN_CLOSING_WORDS = env_int("RRR_WRITER_MIN_CLOSING_WORDS", 120)
# The prompt identifies 140-180 words as the target and states the complete
# 120-200 hard range. This tolerance limits needless retries while rejecting
# the citation-heavy 220+ word closings found in the first v19 smoke pair.
_MAX_CLOSING_WORDS = 200
# A closing should cite each selected document-page identity once. This keeps
# the synthesis compact and prevents every sentence from repeating the full
# source packet.
_MAX_CLOSING_IDENTITY_REPETITIONS = 1
# v19: one initial closing call plus, when a deterministic gate fails, one
# fault-routed clean-room retry. Keeping the cap explicit makes call
# amplification observable and prevents a rejected closing from turning into
# an open-ended generation loop.
_MAX_CLOSING_MODEL_CALLS = 2
# A closing needs enough source range to synthesize across streams while
# keeping its call packet materially smaller than an ordinary section packet.
_MAX_CLOSING_DOCS = 4
# This marker travels through final assembly with the generated closing. It is
# removed before the review is written. Keeping an explicit identity prevents
# a surviving stream tail from being mistaken for the closing when cleanup
# removes closing material.
_CLOSING_START_MARKER = "[[RRR-CLOSING-START]]."
# v13: RRR_WRITER_ENFORCE_COVERAGE retired (always on). Section-level coverage
# enforcement is part of the corpus-grounding contract; disabling it is no
# longer a supported configuration.
_ENFORCE_COVERAGE = True

# v19 records the scholarly output contract separately from the execution
# profile. Host and local runs share the same acceptance rules. The local
# profile retains compact continuity context and structured Qwen transport;
# the host profile sends the claims-first brief without the raw prior tail.
# Accepted sections are rendered and validated before final assembly, where
# semantic mutation is prohibited.
_WRITER_ARTIFACT_CONTRACT = "rrr-writer-v19"
_WRITER_LOCAL_EXECUTION_PROFILE = "local-assisted-v19"
_WRITER_HOST_EXECUTION_PROFILE = "host-lean-v19"
_WRITER_API_EXECUTION_PROFILE = "api-bounded-v19"
_CITATION_REPRESENTATION_VERSION = "evidence-id-display-v19"
_POST_ACCEPTANCE_MUTATION_POLICY = "frozen-sections-v1"


def _writer_execution_profile() -> str:
    """Return the auditable execution profile selected by the runtime."""
    explicit = os.environ.get("RRR_WRITER_EXECUTION_PROFILE", "").strip()
    if explicit:
        supported = {
            _WRITER_LOCAL_EXECUTION_PROFILE,
            _WRITER_HOST_EXECUTION_PROFILE,
            _WRITER_API_EXECUTION_PROFILE,
        }
        if explicit not in supported:
            raise ValueError(
                "RRR_WRITER_EXECUTION_PROFILE must be one of "
                + ", ".join(sorted(supported))
            )
        return explicit

    runtime = os.environ.get("RRR_RUNTIME", "").strip().lower()
    if runtime == "host":
        return _WRITER_HOST_EXECUTION_PROFILE
    if runtime == "api":
        return _WRITER_API_EXECUTION_PROFILE
    return _WRITER_LOCAL_EXECUTION_PROFILE


def _writer_uses_raw_previous_tail() -> bool:
    """Keep raw continuity prose within the assisted local profile only."""
    return _writer_execution_profile() == _WRITER_LOCAL_EXECUTION_PROFILE


def _writer_system_prompt() -> str:
    """v15.12: the writer is the SINGLE output-language boundary. Internal
    reasoning + evidence are in the corpus/pivot language; the writer reads
    that material and produces the entire review in the topic language.

    When topic_lang == corpus_lang (the common case, both English) the
    prefix is empty and the system prompt is byte-identical to pre-v15.11.
    When they differ, an explicit, forceful contract makes the model write
    100% in the topic language while quoting/citing sources that are in the
    corpus language — this fixes the v15.11 mixed FR/EN prose.
    """
    from rrr.language import language_name
    topic_lang = os.environ.get("RRR_TOPIC_LANG", "en")
    corpus_lang = os.environ.get("RRR_CORPUS_LANG", "en")
    if topic_lang == corpus_lang:
        return _SYSTEM_CITATION_INSTRUCTION
    tl = language_name(topic_lang)
    cl = language_name(corpus_lang)
    contract = (
        f"OUTPUT LANGUAGE — READ FIRST. The source evidence, cluster "
        f"syntheses, and section notes below are in {cl}. Write the review "
        f"in {tl}, with this rule:\n"
        f"  - YOUR OWN WORDS — all narration, analysis, synthesis, topic "
        f"sentences, transitions, everything that is not a quotation — MUST "
        f"be in {tl}. Do NOT copy {cl} sentences from the evidence into your "
        f"prose as if they were your narration; read the {cl} evidence, "
        f"understand it, and express the point in {tl}.\n"
        f"  - VERBATIM QUOTATIONS may stay in {cl}. A verbatim quote is "
        f"scholarly evidence and keeps its ORIGINAL language — do NOT "
        f"translate a quote. But it MUST be a genuine short excerpt "
        f"(<=12 words) inside quotation marks (\" \") followed by its "
        f"[E####] citation. Anything in {cl} that is NOT inside quotation "
        f"marks is a mistake — rewrite it in {tl}.\n"
        f"  - Author surnames and years inside citation markers stay as-is.\n"
        f"Net: a reader sees {tl} prose throughout, with occasional clearly-"
        f"marked {cl} quotations. No unquoted {cl} sentences anywhere."
    )
    return f"{contract}\n\n{_SYSTEM_CITATION_INSTRUCTION}"


# v10: cite as 'Author (Year, p.N)'. Underscore-keyed canonical form retired.
_SYSTEM_CITATION_INSTRUCTION = (
    # v14.2.1 simplification: reorganised into BOUNDARY (architecturally
    # enforced) and FORMAT (style) buckets so the model sees the hard rules
    # first. Citation-format material moved here from _PROSE_DIRECTIVE
    # (evidence-ID syntax + multi-source preference) since both belong with
    # the format example, not with style guidance.
    # v15.5: ONE citation surface — the evidence-ID marker [E####]. The
    # renderer produces the appropriate narrative or parenthetical form
    # depending on whether the author is already named in the prose, so
    # the writer never decides citation surface. This eliminates the
    # whole class of citation-format bugs the older pipeline patched
    # (placeholder leaks, orphan parens, redundant inline+trailing,
    # nested parens, etc.).
    "CITE only via evidence IDs: every citation is one [E####] marker copied "
    "from the ALLOWED CITATIONS list. Prefer claim-led prose followed by its "
    "allowed marker. When an author is named in prose, put that paper's "
    "allowed marker immediately after the author name and before the verb. "
    "For a multi-source claim, place the allowed markers next to one another "
    "with spaces only. DO NOT write '(Author Year, p.N)', "
    "'Author (Year, p.N)', '(p.5)' or any other citation surface — the "
    "renderer converts every [E####] to the right surface for you.\n\n"
    "BOUNDARY RULES:\n"
    "1. Use only [E####] IDs from the ALLOWED CITATIONS list.\n"
    "   Put each marker before the sentence's terminal punctuation. Never "
    "place a marker after a completed sentence or at a paragraph opening.\n"
    "2. If a claim is not supported by allowed evidence, state it WITHOUT "
    "a citation. Do not invent evidence IDs.\n"
    "3. QUOTED TEXT: paraphrase by default. Reserve quotation marks (\" \") "
    "for SHORT phrases (≤12 words) you copy-paste verbatim from the "
    "Evidence section. Multi-sentence quotes, ellipsis-truncated quotes, "
    "and paraphrased-but-quoted material are fabrications and get "
    "rejected with the section.\n"
    "4. Finish every paragraph and every section with a complete, "
    "standalone sentence. Section transitions are thematic. Never leave a "
    "clause or sentence for the next section to complete.\n"
    "5. Return only the requested review prose. Begin with its first "
    "sentence. Do not include planning, analysis, a checklist, headings, "
    "notes, a word count, draft labels, or commentary.\n"
)

_OUTPUT_ONLY_DIRECTIVE = (
    "Return only the requested review prose. Begin immediately with its "
    "first sentence. Do not include planning, analysis, a checklist, "
    "headings, notes, a word count, draft labels, or commentary."
)

_QWEN_PROSE_SCHEMA = {
    "type": "object",
    "properties": {"prose": {"type": "string"}},
    "required": ["prose"],
    "additionalProperties": False,
}
_QWEN_PROSE_TRANSPORT_INSTRUCTION = (
    "For transport, return exactly one JSON object with one key named "
    "prose. Put only the requested review prose in that string."
)


def _uses_qwen_prose_transport(model: str) -> bool:
    runtime = os.environ.get("RRR_RUNTIME", "").strip().lower()
    return "qwen3" in str(model or "").lower() and runtime in {
        "", "local", "ollama",
    }


def _decode_qwen_prose_transport(raw: str) -> str:
    try:
        payload = json.loads(raw or "")
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Qwen prose transport returned invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"prose"}:
        raise ValueError("Qwen prose transport must contain only 'prose'")
    prose = payload.get("prose")
    if not isinstance(prose, str) or not prose.strip():
        raise ValueError("Qwen prose transport returned empty prose")
    return prose.strip()


def _score_doc(d) -> float:
    return d.get("avg_score", 0)


def _clip(s: str, n=260) -> str:
    s = (s or "").strip().replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return (s[:n] + "...") if len(s) > n else s


def _writer_passage_text(q) -> str:
    """Return the complete normalized passage supplied to the writer."""
    return re.sub(r"\s+", " ", str(q.get("text", "") or "")).strip()


def _format_quote(q) -> str:
    # v15.7: removed trailing render_citation. The display surface
    # 'Author (Year, p.N)' shown here was the model's training set for
    # copy-pasting the forbidden surface into its prose. Provenance is
    # carried by the [E####] marker plus the descriptive line ('from paper
    # X, page Y') so the writer can attribute correctly without ever
    # seeing the surface it is forbidden to emit.
    did = str(q.get("doc_id", "")).strip()
    pg = int(q.get("page", 0) or 0)
    tx = _writer_passage_text(q)
    eid = str(q.get("evidence_id", "")).strip()
    prefix = f"[{eid}] " if eid else ""
    suffix = f" (from {did}, p.{pg})" if did and pg else ""
    return f'{prefix}"{tx}"{suffix}'


def _writer_quotes_per_doc() -> int:
    """Return the retained-passage limit applied to one writer call."""
    return max(1, env_int("RRR_WRITER_QUOTES_PER_DOC", 2))


def _select_call_evidence(docs, quotes_per_doc=None):
    """Build the authoritative evidence packet for one writer call.

    The ledger remains unchanged. Each returned document is a shallow copy
    whose ``quotes`` list contains only the passages supplied to this call.
    """
    limit = max(1, int(quotes_per_doc or _writer_quotes_per_doc()))
    packet = []
    for source_doc in docs or []:
        d = dict(source_doc)
        selected = list(source_doc.get("quotes") or [])[:limit]
        if not selected:
            raise ValueError(
                f"writer call document {source_doc.get('doc_id')!r} has no evidence passages"
            )
        for q in selected:
            eid = str(q.get("evidence_id", "")).strip()
            did = str(q.get("doc_id", source_doc.get("doc_id", ""))).strip()
            page = int(q.get("page", 0) or 0)
            if not eid or not did or page <= 0:
                raise ValueError(
                    f"writer call document {source_doc.get('doc_id')!r} contains "
                    "a selected passage without evidence_id, doc_id, or page"
                )
        d["quotes"] = selected
        packet.append(d)
    return packet


def _format_doc_entry(d) -> str:
    """Format every passage in an already bounded writer-call packet."""
    did = str(d.get("doc_id", "")).strip()
    author_label = _doc_id_to_author_label(did)
    lines = [f"[{did}] cite as {author_label!s}"]
    claim = str(d.get("claim", "") or "").strip()
    if claim:
        lines.append(f"  Paper's central claim: {_clip(claim, n=240)}")
    qs = d.get("quotes") or []
    for q in qs:
        lines.append(f"  {_format_quote(q)}")
    return "\n".join(lines)


def _list_allowed_citations(docs, allowed_pages_by_doc) -> str:
    # v15.7: emit DESCRIPTIVE provenance only — never the rendered display
    # surface. Each line maps [E####] to (paper doc_id, page) plus the
    # author surname as plain text. The writer's system prompt forbids
    # display form '(Author Year, p.N)' yet the v15.5/v15.6 list shown
    # the surface 32+ times per prompt, which is the source of the 16
    # display-form leaks the v15.6 audit found. Closing the source.
    lines = []
    evidence_lines = []
    for d in docs:
        did = str(d.get("doc_id", "")).strip()
        for q in d.get("quotes", []) or []:
            eid = str(q.get("evidence_id", "")).strip()
            page = int(q.get("page", 0) or 0)
            if eid and did and page:
                surnames = _author_surnames_only(_doc_id_to_author_label(did))
                evidence_lines.append(
                    f"  - [{eid}] (paper {did}, page {page}; author surname: {surnames})"
                )
        if evidence_lines:
            continue
        pages = sorted(list(allowed_pages_by_doc.get(did, set())))
        if pages:
            surnames = _author_surnames_only(_doc_id_to_author_label(did))
            page_str = ", ".join(f"p.{p}" for p in pages[:6])
            lines.append(f"  - (paper {did}, pages {page_str}; author surname: {surnames})")
    if evidence_lines:
        return "\n".join(evidence_lines)
    return "\n".join(lines)


def _build_evidence_id_map(docs):
    evidence = {}
    for d in docs:
        for q in d.get("quotes", []) or []:
            eid = str(q.get("evidence_id", "")).strip()
            did = str(q.get("doc_id", "")).strip()
            page = int(q.get("page", 0) or 0)
            if eid and did and page:
                evidence[eid] = {"doc_id": did, "page": page}
    return evidence


def _repair_allowed_author_year_mentions(
    text: str,
    docs,
    *,
    stage: str = "",
) -> tuple:
    """Prepare a provenance-recorded rescue for exact call-packet mentions.

    The function only markerizes literal author-year surfaces that identify a
    unique document in the current writer call. Each repair binds the first
    selected passage for that document, matching the call-packet order shown to
    the model. The caller must render, re-audit, and accept the candidate as one
    transaction.
    """
    source = text or ""
    evidence_map = _build_evidence_id_map(docs)
    surface_entries = defaultdict(list)

    for doc in docs or []:
        doc_id = str(doc.get("doc_id", "") or "").strip()
        label = _doc_id_to_author_label(doc_id)
        label_match = re.fullmatch(r"(.+?)\s*\((\d{4}[a-z]?)\)", label or "")
        if not doc_id or not label_match:
            continue
        selected = None
        for quote in doc.get("quotes", []) or []:
            eid = str(quote.get("evidence_id", "") or "").strip()
            quote_doc = str(quote.get("doc_id", "") or "").strip()
            page = int(quote.get("page", 0) or 0)
            mapped = evidence_map.get(eid)
            if (
                eid and quote_doc == doc_id and page > 0 and mapped
                and mapped.get("doc_id") == doc_id
                and int(mapped.get("page", 0) or 0) == page
            ):
                selected = {"evidence_id": eid, "page": page}
                break
        if not selected:
            continue
        author = label_match.group(1).strip()
        year = label_match.group(2)
        common = {
            "doc_id": doc_id,
            "evidence_id": selected["evidence_id"],
            "page": selected["page"],
        }
        surface_entries[("narrative", label)].append(common)
        surface_entries[("parenthetical", f"({author} {year})")].append(common)

    unique_surfaces = {
        key: entries[0]
        for key, entries in surface_entries.items()
        if len({entry["doc_id"] for entry in entries}) == 1
    }
    collision_skips = sum(
        1
        for entries in surface_entries.values()
        if len({entry["doc_id"] for entry in entries}) > 1
    )

    quoted_spans = [
        match.span()
        for pattern in (
            r'"[^"\n]*"',
            r"\u201c[^\u201d\n]*\u201d",
            r"'[^'\n]*'",
            r"\u2018[^\u2019\n]*\u2019",
            r"(?m)^\s*>[^\n]*$",
        )
        for match in re.finditer(pattern, source)
    ]

    def is_quoted(start: int, end: int) -> bool:
        return any(start < quote_end and quote_start < end
                   for quote_start, quote_end in quoted_spans)

    proposed = []
    occupied = []
    for (surface_kind, surface), entry in sorted(
        unique_surfaces.items(), key=lambda item: len(item[0][1]), reverse=True,
    ):
        pattern = re.compile(
            rf"(?<!\w){re.escape(surface)}(?!\w)"
            r"(?![\s,;:]*\[[Ee]\d{1,5}\])"
        )
        for match in pattern.finditer(source):
            if is_quoted(match.start(), match.end()):
                continue
            if any(match.start() < end and start < match.end()
                   for start, end in occupied):
                continue
            replacement = (
                f"{surface.rsplit('(', 1)[0].rstrip()} "
                f"[{entry['evidence_id']}]"
                if surface_kind == "narrative"
                else f"[{entry['evidence_id']}]"
            )
            proposed.append({
                "start": match.start(),
                "end": match.end(),
                "replacement": replacement,
                "stage": stage,
                "phase": "after_retry",
                "surface_kind": surface_kind,
                "matched_text": match.group(0),
                "doc_id": entry["doc_id"],
                "evidence_id": entry["evidence_id"],
                "page": entry["page"],
                "selection_policy": "first_call_packet_passage",
            })
            occupied.append(match.span())

    repaired = source
    for record in sorted(proposed, key=lambda item: item["start"], reverse=True):
        repaired = (
            repaired[:record["start"]]
            + record["replacement"]
            + repaired[record["end"]:]
        )
    public_records = [
        {key: value for key, value in record.items()
         if key not in {"start", "end", "replacement"}}
        for record in sorted(proposed, key=lambda item: item["start"])
    ]
    return repaired, public_records, collision_skips


def _group_evidence_ids_by_doc(docs):
    """Group only the evidence IDs present in one bounded call packet."""
    grouped = defaultdict(list)
    for eid, ev in _build_evidence_id_map(docs).items():
        did = str(ev.get("doc_id", "") or "").strip()
        if did:
            grouped[did].append(eid)
    return {
        did: sorted(eids)
        for did, eids in grouped.items()
    }


def _build_call_contract(stage: str, docs, allowed_list: str, prompt: str) -> dict:
    """Describe and verify the evidence interface for one generation call."""
    system_prompt = _writer_system_prompt()
    evidence_map = _build_evidence_id_map(docs)
    allowed_pairs, allowed_docs, _ = _build_allowed_citations(docs)
    displayed_ids = set(evidence_map)
    listed_ids = set(re.findall(r"\[(E\d{4})\]", allowed_list or ""))
    prompt_ids = set(re.findall(r"\[(E\d{4})\]", prompt or ""))
    system_prompt_ids = set(
        re.findall(r"\[(E\d{4})\]", system_prompt or "")
    )

    if displayed_ids != listed_ids:
        raise AssertionError(
            f"{stage}: displayed evidence IDs differ from the allowed citation list"
        )
    if not prompt_ids.issubset(displayed_ids):
        extra = sorted(prompt_ids - displayed_ids)
        raise AssertionError(
            f"{stage}: prompt contains evidence IDs outside the call packet: {extra}"
        )
    if not system_prompt_ids.issubset(displayed_ids):
        extra = sorted(system_prompt_ids - displayed_ids)
        raise AssertionError(
            f"{stage}: system prompt contains evidence IDs outside the call "
            f"packet: {extra}"
        )

    passages = []
    for d in docs:
        source_did = str(d.get("doc_id", "")).strip()
        for q in d.get("quotes", []) or []:
            eid = str(q.get("evidence_id", "")).strip()
            did = str(q.get("doc_id", source_did)).strip() or source_did
            page = int(q.get("page", 0) or 0)
            source_text = str(q.get("text", "") or "")
            display_text = _writer_passage_text(q)
            passages.append({
                "evidence_id": eid,
                "doc_id": did,
                "page": page,
                "display_text": display_text,
                "display_text_sha256": hashlib.sha256(
                    display_text.encode("utf-8")
                ).hexdigest(),
                "source_text_sha256": hashlib.sha256(
                    source_text.encode("utf-8")
                ).hexdigest(),
            })

    displayed_texts_present = all(
        passage["display_text"] in (prompt or "") for passage in passages
    )
    if not displayed_texts_present:
        raise AssertionError(
            f"{stage}: a selected passage is absent or truncated in the prompt"
        )

    return {
        "stage": stage,
        "artifact_contract": _WRITER_ARTIFACT_CONTRACT,
        "execution_profile": _writer_execution_profile(),
        "citation_representation": _CITATION_REPRESENTATION_VERSION,
        "post_acceptance_mutation_policy": _POST_ACCEPTANCE_MUTATION_POLICY,
        "document_ids": sorted(allowed_docs),
        "evidence_ids": sorted(displayed_ids),
        "allowed_doc_page_pairs": [
            {"doc_id": did, "page": page}
            for did, page in sorted(allowed_pairs)
        ],
        "prompt_sha256": hashlib.sha256((prompt or "").encode("utf-8")).hexdigest(),
        "prompt_chars": len(prompt or ""),
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "system_prompt_chars": len(system_prompt),
        "full_request_sha256": hashlib.sha256(
            (system_prompt + "\n\n" + (prompt or "")).encode("utf-8")
        ).hexdigest(),
        "passages": passages,
        "invariants": {
            "displayed_ids_equal_listed_ids": displayed_ids == listed_ids,
            "displayed_passage_texts_present": displayed_texts_present,
            "prompt_ids_within_call_packet": prompt_ids.issubset(displayed_ids),
            "system_prompt_ids_within_call_packet": (
                system_prompt_ids.issubset(displayed_ids)
            ),
            "renderer_ids_equal_displayed_ids": set(evidence_map) == displayed_ids,
            "validator_pairs_equal_displayed_pairs": set(allowed_pairs) == {
                (v["doc_id"], int(v["page"])) for v in evidence_map.values()
            },
        },
    }


def _author_surnames_only(label: str) -> str:
    """Strip the trailing '(Year)' from an author label so we get just the
    surname(s) — 'Acemoglu et al. (2001)' -> 'Acemoglu et al.'."""
    if not label:
        return ""
    return re.sub(r"\s*\(\d{4}[a-z]?\)\s*$", "", label).strip()


# v15.5: how many chars to look BACK from an [E####] marker to decide
# whether the author surname is already named in the preceding prose.
# Keeps the lookback short so we don't false-positive across sentence
# boundaries.
_EID_LOOKBACK_CHARS = 80


def _render_evidence_id_citations(text: str, evidence_map: dict) -> tuple:
    """v15.7: context-aware evidence-ID renderer + attribution gate.

    For each [E####] marker:
      - if the author label is IMMEDIATELY before the marker (the
        immediate prior text ends with the surname label), render as
        just '(Year, p.N)' so the result forms 'Author (Year, p.N)' —
        which DISPLAY_CITE_RE recognises as a citation.
      - otherwise render as the parenthetical '(Author Year, p.N)' —
        which DISPLAY_PAREN_CITE_RE recognises.

    v15.7 (attribution gate): if the lookback finds a DIFFERENT surname
    immediately before the marker than the doc_id's expected surnames,
    REFUSE the bare-year form (which would attach the wrong author to
    the cite). Emit '(CorrectAuthor Year, p.N)' instead so the prose's
    wrong surname is at least not reinforced by a year-only cite the
    reader will attribute to that surname. Record the snippet in
    stats['attribution_mismatches'] so the caller can decide whether to
    trigger a coverage retry.

    v15.5 used a substring-anywhere-in-last-60-chars check, which fired
    on 'Mokyr argues that institutions follow from technology [E0001]'
    and rendered the marker as a bare '(1989, p.5)' embedded in lowercase
    prose. v15.6 tightened to immediate-adjacency. v15.7 adds the
    attribution check at the same lookback so the renderer becomes the
    single integrity gate.

    Returns (text, stats) where stats is a dict with keys:
        replacements: int (count of [E####] markers expanded)
        unknown_eids: int (count of markers whose eid was not in evidence_map)
        unknown_eid_snippets: list[str] (up to 10 ~120-char snippets)
        attribution_mismatches: int (count of prose-surname/eid-author mismatches)
        attribution_mismatch_snippets: list[dict] with keys
            {snippet, prose_surname, expected_surnames, eid, doc_id, page}
    """
    stats = {
        "replacements": 0,
        "unknown_eids": 0,
        "unknown_eid_snippets": [],
        "attribution_mismatches": 0,
        "attribution_mismatch_snippets": [],
    }
    if not text:
        return text or "", stats
    # v15.7.2: widened to 1-5 digits and normalised to 4-digit zero-padded
    # form before lookup. The v15.7 smoke shipped '[E009]' (3-digit — model
    # dropped a leading zero from E0009); previous regex \d{4} skipped it
    # entirely so it wasn't rendered AND wasn't counted as unknown. Now:
    # any [E<digits>] shape (1-5 digits) is caught, normalised via
    # zfill(4), and either resolved or counted.
    pattern = re.compile(r"\[([Ee]\d{1,5})\]")
    out_parts: list = []
    last_end = 0
    # v15.7: pre-build a known-surname set across the evidence_map so the
    # attribution check can recognise "prose surname is a valid author
    # surname for SOME other doc" — distinguishing a genuine wrong-author
    # binding from a model artefact like "He argues" where the immediate
    # lookback token is a pronoun.
    known_surnames = set()
    for _ev in evidence_map.values():
        _did = _ev.get("doc_id", "")
        _lbl = _author_surnames_only(_doc_id_to_author_label(_did))
        if _lbl:
            known_surnames.add(_lbl.lower())
    for match in pattern.finditer(text):
        raw_eid = match.group(1).upper()
        # v15.7.2: normalise to canonical 'E####' 4-digit zero-padded form.
        # Strip leading zeros first so '[E00009]' → '9' → '0009' → 'E0009'
        # (matches whatever the reasoner minted). '[E0]' or '[E]' would
        # normalise to 'E0000' and fail lookup (bumps unknown counter),
        # which is the right behaviour for a marker with no numeric ID.
        digits = raw_eid[1:].lstrip("0") or "0"
        eid = "E" + digits.zfill(4)
        ev = evidence_map.get(eid)
        if not ev:
            stats["unknown_eids"] += 1
            if len(stats["unknown_eid_snippets"]) < 10:
                lo = max(0, match.start() - 60)
                hi = min(len(text), match.end() + 60)
                stats["unknown_eid_snippets"].append(text[lo:hi].replace("\n", " "))
            # Remove the unknown marker. The surrounding prose remains
            # available to the coverage audit, which can request a retry.
            out_parts.append(text[last_end:match.start()])
            last_end = match.end()
            continue
        doc_id = ev["doc_id"]
        page = int(ev["page"])
        full_label = _doc_id_to_author_label(doc_id)  # "Author et al. (Year)"
        surnames = _author_surnames_only(full_label)   # "Author et al."
        # Extract bare year from the label.
        ym = re.match(r"^.*?\((\d{4}[a-z]?)\)\s*$", full_label)
        year = ym.group(1) if ym else ""

        # v15.6: stricter adjacency — author label must END the immediate
        # prefix (no intervening prose). Case-insensitive because the
        # model may capitalise particle prefixes (Van/van).
        prefix = text[max(0, match.start() - _EID_LOOKBACK_CHARS):match.start()]
        immediate = prefix.rstrip()
        author_immediately_before = False
        prose_surname_mismatch = None
        if surnames:
            escaped = re.escape(surnames)
            adjacency_re = re.compile(rf"(?<![A-Za-z0-9_]){escaped}$", re.IGNORECASE)
            author_immediately_before = (
                immediate.lower().endswith(surnames.lower())
                and (
                    immediate.lower() == surnames.lower()
                    or bool(adjacency_re.search(immediate))
                )
            )
            # v15.7 attribution gate: if THIS doc's surname is not
            # adjacent, check whether some OTHER known author surname is.
            # If yes, that is a prose-vs-cite mismatch — record and force
            # case-B (parenthetical with correct author) so the bare-year
            # would not be misread as belonging to the prose surname.
            if not author_immediately_before:
                for other_surnames in known_surnames:
                    if other_surnames == surnames.lower():
                        continue
                    other_esc = re.escape(other_surnames)
                    other_re = re.compile(
                        rf"(?<![A-Za-z0-9_]){other_esc}$", re.IGNORECASE
                    )
                    if other_re.search(immediate):
                        prose_surname_mismatch = other_surnames
                        break

        if prose_surname_mismatch:
            stats["attribution_mismatches"] += 1
            if len(stats["attribution_mismatch_snippets"]) < 20:
                stats["attribution_mismatch_snippets"].append({
                    "snippet": (immediate[-80:] + match.group(0)).replace("\n", " "),
                    "prose_surname": prose_surname_mismatch,
                    "expected_surnames": surnames,
                    "eid": eid,
                    "doc_id": doc_id,
                    "page": page,
                })
            # Force case-B with the CORRECT author so the wrong-named
            # prose surname is not reinforced by a bare-year cite.
            if surnames and year:
                rendered = f"({surnames} {year}, p.{page})"
            else:
                rendered = render_citation(doc_id, page)
        elif author_immediately_before and year:
            # Author label is right before the marker: emit bare
            # (Year, p.N). Surrounding prose forms 'Author (Year, p.N)'
            # which DISPLAY_CITE_RE matches.
            rendered = f"({year}, p.{page})"
        elif surnames and year:
            # Author not immediately adjacent: emit (Author Year, p.N).
            # DISPLAY_PAREN_CITE_RE matches.
            rendered = f"({surnames} {year}, p.{page})"
        else:
            # Fall back to canonical narrative renderer.
            rendered = render_citation(doc_id, page)

        out_parts.append(text[last_end:match.start()])
        out_parts.append(rendered)
        last_end = match.end()
        stats["replacements"] += 1
    out_parts.append(text[last_end:])
    return "".join(out_parts), stats


# v14 FIX-BRACKET: catch [Doc_Year] / [Doc&Doc_Year] form the model invents when
# it confuses bracket-evidence-id syntax (the [E####] form we prompt for) with a
# bracketed canonical doc_id. Observed in the v13.1 3x3 smoke run 04, which
# produced `([Nunn_2008]; [Sokoloff&Engerman_2000])` -- bare canonical doc_ids
# wrapped in square brackets, no pages. _render_evidence_id_citations only
# matches the 4-digit [E####] shape, so this form slipped through.
_BRACKETED_DOC_ID_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9&.\-]*_\d{4}[a-z]?)\]")
_EVIDENCE_ID_LIKE_RE = re.compile(r"\[[Ee][^\]\r\n]{0,8}\]")


def _render_bracketed_doc_ids(text: str, allowed_docs, doc_to_eids) -> tuple:
    """v14 FIX-BRACKET: rewrite [Doc_Year] / [Doc&Doc_Year] bracketed doc_ids
    into the display surface 'Author (Year)'. When the bracketed doc_id is in
    allowed_docs we drop the brackets and keep the display label so the prose
    still mentions the source; no page is invented because the model gave none.
    Returns (text, count). Conservative: unknown doc_ids are left untouched so
    a later validator can flag them.
    """
    if not text:
        return text or "", 0
    allowed = set(allowed_docs or ())
    count = 0

    def repl(m):
        nonlocal count
        did = m.group(1)
        if did not in allowed:
            return m.group(0)
        label = _doc_id_to_author_label(did)
        if not label:
            return m.group(0)
        count += 1
        return label

    rendered = _BRACKETED_DOC_ID_RE.sub(repl, text)
    return rendered, count


def _strip_wrapping(text: str) -> str:
    t = (text or "").strip()
    # Certain qwen3 builds leak the reasoning channel into content and leave a
    # closing tag immediately before the usable answer. Preserve only the text
    # after the final boundary. If no final answer exists, the normal empty or
    # structural gate fails closed.
    leaked_think_boundaries = list(
        re.finditer(r"</think\s*>", t, flags=re.IGNORECASE)
    )
    if leaked_think_boundaries:
        t = t[leaked_think_boundaries[-1].end():].strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    return t


# ---------------------------------------------------------------------------
# v15.9 (#6): citations.json provenance manifest + optional --linkify
# ---------------------------------------------------------------------------

def _emit_citations_manifest(
    text: str,
    allowed_docs: set,
    display_lookup: dict,
    pdf_paths_by_docid: dict,
    pdf_page_offsets: dict,
    dois_by_docid: dict,
    linkify: bool = False,
):
    """Walk every citation in the final review text; build a machine-readable
    manifest with (doc_id, page, pdf_path, pdf_page, doi_or_url) per hit.
    Optionally rewrite the text so each cite becomes a markdown link to the
    source PDF at the right page.

    Returns (maybe_rewritten_text, manifest_dict). manifest_dict has:
        {
          "citations": [{
             "index": int,          # order of appearance in the text
             "cite_text": str,      # exact matched substring
             "doc_id": str|None,
             "page": int,
             "pdf_path": str|None,
             "pdf_page": int|None,  # page + pdf_page_offset
             "doi_or_url": str|None,
             "start": int,          # char offset in the PRE-linkify text
             "end": int,
             "surface": str         # 'canonical'|'display_narrative'|'display_paren'
          }, ...],
          "distinct_docs": int,
          "linkify": bool
        }
    """
    from rrr.render import parse_citations
    citations = []
    seen_docs = set()
    for c in parse_citations(text, display_lookup=display_lookup):
        doc_id = c.get("doc_id")
        page = int(c.get("page") or 0) or None
        if doc_id and doc_id in allowed_docs:
            seen_docs.add(doc_id)
        pdf_path = pdf_paths_by_docid.get(doc_id) if doc_id else None
        offset = int(pdf_page_offsets.get(doc_id, 0) or 0) if doc_id else 0
        pdf_page = (page + offset) if (page is not None) else None
        doi = dois_by_docid.get(doc_id) if doc_id else None
        citations.append({
            "index": len(citations),
            "cite_text": c["raw"],
            "doc_id": doc_id,
            "page": page,
            "pdf_path": pdf_path,
            "pdf_page": pdf_page,
            "doi_or_url": doi or None,
            "start": c["start"],
            "end": c["end"],
            "surface": c["surface"],
        })

    if not citations:
        return text, {"citations": [], "distinct_docs": 0, "linkify": False}

    out_text = text
    if linkify:
        # Rewrite from end to start so char offsets stay valid.
        for cite in sorted(citations, key=lambda x: -x["start"]):
            if not cite["pdf_path"] or not cite["pdf_page"]:
                continue
            raw = cite["cite_text"]
            url = _pdf_page_url(cite["pdf_path"], cite["pdf_page"])
            link = f"[{raw}]({url})"
            out_text = out_text[:cite["start"]] + link + out_text[cite["end"]:]

    return out_text, {
        "citations": citations,
        "distinct_docs": len(seen_docs),
        "linkify": linkify,
    }


def _pdf_page_url(pdf_path: str, page: int) -> str:
    """Build a file:// URL with #page=N. Works in Preview, Chrome, Edge,
    Adobe, Zotero, and Obsidian."""
    from urllib.parse import quote
    p = str(pdf_path).replace("\\", "/")
    # Absolute file URL: file:///abs/path.pdf on unix, file:///C:/... on win
    if len(p) > 1 and p[1] == ":":  # Windows drive letter
        url = "file:///" + quote(p, safe="/:")
    elif p.startswith("/"):
        url = "file://" + quote(p, safe="/")
    else:
        url = "file://" + quote(p, safe="/")
    return f"{url}#page={int(page)}"


def _citation_identity(citation: dict) -> tuple:
    doc_id = str(citation.get("doc_id") or "").strip().lower()
    if doc_id:
        return ("doc", doc_id, int(citation.get("page") or 0))
    return (
        "display",
        str(citation.get("label") or "").strip().lower(),
        str(citation.get("year") or ""),
        int(citation.get("page") or 0),
    )


def _dedupe_grouped_citation_identities(text: str, display_lookup=None) -> tuple:
    """Remove repeated citation identities within semicolon groups."""
    source = text or ""
    group_re = re.compile(r"\((?P<body>[^()\n;]+(?:;[^()\n;]+)+)\)")
    citations = list(parse_citations(source, display_lookup=display_lookup))
    replacements = []
    removed = 0

    for group in group_re.finditer(source):
        body = group.group("body")
        body_start = group.start("body")
        pieces = body.split(";")
        piece_records = []
        cursor = 0
        valid_group = True
        for piece in pieces:
            start = body_start + cursor
            end = start + len(piece)
            hits = [
                citation for citation in citations
                if citation["start"] >= start and citation["end"] <= end
            ]
            if len(hits) != 1:
                valid_group = False
                break
            piece_records.append((piece.strip(), _citation_identity(hits[0])))
            cursor += len(piece) + 1
        if not valid_group:
            continue

        seen = set()
        unique_pieces = []
        for piece, identity in piece_records:
            if identity in seen:
                removed += 1
                continue
            seen.add(identity)
            unique_pieces.append(piece)
        if len(unique_pieces) != len(piece_records):
            replacements.append((
                group.start(), group.end(),
                "(" + "; ".join(unique_pieces) + ")",
            ))

    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]
    return source, removed


def _merge_adjacent_paren_cites(text: str, display_lookup=None,
                                stats: dict = None) -> tuple:
    """v15.7.2: merge whitespace-adjacent DISPLAY_PAREN cites into one
    semicolon-joined parenthetical.
        '(Ogilvie 2007, p.9) (North 1989, p.6)' →
        '(Ogilvie 2007, p.9; North 1989, p.6)'
    Chains of 3+ merge similarly. Only merges pairs separated by pure
    whitespace (space/tab/newline) — never across prose. Narrative-form
    cites ('Author (Year, p.N)') are NOT merged because 'Author' is
    sentence prose. Runs as the very last postproc step so upstream
    validators, _collect_cited_docs, and quality_manifest all see the
    unmerged per-cite form; the merged form is user-facing only.

    Returns (text, count) where count is the number of pairs merged.
    """
    if not text:
        if stats is not None:
            stats["duplicate_identities_removed"] = 0
        return text or "", 0
    matches = list(DISPLAY_PAREN_CITE_RE.finditer(text))
    groups: list = []
    current: list = [matches[0]] if matches else []
    horizontal_ws_re = re.compile(r"\A[ \t]+\Z")
    for m in matches[1:]:
        between = text[current[-1].end():m.start()]
        if horizontal_ws_re.match(between):
            current.append(m)
        else:
            if len(current) > 1:
                groups.append(current)
            current = [m]
    if len(current) > 1:
        groups.append(current)
    out: list = []
    cursor = 0
    merged_pairs = 0
    for grp in groups:
        out.append(text[cursor:grp[0].start()])
        # Strip outer '(' ')' from each match; join with '; '.
        inners = [m.group(0)[1:-1] for m in grp]
        out.append("(" + "; ".join(inners) + ")")
        cursor = grp[-1].end()
        merged_pairs += len(grp) - 1
    out.append(text[cursor:])
    merged = "".join(out) if groups else text
    deduped, duplicate_count = _dedupe_grouped_citation_identities(
        merged, display_lookup=display_lookup,
    )
    if stats is not None:
        stats["duplicate_identities_removed"] = duplicate_count
    return deduped, merged_pairs


def _parenthetical_citation_containers(text: str) -> list:
    """Return spans whose parenthetical content consists only of citations."""
    source = text or ""
    citations = list(parse_citations(source))
    containers = []
    for match in re.finditer(r"\([^()\r\n]+\)", source):
        hits = [
            citation
            for citation in citations
            if citation["start"] >= match.start()
            and citation["end"] <= match.end()
        ]
        if not hits:
            continue
        masked = list(match.group(0))
        for citation in hits:
            start = citation["start"] - match.start()
            end = citation["end"] - match.start()
            masked[start:end] = " " * (end - start)
        if re.fullmatch(r"[\s();,]*", "".join(masked)):
            containers.append((match.start(), match.end()))
    return containers


def _audit_citation_surface_contract(text: str) -> dict:
    """Detect citation surfaces that require a local section retry."""
    source = text or ""
    stats = {
        "separated_parenthetical_groups": 0,
        "separated_parenthetical_snippets": [],
        "redundant_author_parentheticals": 0,
        "redundant_author_parenthetical_snippets": [],
        "orphan_parenthetical_groups": 0,
        "orphan_parenthetical_snippets": [],
    }
    if not source:
        return stats

    paren_cites = list(DISPLAY_PAREN_CITE_RE.finditer(source))
    for left, right in zip(paren_cites, paren_cites[1:]):
        between = source[left.end():right.start()]
        if re.fullmatch(r"[ \t]*;[ \t]*", between):
            stats["separated_parenthetical_groups"] += 1
            if len(stats["separated_parenthetical_snippets"]) < 10:
                lo = max(0, left.start() - 60)
                hi = min(len(source), right.end() + 60)
                stats["separated_parenthetical_snippets"].append(
                    source[lo:hi].replace("\n", " ")
                )

    for citation in paren_cites:
        label = citation.group(1).strip()
        prefix = source[:citation.start()]
        # A prior citation to the same paper is not a narrative author
        # mention, so hide parenthetical material before inspecting prose.
        visible_prefix = re.sub(
            r"\([^()\r\n]*\)",
            lambda match: " " * len(match.group(0)),
            prefix,
        )
        boundary = 0
        for match in re.finditer(
            r"[.!?][\"')\]]*(?:[ \t]+(?=[A-Z])|\r?\n+)",
            visible_prefix,
        ):
            boundary = match.end()
        sentence_prefix = visible_prefix[boundary:]

        label_variants = [label]
        if re.search(r"\s+et\s+al\.?$", label, re.IGNORECASE):
            label_variants.append(
                re.sub(r"\s+et\s+al\.?$", "", label, flags=re.IGNORECASE)
            )
        if re.search(r"\s+and\s+", label, re.IGNORECASE):
            label_variants.append(
                re.split(
                    r"\s+and\s+", label, maxsplit=1, flags=re.IGNORECASE,
                )[0]
            )

        named_in_sentence = False
        for variant in dict.fromkeys(v for v in label_variants if v):
            author_re = re.compile(
                rf"(?<![A-Za-z'\u2019\-]){re.escape(variant)}"
                rf"(?=$|[^A-Za-z'\u2019\-])",
                re.IGNORECASE,
            )
            if author_re.search(sentence_prefix):
                named_in_sentence = True
                break
        if not named_in_sentence:
            continue

        stats["redundant_author_parentheticals"] += 1
        if len(stats["redundant_author_parenthetical_snippets"]) < 10:
            lo = max(boundary, citation.start() - 120)
            hi = min(len(source), citation.end() + 40)
            stats["redundant_author_parenthetical_snippets"].append(
                source[lo:hi].replace("\n", " ")
            )

    for start, end in _parenthetical_citation_containers(source):
        paragraph_start = 0
        for boundary in re.finditer(r"\r?\n[ \t]*\r?\n", source[:start]):
            paragraph_start = boundary.end()
        paragraph_prefix = source[paragraph_start:start]
        prior = paragraph_prefix.rstrip()
        at_paragraph_start = not prior
        follows_completed_sentence = bool(
            re.search(r"[.!?][\"'\u2019\u201d)]*$", prior)
        )
        if not (at_paragraph_start or follows_completed_sentence):
            continue
        stats["orphan_parenthetical_groups"] += 1
        if len(stats["orphan_parenthetical_snippets"]) < 10:
            lo = max(paragraph_start, start - 100)
            hi = min(len(source), end + 80)
            stats["orphan_parenthetical_snippets"].append(
                source[lo:hi].replace("\n", " ")
            )

    return stats


def _audit_raw_citation_contract(text: str, chunk_docs) -> dict:
    """Detect model-written citation surfaces before EID rendering.

    The writer contract grants the model one citation representation,
    ``[E####]``. Any document identifier, rendered citation, or page-less
    author-year form in the raw candidate is therefore a local evidence fault.
    """
    source = text or ""
    doc_ids = [
        str(doc.get("doc_id", "") or "").strip()
        for doc in (chunk_docs or [])
        if str(doc.get("doc_id", "") or "").strip()
    ]
    allowed_docs = set(doc_ids)
    display_lookup = _build_display_lookup(allowed_docs)

    rendered_citations = list(
        parse_citations(source, display_lookup=display_lookup)
    )
    bracketed_doc_ids = list(_BRACKETED_DOC_ID_RE.finditer(source))
    malformed_evidence_ids = [
        match
        for match in _EVIDENCE_ID_LIKE_RE.finditer(source)
        if not re.fullmatch(r"\[[Ee]\d{1,5}\]", match.group(0))
    ]
    raw_doc_matches = []
    pageless_author_year = []
    hybrid_doc_eid = []

    for doc_id in doc_ids:
        doc_pattern = re.compile(
            rf"(?<![A-Za-z0-9_&.\-]){re.escape(doc_id)}"
            rf"(?![A-Za-z0-9_&.\-])"
        )
        matches = list(doc_pattern.finditer(source))
        raw_doc_matches.extend(matches)
        for match in matches:
            window = source[match.end():match.end() + 40]
            if re.search(r"\s*\[[Ee]\d{1,5}\]", window):
                hybrid_doc_eid.append(match)

        label = _doc_id_to_author_label(doc_id)
        label_match = re.match(
            r"^(.*?)\s*\((\d{4}[a-z]?)\)\s*$",
            label or "",
        )
        if not label_match:
            continue
        surnames, year = label_match.groups()
        pageless_patterns = (
            re.compile(
                rf"(?<!\w){re.escape(surnames)}\s*"
                rf"\(\s*{re.escape(year)}\s*\)"
            ),
            re.compile(
                rf"\(\s*{re.escape(surnames)}\s+"
                rf"{re.escape(year)}\s*\)"
            ),
        )
        for pattern in pageless_patterns:
            pageless_author_year.extend(pattern.finditer(source))

    fault_count = (
        len(rendered_citations)
        + len(bracketed_doc_ids)
        + len(malformed_evidence_ids)
        + len(raw_doc_matches)
        + len(pageless_author_year)
        + len(hybrid_doc_eid)
    )
    snippets = []
    spans = []
    spans.extend((item["start"], item["end"]) for item in rendered_citations)
    spans.extend(match.span() for match in bracketed_doc_ids)
    spans.extend(match.span() for match in malformed_evidence_ids)
    spans.extend(match.span() for match in raw_doc_matches)
    spans.extend(match.span() for match in pageless_author_year)
    spans.extend(match.span() for match in hybrid_doc_eid)
    for start, end in sorted(set(spans))[:10]:
        lo = max(0, start - 60)
        hi = min(len(source), end + 60)
        snippets.append(source[lo:hi].replace("\n", " "))

    return {
        "raw_citation_faults": fault_count,
        "raw_rendered_citations": len(rendered_citations),
        "raw_bracketed_doc_ids": len(bracketed_doc_ids),
        "raw_malformed_evidence_ids": len(malformed_evidence_ids),
        "raw_document_ids": len(raw_doc_matches),
        "raw_pageless_author_year": len(pageless_author_year),
        "raw_hybrid_doc_eids": len(hybrid_doc_eid),
        "raw_citation_fault_snippets": snippets,
    }


def _audit_quote_contract(text: str, allowed_docs, display_lookup=None) -> dict:
    """Verify long quoted spans without mutating the candidate."""
    _discarded_mutation, stats = _strip_fabricated_quotes(
        text or "",
        allowed_docs,
        metrics=None,
        display_lookup=display_lookup,
    )
    fault_keys = (
        "fabricated_stripped",
        "strand_guard_kept",
        "fabricated_kept_no_citation",
        "fabricated_kept_doc_not_in_corpus",
        "fabricated_kept_page_not_readable",
    )
    result = dict(stats)
    result["quote_faults"] = sum(
        int(result.get(key, 0) or 0) for key in fault_keys
    )
    result["candidate_unchanged"] = True
    return result


_POST_SENTENCE_EID_GROUP_RE = re.compile(
    r"(?P<terminal>[.!?](?:[\"'\u2019\u201d)]*))"
    r"[ \t]+"
    r"(?P<markers>\[[Ee]\d{1,5}\]"
    r"(?:[ \t]+\[[Ee]\d{1,5}\])*)"
    r"[ \t]*(?=$|\r?\n|[A-Z])"
)


def _reattach_post_sentence_evidence_markers(text: str) -> tuple:
    """Move a detached EID group inside the preceding sentence boundary."""
    source = text or ""
    count = 0

    def repl(match):
        nonlocal count
        count += 1
        next_char = (
            source[match.end()]
            if match.end() < len(source)
            else ""
        )
        separator = " " if next_char and next_char not in "\r\n" else ""
        return (
            " "
            + match.group("markers")
            + match.group("terminal")
            + separator
        )

    return _POST_SENTENCE_EID_GROUP_RE.sub(repl, source), count


def _collapse_double_parens(text: str) -> tuple:
    # v8 (R5): collapse ((DocId: p.N)) -> (DocId: p.N).
    # v9.1: also collapse ((DocId: p.X): p.Y) -> (DocId: p.X).
    # v15.7: extended to the THREE display-form patterns the v15.6 audit
    # found 50 times in production output while writer_double_paren_collapsed
    # reported 0 (the helper had no call site in the chunked path AND the
    # regex was canonical-only). All three normalise (cite_inner) by stripping
    # the redundant outer parens.
    count = 0

    def repl_inner_only(m):
        nonlocal count
        count += 1
        return "(" + m.group(1) + ")"

    def repl_nested_suffix(m):
        nonlocal count
        count += 1
        inner = re.match(r"\(\(([A-Za-z0-9_&.\-]+:\s*p\.\d+)\):\s*p\.\d+\)", m.group(0))
        if inner:
            return "(" + inner.group(1) + ")"
        return m.group(0)

    def repl_group(m):
        nonlocal count
        inner = m.group(1).strip().rstrip(";,").strip()
        # Sanity: every char inside must be either a single-paren cite or
        # whitespace/punctuation. We've already constrained that via the
        # regex, but double-check no stray prose.
        if re.search(r"[A-Za-z][A-Za-z0-9]{2,}", re.sub(r"\([^()]*\)", "", inner)):
            return m.group(0)
        count += 1
        # Re-join the inner cites with '; ' for a clean group.
        cites = re.findall(r"\([^()]+?\)", inner)
        return "; ".join(cites)

    def repl_asym(m):
        # v15.7.1: '((cite).' with the OUTER paren unclosed. Strip ONE of
        # the leading '(' (the unmatched one) and keep the well-formed inner.
        nonlocal count
        count += 1
        return "(" + m.group(1)

    text, _ = (_DOUBLE_PAREN_CITE_RE.subn(repl_inner_only, text))
    text, _ = (_NESTED_PAGE_SUFFIX_RE.subn(repl_nested_suffix, text))
    text, _ = (_DOUBLE_PAREN_DISPLAY_RE.subn(repl_inner_only, text))
    text, _ = (_DOUBLE_PAREN_DISPLAY_GROUP_RE.subn(repl_group, text))
    text, _ = (_ASYM_OUTER_PAREN_RE.subn(repl_asym, text))
    return text, count


def _count_author_led_openings(text: str) -> int:
    # v8 (R5): count sentences that open with "Author argues / posits / ...".
    # v10: kept as metric only; user explicitly ditched the prompt rule because
    # mistral-small:24b and above produce them sparingly, in natural academic
    # register, not as a parade pattern.
    count = 0
    for sentence in _split_sentences_for_cleanup(text):
        s = sentence.strip()
        if not s:
            continue
        head = re.sub(r"^[\(\[\"\'\s]+", "", s)[:120]
        if _AUTHOR_VERB_RE.search(head):
            count += 1
    return count


def _count_source_parade_paragraphs(text: str) -> int:
    """Count paragraphs containing three or more author-led sentences."""
    count = 0
    for paragraph in _prose_paragraphs(text):
        author_led = 0
        for sentence in _split_sentences_for_cleanup(paragraph):
            head = re.sub(r"^[\(\[\"\'\s]+", "", sentence.strip())[:120]
            if head and _AUTHOR_VERB_RE.search(head):
                author_led += 1
        if author_led >= 3:
            count += 1
    return count


# ============================================================================
# v10: STYLE ENFORCEMENT — detect-then-LLM-rewrite for ambiguous violations.
# Strippable patterns (trailing significance) are handled mechanically first;
# the remaining sentences with em-dashes, forbidden lexemes, adversative
# framings, or colon setups are routed to a single batched LLM rewrite call
# per writer postprocess pass.
# ============================================================================

def _strip_trailing_significance(text: str) -> tuple:
    # v10 (rule 6): mechanically strip "and that matters" / "which matters
    # because ..." trailing clauses. These have a stable shape and live at end
    # of sentence; removing them does not change the substantive claim head.
    matches = list(_TRAILING_SIGNIFICANCE_RE.finditer(text or ""))
    if not matches:
        return text or "", 0
    cleaned = _TRAILING_SIGNIFICANCE_RE.sub("", text)
    # Recompose terminal punctuation if the strip removed it.
    cleaned = re.sub(r"([A-Za-z0-9_\)\]])(?=\n|$)", r"\1.", cleaned)
    return cleaned, len(matches)


def _mechanical_dash_replace(text: str) -> tuple:
    # v13: mechanical fallback for em-dashes when the LLM style rewriter rejects
    # the rewrite because preserving every (Year, p.N) citation token would
    # require an unsupported edit. The post-v12 prose audit found one such
    # sentence in the opening of the v12 smoke ("institutions—defined as the
    # rules and norms governing social interactions—determine long-term
    # economic outcomes"): the model placed em-dashes around a parenthetical
    # gloss, and the rewriter couldn't replace them without rephrasing the
    # surrounding clauses, which would risk dropping citations. The mechanical
    # substitution is safe because em/en dashes flanking a parenthetical gloss
    # always reduce to commas, and dashes used as terminal pause always reduce
    # to commas or semicolons. We pick comma uniformly — it preserves the gloss
    # structure and never changes citation tokens.
    if not text:
        return text or "", 0
    # v15.14: never touch a dash BETWEEN DIGITS. "1846–1873", "pp. 5–9" and
    # similar numeric ranges are factual content in an economic-history
    # corpus; replacing the dash with a comma ("1846, 1873") silently
    # corrupts dates. Only gloss/pause dashes (non-numeric context) reduce
    # to commas.
    count = 0

    def _dash_sub(m):
        nonlocal count
        s, e = m.start(), m.end()
        prev_c = text[s - 1] if s > 0 else ""
        next_c = text[e] if e < len(text) else ""
        if prev_c.isdigit() and next_c.isdigit():
            return m.group(0)
        count += 1
        return ","

    cleaned = _TYPOGRAPHIC_DASH_RE.sub(_dash_sub, text)
    if count == 0:
        return text, 0
    # The dash often sat between two words with no surrounding space ("word—word")
    # or with single spaces ("word — word"). Either way, ", " is the target.
    # Collapse "  ," and ",  " and ", ," that the naive substitution can create.
    # v15.14: horizontal whitespace only — the old \s* forms could eat a
    # newline after a line-final comma and silently join paragraphs.
    cleaned = re.sub(r"[ \t]*,[ \t]*", ", ", cleaned)
    # Don't double up on existing punctuation: ", ," -> ",", ". ," -> ".",
    cleaned = re.sub(r"([.,;:!?])[ \t]*,", r"\1", cleaned)
    cleaned = re.sub(r",[ \t]*([.,;:!?])", r"\1", cleaned)
    return cleaned, count


def _drop_zero_citation_paragraphs(text: str, keep_closing: bool = False) -> tuple:
    """v13: hard enforcement of rule 9 (each paragraph must integrate at least
    one cited document; the prompt asks for >=2 but the safety net catches the
    worst case of zero). The v12 smoke had three zero-citation paragraphs
    (paras 4, 9, 12 — the closing) that the prose-grounding contract forbids.
    Drop them.

    Returns (cleaned_text, [removed_paragraph_snippets], kept_closing_snippet|None).
    Conservative: we only count CANONICAL `(Doc_Year: p.N)` cites or DISPLAY-form
    `Author (Year, p.N)` / `(Author Year, p.N)` cites. If neither shape is
    present the paragraph cites zero documents and gets dropped.

    v14: when keep_closing=True and the LAST non-empty paragraph would be
    dropped, keep it anyway (a weak closing reads better than a missing
    closing — the closing is structurally important to the essay). The
    returned third tuple element is the snippet of that kept closing (or None).
    """
    if not text:
        return text or "", [], None
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return text or "", [], None
    last_idx = len(paragraphs) - 1
    kept = []
    removed = []
    kept_closing_snippet = None
    # v15.7: recognise unrendered [E####] markers as a citation form too.
    # When the renderer can't map a marker (unknown eid), it leaves the
    # bare marker in place AND bumps writer_unknown_evidence_id. Without
    # this check, the surrounding prose would be silently dropped as
    # zero-citation when the real failure is a single bad marker.
    # v15.7.2: also recognise 1-3 digit variants and non-numeric E-prefixed
    # sentinels (e.g. '[E9]', '[Evidence]') so a hallucinated-marker
    # paragraph isn't lost — the quality manifest surfaces them separately.
    _UNRENDERED_EID_RE = re.compile(r"\[[Ee]\w*\]")
    for i, para in enumerate(paragraphs):
        has_canonical = bool(CITE_RE.search(para))
        has_display_narrative = bool(DISPLAY_CITE_RE.search(para))
        has_display_paren = bool(DISPLAY_PAREN_CITE_RE.search(para))
        has_unrendered_eid = bool(_UNRENDERED_EID_RE.search(para))
        if has_canonical or has_display_narrative or has_display_paren or has_unrendered_eid:
            kept.append(para)
            continue
        snippet = re.sub(r"\s+", " ", para.strip())[:200]
        snippet = snippet + ("…" if len(para.strip()) > 200 else "")
        if keep_closing and i == last_idx:
            # Closing is structurally important; keep the weak closing rather
            # than truncate the essay. Emit the metric via the caller.
            kept.append(para)
            kept_closing_snippet = snippet
            continue
        removed.append(snippet)
    cleaned = "\n\n".join(kept)
    return cleaned, removed, kept_closing_snippet


def _apply_zero_citation_gate(text: str, enforce_citation_integrity: bool):
    """Apply paragraph citation removal only when validation is enabled."""
    if not enforce_citation_integrity:
        return text or "", [], None
    return _drop_zero_citation_paragraphs(text, keep_closing=False)


def _apply_residual_duplicate_gate(final_structure: dict,
                                   residual_duplicates: int,
                                   enforce_citation_integrity: bool) -> dict:
    """Record duplicate residue while gating it only in validation mode."""
    result = dict(final_structure)
    result["residual_duplicate_citation_identities"] = residual_duplicates
    result["ok"] = bool(
        result.get("ok")
        and (
            not enforce_citation_integrity
            or residual_duplicates == 0
        )
    )
    return result


def _classify_sentence_violations(sentence: str) -> list:
    reasons = []
    if _TYPOGRAPHIC_DASH_RE.search(sentence):
        reasons.append("em_dash")
    if _FORBIDDEN_LEXEME_RE.search(sentence):
        reasons.append("forbidden_lexeme")
    if _ADVERSATIVE_RE.search(sentence):
        reasons.append("adversative")
    if _COLON_SETUP_RE.search(sentence):
        reasons.append("colon_setup")
    return reasons


_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n")
_TERMINAL_PUNCTUATION_RE = re.compile(r"[.!?](?:[\"'\u2019\u201d)\]]*)$")
_TERMINAL_TOKEN_RE = re.compile(r"[.!?](?:[\"'\u2019\u201d)\]]*)")
_EVIDENCE_ID_AUDIT_RE = re.compile(r"\[[Ee]\d{1,5}\]")
_CITATION_SUFFIX_DELIMITER_RE = re.compile(r"[\s;,()\[\]]*")
_NONTERMINAL_ABBREVIATION_RES = (
    re.compile(r"\bet\s+al\.", re.IGNORECASE),
    re.compile(r"\b(?:e\.g|i\.e|cf)\.", re.IGNORECASE),
    re.compile(r"\bpp?\.(?=\s*\d)", re.IGNORECASE),
)


def _prose_paragraphs(text: str) -> list:
    """Return non-empty prose paragraphs without crossing blank lines."""
    return [
        paragraph.strip()
        for paragraph in _PARAGRAPH_BREAK_RE.split(text or "")
        if paragraph.strip()
    ]


def _sentences_paragraph_local(text: str) -> list:
    """Split sentences within each paragraph and retain source order.

    A paragraph that lacks terminal punctuation remains one unit. It never
    absorbs the opening sentence of the following paragraph.
    """
    sentences = []
    for paragraph in _prose_paragraphs(text):
        sentences.extend(_split_sentences_for_cleanup(paragraph))
    return sentences


def _ends_with_terminal_punctuation(text: str) -> bool:
    candidate = (text or "").strip()

    def mask_nonterminal_abbreviation_periods(value: str) -> str:
        masked = list(value)
        for pattern in _NONTERMINAL_ABBREVIATION_RES:
            for match in pattern.finditer(value):
                for index in range(match.start(), match.end()):
                    if masked[index] == ".":
                        masked[index] = " "
        return "".join(masked)

    # A final abbreviation period does not establish a complete sentence.
    # Apply the same abbreviation mask to the direct path and the
    # citation-suffix path so both variants fail closed.
    if _TERMINAL_PUNCTUATION_RE.search(
        mask_nonterminal_abbreviation_periods(candidate)
    ):
        return True

    # Evidence markers are rendered as citations before this audit. Models
    # sometimes place those markers after an already complete sentence, which
    # leaves the rendered paragraph ending in a citation cluster rather than
    # the sentence's period. Remove citation-only material for the terminal
    # check. An unfinished clause followed by a citation still lacks terminal
    # punctuation and therefore still fails closed.
    citations = list(parse_citations(candidate))
    spans = [
        (citation["start"], citation["end"])
        for citation in citations
    ]
    # In a grouped narrative citation such as
    # ``Austin (2008, p.1; North 1989, p.2)``, the parser's first exact span
    # begins at the year. Mask the immediately preceding parsed author label
    # while leaving the shared parenthesis for delimiter validation.
    for citation in citations:
        if citation.get("surface") != "display_narrative":
            continue
        if not re.match(r"^\d{4}", citation.get("raw", "")):
            continue
        label = str(citation.get("label", "") or "").strip()
        if not label:
            continue
        prefix = candidate[:citation["start"]]
        label_match = re.search(
            rf"(?<!\w)({re.escape(label)})\s*\($",
            prefix,
            flags=re.IGNORECASE,
        )
        if label_match:
            spans.append(label_match.span(1))
    spans.extend(
        (match.start(), match.end())
        for match in _EVIDENCE_ID_AUDIT_RE.finditer(candidate)
    )
    if not spans:
        return False

    masked = list(candidate)
    for start, end in spans:
        masked[start:end] = " " * (end - start)
    masked_text = "".join(masked)

    # Periods in common scholarly abbreviations cannot establish that the
    # preceding prose is a complete sentence. Mask them with spaces while
    # preserving offsets, using the same surfaces protected by the sentence
    # splitter below.
    masked_text = mask_nonterminal_abbreviation_periods(masked_text)

    def balanced_delimiters(suffix: str) -> bool:
        stack = []
        pairs = {")": "(", "]": "["}
        for char in suffix:
            if char in "([":
                stack.append(char)
            elif char in ")]":
                if not stack or stack.pop() != pairs[char]:
                    return False
        return not stack

    for terminal in reversed(list(_TERMINAL_TOKEN_RE.finditer(masked_text))):
        suffix = masked_text[terminal.end():]
        if not _CITATION_SUFFIX_DELIMITER_RE.fullmatch(suffix):
            continue
        if not balanced_delimiters(suffix):
            continue
        if any(start >= terminal.end() for start, _ in spans):
            return True
    return False


def _audit_section_structure(text: str, section_kind: str) -> dict:
    """Audit paragraph completeness and the closing's minimum substance."""
    paragraphs = _prose_paragraphs(text)
    incomplete = [
        {
            "paragraph_index": index,
            "snippet": _clip(paragraph, n=180),
        }
        for index, paragraph in enumerate(paragraphs)
        if not _ends_with_terminal_punctuation(paragraph)
    ]
    word_count = _count_words(text)
    closing_word_floor_ok = (
        section_kind != "closing" or word_count >= _MIN_CLOSING_WORDS
    )
    closing_word_ceiling_ok = (
        section_kind != "closing" or word_count <= _MAX_CLOSING_WORDS
    )
    closing_paragraph_count_ok = (
        section_kind != "closing" or len(paragraphs) == 1
    )
    return {
        "paragraph_count": len(paragraphs),
        "word_count": word_count,
        "incomplete_paragraph_count": len(incomplete),
        "incomplete_paragraphs": incomplete,
        "closing_word_floor": (
            _MIN_CLOSING_WORDS if section_kind == "closing" else 0
        ),
        "closing_word_floor_ok": closing_word_floor_ok,
        "closing_word_ceiling": (
            _MAX_CLOSING_WORDS if section_kind == "closing" else 0
        ),
        "closing_word_ceiling_ok": closing_word_ceiling_ok,
        "closing_paragraph_count_ok": closing_paragraph_count_ok,
        "ok": (
            bool(paragraphs)
            and not incomplete
            and closing_word_floor_ok
            and closing_word_ceiling_ok
            and closing_paragraph_count_ok
        ),
    }


def _collect_style_violations(text: str):
    """Return list of (sentence_index, sentence_text, [reasons]) for the
    sentences in `text` that warrant LLM rewrite. Paragraph structure is
    preserved by splitting every paragraph independently. An unfinished
    paragraph tail can therefore never fuse with the next paragraph's opener.
    """
    sentences = _sentences_paragraph_local(text)
    violations = []
    for idx, sent in enumerate(sentences):
        s = sent.strip()
        if not s:
            continue
        reasons = _classify_sentence_violations(s)
        if reasons:
            violations.append((idx, s, reasons))
    return sentences, violations


def _build_style_rewrite_prompt(violations) -> str:
    """One short prompt covering all flagged sentences from this section."""
    parts = [
        "You are revising sentences from a literature review. Rewrite each "
        "numbered sentence below so that it expresses the same meaning while "
        "observing ALL of the following constraints:",
        "",
        "- Do not use em dashes or en dashes. Use commas, periods, semicolons, "
        "or parentheses.",
        "- Do not use the words 'legible', 'sharp', or 'sharply'.",
        "- Do not use adversative framings as the default prose engine. Avoid "
        "'not X but Y', 'rather than', 'unlike', 'in contrast', and "
        "contrastive 'instead of'.",
        "- Do not introduce a claim with a colon followed by an explanatory "
        "clause. Use a full sentence.",
        "- Preserve every 'Author (Year, p.N)' citation exactly as written.",
        "- Preserve the substantive meaning. Do not add or remove evidence.",
        "- Keep prose natural and varied. Do not produce staccato short "
        "sentences in place of the originals.",
        "",
        "Sentences:",
    ]
    for n, (_, sent, _) in enumerate(violations, start=1):
        parts.append(f"{n}. {sent}")
    parts.append("")
    parts.append(
        "Return ONLY a JSON object with one key 'rewritten' whose value is a "
        "JSON array of strings, the rewritten sentences in the same order. "
        "Do not include any commentary."
    )
    return "\n".join(parts)


_EVIDENCE_ID_TOKEN_RE = re.compile(r"\[E\d{4,5}\]")
_STYLE_CITATION_GUARD_VERSION = "all-surfaces-v1"


def _citation_fingerprints(text: str):
    """Return a surface-agnostic citation multiset for rewrite guards."""
    fingerprints = []
    for citation in parse_citations(text or ""):
        label = (
            citation.get("label") or citation.get("doc_id") or ""
        ).strip().lower()
        fingerprints.append((
            label,
            str(citation.get("year") or ""),
            int(citation.get("page") or 0),
        ))
    fingerprints.extend(
        ("__eid__", token, 0)
        for token in _EVIDENCE_ID_TOKEN_RE.findall(text or "")
    )
    return sorted(fingerprints)


def _rewrite_style_violations(sentences, violations, metrics=None):
    """Run one batched LLM call to rewrite all flagged sentences.

    Returns a new list of sentences (same length as input). On any failure the
    original sentences are returned unchanged.
    """
    if not violations:
        return sentences, 0, "no_violations"
    prompt = _build_style_rewrite_prompt(violations)
    system = "You revise academic prose. Be concise. Return JSON only."
    prompt_path = _dump_writer_prompt("style_rewrite", system, prompt)
    try:
        import ollama
        import time
        start = time.perf_counter()
        res = ollama.chat(
            model=_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            options=_OPTIONS_STYLE_REWRITE,
            keep_alive=_KEEP_ALIVE,
            format="json",
            stream=False,
        )
        raw = (res.get("message", {}).get("content") or "").strip()
        _dump_writer_response(prompt_path, response=raw)
        if metrics:
            metrics.record_llm("style_rewrite", _MODEL, options=_OPTIONS_STYLE_REWRITE,
                               duration_s=time.perf_counter() - start,
                               prompt_chars=len(prompt),
                               response_chars=len(raw))
    except Exception as e:
        _dump_writer_response(prompt_path, error=e)
        if metrics:
            metrics.record_llm("style_rewrite", _MODEL, options=_OPTIONS_STYLE_REWRITE,
                               success=False, error=e)
        return sentences, 0, f"llm_call_failed:{type(e).__name__}"

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return sentences, 0, "json_braces_missing"
        obj = json.loads(raw[start:end + 1])
        rewritten = obj.get("rewritten", [])
        if not isinstance(rewritten, list) or len(rewritten) != len(violations):
            return sentences, 0, "wrong_count"
    except Exception:
        return sentences, 0, "json_parse_failed"

    new_sentences = list(sentences)
    rewrites_applied = 0
    for (idx, original, _reasons), new_text in zip(violations, rewritten):
        if not isinstance(new_text, str):
            continue
        new_text = new_text.strip()
        if not new_text:
            continue
        # One input item is one sentence from one paragraph. A rewrite that
        # injects a line break or changes the sentence count would make the
        # paragraph layout ambiguous, so retain the original sentence.
        if "\n" in new_text or "\r" in new_text:
            continue
        if (
            len(_split_sentences_for_cleanup(new_text)) != 1
            or not _ends_with_terminal_punctuation(new_text)
        ):
            continue
        # Refuse a rewrite that REINTRODUCES the same violation (model loop).
        if _classify_sentence_violations(new_text):
            continue
        # Refuse a rewrite that drops, duplicates, or changes any citation.
        # The previous guard inspected only narrative '(Year, p.N)' tokens.
        # Parenthetical '(Author Year, p.N)' citations therefore compared as
        # two empty sets, allowing the style model to change a cited paper or
        # page. Fingerprint every production surface and evidence-ID marker.
        original_cites = _citation_fingerprints(original)
        new_cites = _citation_fingerprints(new_text)
        if original_cites != new_cites:
            continue
        new_sentences[idx] = new_text
        rewrites_applied += 1

    return new_sentences, rewrites_applied, "ok"


def _splice_sentences_back_checked(original_text: str, new_sentences) -> tuple:
    """Reconstruct paragraph-locally, returning the original on any mismatch."""
    if not original_text:
        return "", "empty"

    parts = re.split(r"(\n[ \t]*\n)", original_text)
    paragraph_indices = list(range(0, len(parts), 2))
    expected_counts = [
        len(_split_sentences_for_cleanup(parts[index]))
        for index in paragraph_indices
    ]
    if sum(expected_counts) != len(new_sentences):
        return original_text, "sentence_count_mismatch"
    for sentence in new_sentences:
        if not isinstance(sentence, str) or "\n" in sentence or "\r" in sentence:
            return original_text, "paragraph_injection"
        if len(_split_sentences_for_cleanup(sentence.strip())) != 1:
            return original_text, "replacement_sentence_count_mismatch"

    out_parts = list(parts)
    cursor = 0
    for part_index, n in zip(paragraph_indices, expected_counts):
        para = parts[part_index]
        if n == 0:
            out_parts[part_index] = para
            continue
        replacement = new_sentences[cursor:cursor + n]
        cursor += n
        if len(replacement) != n or any(not s.strip() for s in replacement):
            return original_text, "empty_replacement"
        out_parts[part_index] = " ".join(s.strip() for s in replacement)

    rewritten = "".join(out_parts).strip()
    if len(_prose_paragraphs(rewritten)) != len(_prose_paragraphs(original_text)):
        return original_text, "paragraph_count_mismatch"
    return rewritten, "ok"


def _splice_sentences_back(original_text: str, new_sentences) -> str:
    """Reconstruct text while retaining every original paragraph boundary."""
    rewritten, _reason = _splice_sentences_back_checked(
        original_text, new_sentences,
    )
    return rewritten


def _apply_style_enforcement(text: str, metrics=None):
    """v10 single entry point for prose-style cleanup.

    Order matters:
      1. Strip the trailing-significance clause shape mechanically. Safe.
      2. Detect remaining violations (em-dash, lexeme, adversative, colon).
      3. Batch-LLM-rewrite the flagged sentences and splice back.
      4. v13: any em-dashes that survived (because the rewriter rejected the
         rewrite to preserve citation tokens) get mechanically replaced with
         commas. This is the post-v12 fix for the opening-sentence em-dash that
         the LLM rewriter couldn't touch because the sentence had four cites
         and the rewriter's citation-preservation guard fired.

    Returns (text, stats_dict). Stats keys: 'trailing_stripped', 'violations',
    'rewrites_applied', 'fallback_reason', 'mechanical_dashes_replaced'.
    """
    # v13: RRR_STYLE_ENFORCE retired (always on); a no-style-enforcement
    # writer is no longer a supported configuration.
    if not text:
        return text, {"trailing_stripped": 0, "violations": 0,
                      "rewrites_applied": 0, "fallback_reason": "empty",
                      "mechanical_dashes_replaced": 0,
                      "paragraphs_before": 0, "paragraphs_after": 0,
                      "paragraph_layout_preserved": True,
                      "layout_fallback_reason": "empty",
                      "citation_fingerprint_guard":
                          _STYLE_CITATION_GUARD_VERSION}

    text, trailing_stripped = _strip_trailing_significance(text)
    paragraphs_before = len(_prose_paragraphs(text))

    sentences, violations = _collect_style_violations(text)
    if not violations:
        # Even with no other violations, a stray em-dash can sit in text that
        # didn't make the violation list (e.g. dashes in the input were already
        # there before the run). Run the mechanical pass anyway as a safety net.
        text, mechanical_replaced = _mechanical_dash_replace(text)
        return text, {"trailing_stripped": trailing_stripped, "violations": 0,
                      "rewrites_applied": 0, "fallback_reason": "no_violations",
                      "mechanical_dashes_replaced": mechanical_replaced,
                      "paragraphs_before": paragraphs_before,
                      "paragraphs_after": len(_prose_paragraphs(text)),
                      "paragraph_layout_preserved": (
                          len(_prose_paragraphs(text)) == paragraphs_before
                      ),
                      "layout_fallback_reason": "no_rewrite",
                      "citation_fingerprint_guard":
                          _STYLE_CITATION_GUARD_VERSION}

    new_sentences, applied, reason = _rewrite_style_violations(
        sentences, violations, metrics=metrics,
    )
    if applied == 0:
        # LLM rewrite produced nothing usable. Still clean em-dashes
        # mechanically so a citation-preservation rejection doesn't leave the
        # dashes in the output.
        text, mechanical_replaced = _mechanical_dash_replace(text)
        return text, {"trailing_stripped": trailing_stripped,
                      "violations": len(violations),
                      "rewrites_applied": 0,
                      "fallback_reason": reason,
                      "mechanical_dashes_replaced": mechanical_replaced,
                      "paragraphs_before": paragraphs_before,
                      "paragraphs_after": len(_prose_paragraphs(text)),
                      "paragraph_layout_preserved": (
                          len(_prose_paragraphs(text)) == paragraphs_before
                      ),
                      "layout_fallback_reason": "no_usable_rewrite",
                      "citation_fingerprint_guard":
                          _STYLE_CITATION_GUARD_VERSION}
    rewritten_text, layout_reason = _splice_sentences_back_checked(
        text, new_sentences,
    )
    if layout_reason != "ok":
        applied = 0
    # v13: post-LLM-rewrite mechanical sweep. Sentences whose rewrites were
    # individually rejected (citation drift) still hold their em-dashes.
    rewritten_text, mechanical_replaced = _mechanical_dash_replace(rewritten_text)
    return rewritten_text, {"trailing_stripped": trailing_stripped,
                            "violations": len(violations),
                            "rewrites_applied": applied,
                            "fallback_reason": reason,
                            "mechanical_dashes_replaced": mechanical_replaced,
                            "paragraphs_before": paragraphs_before,
                            "paragraphs_after": len(_prose_paragraphs(rewritten_text)),
                            "paragraph_layout_preserved": (
                                len(_prose_paragraphs(rewritten_text))
                                == paragraphs_before
                            ),
                            "layout_fallback_reason": layout_reason,
                            "citation_fingerprint_guard":
                                _STYLE_CITATION_GUARD_VERSION}


# v9 (R6): word-level content tokens for "shares >=K content tokens with an
# earlier sentence" check. Strips punctuation and short stopwords so the
# overlap measure reflects substantive content rather than function-word
# coincidence. Tuned to be permissive — we only drop on STRICT-SUBSET citation
# overlap PLUS strong content overlap, so a high bar here is fine.
_REDUNDANCY_STOPWORDS = frozenset({
    "the", "and", "of", "in", "to", "a", "an", "is", "are", "was", "were",
    "for", "on", "by", "with", "as", "at", "from", "that", "this", "these",
    "those", "it", "its", "their", "they", "we", "be", "been", "has", "have",
    "had", "or", "but", "not", "no", "nor", "if", "than", "then", "so", "such",
    "into", "out", "over", "under", "between", "among", "while", "when",
    "where", "which", "who", "whom", "whose", "what", "how", "why", "also",
    "however", "moreover", "furthermore", "therefore", "thus", "hence",
    "instance", "example", "p", "pp",
})


def _redundancy_tokens(sentence: str) -> set:
    raw = re.findall(r"[A-Za-z][A-Za-z0-9_]+", sentence.lower())
    return {t for t in raw if len(t) >= 4 and t not in _REDUNDANCY_STOPWORDS}


def _sentence_citation_pairs(sentence: str, display_lookup=None):
    # v15.7: was canonical-only; now resolves display surfaces too via
    # parse_citations + display_lookup. Drop-aware-of-surface so the
    # redundancy walker doesn't keep a sentence that "has no canonical
    # cite" simply because the cite is in display form.
    pairs = set()
    for c in parse_citations(sentence, display_lookup=display_lookup):
        if c["doc_id"] is None:
            continue
        pairs.add((c["doc_id"], c["page"]))
    return pairs


def _sentence_doc_ids(sentence: str, display_lookup=None) -> set:
    """v15.7: helper for the attribution-aware drop guard. Returns just the
    doc_id set (no pages) for a sentence."""
    return {p[0] for p in _sentence_citation_pairs(sentence, display_lookup=display_lookup)}


def _drop_would_strand_surname(prev_text: str, sent: str, allowed_docs,
                                 display_lookup=None) -> bool:
    """v15.7 drop-aware-of-surname guard. Returns True if dropping `sent`
    would leave the IMMEDIATELY-PRECEDING prose in prev_text with an
    author surname that has no surviving cite of theirs in the dropped-
    sentence's cite set — exactly the GD 04 Kuznets→Mokyr failure class.

    Heuristic: take the last 80 chars of prev_text, scan for any known
    author surname (from allowed_docs / display_lookup), and check if any
    of those surnames maps to a doc_id in sent's cite set. If yes, the
    drop strands the surname (because the cite that gave it semantic
    cover would be removed).
    """
    if not prev_text or not sent:
        return False
    tail = prev_text[-120:]
    sent_doc_ids = _sentence_doc_ids(sent, display_lookup=display_lookup)
    if not sent_doc_ids:
        return False
    # Build surname -> doc_ids index for the allowed corpus.
    surname_to_docids: dict = {}
    for did in (allowed_docs or set()):
        surnames = _author_surnames_only(_doc_id_to_author_label(did)).lower()
        if surnames:
            surname_to_docids.setdefault(surnames, set()).add(did)
    # Check tail for any surname whose doc set overlaps sent's cites.
    tail_lower = tail.lower()
    for surnames, dids in surname_to_docids.items():
        if surnames in tail_lower and (dids & sent_doc_ids):
            return True
    return False


def _drop_cross_section_redundancy(text: str, min_token_overlap: int = 4,
                                     display_lookup=None, allowed_docs=None):
    """v9 (R6): walk the assembled review sentence by sentence; drop a sentence
    when BOTH:
      (a) every (doc, page) citation it contains has already been cited earlier
          in the document (strict-subset rule); AND
      (b) it shares >= min_token_overlap content tokens with at least one
          earlier kept sentence (high content overlap).
    Sentences with at least one novel citation are always kept; sentences
    citing already-seen evidence are kept if they introduce a genuinely
    distinct content angle. Returns (cleaned_text, removed_examples).
    """
    if not text:
        return text, []
    # Process paragraph by paragraph so we don't merge cross-paragraph text.
    paragraphs = text.split("\n\n")
    seen_pairs: set = set()
    seen_tokens: list = []  # list of token-sets per kept sentence
    removed_examples = []
    new_paragraphs = []
    for para in paragraphs:
        lines = para.split("\n")
        kept_lines = []
        for line in lines:
            sentences = _split_sentences_for_cleanup(line)
            if not sentences:
                kept_lines.append(line)
                continue
            kept_sentences = []
            for sent in sentences:
                sent_pairs = _sentence_citation_pairs(sent, display_lookup=display_lookup)
                # No citation: keep (we never drop uncited prose here)
                if not sent_pairs:
                    kept_sentences.append(sent)
                    continue
                # Any novel citation: keep, register pairs
                if not sent_pairs.issubset(seen_pairs):
                    kept_sentences.append(sent)
                    seen_pairs |= sent_pairs
                    seen_tokens.append(_redundancy_tokens(sent))
                    continue
                # All citations already seen: check token overlap
                this_tokens = _redundancy_tokens(sent)
                max_overlap = max((len(this_tokens & prev) for prev in seen_tokens), default=0)
                if max_overlap >= min_token_overlap:
                    # v15.7: drop-aware-of-surname guard. Refuse the drop
                    # when removing `sent` would leave a preceding-prose
                    # surname stranded (its only cite was in `sent`).
                    prev_text = " ".join(kept_sentences)
                    if _drop_would_strand_surname(
                        prev_text, sent, allowed_docs, display_lookup=display_lookup,
                    ):
                        kept_sentences.append(sent)
                        seen_tokens.append(this_tokens)
                        continue
                    snippet = re.sub(r"\s+", " ", sent.strip())[:140]
                    removed_examples.append({
                        "snippet": snippet + ("…" if len(snippet) == 140 else ""),
                        "overlap_tokens": int(max_overlap),
                        "cited_pairs": sorted(list(sent_pairs)),
                    })
                    continue
                # Same citation set but distinct content: keep
                kept_sentences.append(sent)
                seen_tokens.append(this_tokens)
            if kept_sentences:
                kept_lines.append(" ".join(kept_sentences))
        if kept_lines:
            new_paragraphs.append("\n".join(kept_lines))
    cleaned = "\n\n".join(new_paragraphs)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, removed_examples


def _drop_body_redundancy_preserving_closing(
        text: str, min_token_overlap: int = 4,
        display_lookup=None, allowed_docs=None) -> tuple:
    """Apply cross-section redundancy only before the closing marker."""
    split = _split_marked_closing(text)
    if not split["ok"]:
        raise ValueError("closing identity missing before redundancy cleanup")
    body, removed = _drop_cross_section_redundancy(
        split["body"],
        min_token_overlap=min_token_overlap,
        display_lookup=display_lookup,
        allowed_docs=allowed_docs,
    )
    joined = (
        body.rstrip()
        + "\n\n"
        + _CLOSING_START_MARKER
        + " "
        + split["closing"].lstrip()
    )
    return joined, removed, _count_words(split["closing"])


def _split_sentences_for_cleanup(line: str):
    sentinel = "__RRR_DOT__"

    def protect(m):
        return m.group(0).replace(".", sentinel)

    protected = re.sub(r"\bet\s+al\.", protect, line, flags=re.IGNORECASE)
    protected = re.sub(r"\b(?:e\.g|i\.e|cf)\.", protect, protected, flags=re.IGNORECASE)
    protected = re.sub(r"\bpp?\.(?=\s*\d)", protect, protected, flags=re.IGNORECASE)
    parts = re.split(r'(?<=[.!?])\s+', protected)
    return [p.replace(sentinel, ".") for p in parts if p.strip()]


def _remove_invalid_citations(text: str, allowed_docs: set, allowed_pairs=None,
                                display_lookup=None) -> tuple:
    # v15.7: surface-agnostic via parse_citations(text, display_lookup).
    # Previously CITE_RE-only — counted writer_removed_citations=0 in all
    # v15.6 runs while the audit found 5 mis-attribution blockers and 16
    # display-form leaks. With display_lookup provided, display-form
    # cites resolve to a doc_id and the validator can detect:
    #   - unknown_doc: doc_id not in allowed_docs (cite to a paper outside
    #     this chunk's allowed set)
    #   - invalid_page: doc_id known but (doc, page) not in allowed_pairs
    #   - unresolved_display: display-form whose (label, year) does not
    #     resolve via display_lookup at all (probably an author label the
    #     model invented or a collision in the lookup)
    allowed_lower = {did.lower() for did in allowed_docs}
    lower_to_canonical = {did.lower(): did for did in allowed_docs}
    allowed_pairs = set(allowed_pairs or [])

    removed = []

    def find_invalid_citations_in_text(txt):
        invalid = []
        for c in parse_citations(txt, display_lookup=display_lookup):
            page = int(c["page"])
            doc_id = c["doc_id"]
            if doc_id is None:
                # Display-form that did not resolve via display_lookup.
                invalid.append((
                    c["start"], c["end"],
                    c.get("label") or "?",
                    page,
                    "unresolved_display",
                ))
                continue
            if doc_id.lower() not in allowed_lower:
                invalid.append((c["start"], c["end"], doc_id, page, "unknown_doc"))
                continue
            canonical = lower_to_canonical.get(doc_id.lower(), doc_id)
            if allowed_pairs and (canonical, page) not in allowed_pairs:
                invalid.append((c["start"], c["end"], canonical, page, "invalid_page"))
        return invalid

    invalid_citations = find_invalid_citations_in_text(text)
    if not invalid_citations:
        return text, []

    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line_invalid = find_invalid_citations_in_text(line)
        if not line_invalid:
            cleaned_lines.append(line)
            continue
        sentences = _split_sentences_for_cleanup(line)
        kept_sentences = []
        for sent in sentences:
            sent_invalid = find_invalid_citations_in_text(sent)
            if sent_invalid:
                for _, _, doc_id, page, reason in sent_invalid:
                    removed.append({
                        'doc_id': doc_id,
                        'page': page,
                        'reason': reason,
                        'sentence': sent[:100] + '...' if len(sent) > 100 else sent
                    })
            else:
                kept_sentences.append(sent)
        if kept_sentences:
            cleaned_lines.append(' '.join(kept_sentences))

    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = re.sub(r'  +', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\n\n+', '\n\n', cleaned_text)
    return cleaned_text, removed


def _remove_style_violations(text: str) -> tuple:
    removed = []
    cleaned_lines = []
    for line in text.split('\n'):
        if not _GENERIC_STYLE_RE.search(line):
            cleaned_lines.append(line)
            continue

        kept = []
        for sent in _split_sentences_for_cleanup(line):
            if _GENERIC_STYLE_RE.search(sent):
                removed.append(_clip(sent, n=180))
            else:
                kept.append(sent)
        if kept:
            cleaned_lines.append(' '.join(kept))

    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = re.sub(r'  +', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\n\n+', '\n\n', cleaned_text)
    return cleaned_text.strip(), removed


# v12: detect meta-commentary about the review/paper itself. The closing
# section of v11.2 leaked sentences like "The literature reviewed here
# converges on...". Catch-and-strip rather than catch-and-warn because these
# are pure noise — the substantive claim sits in the next sentence.
_META_COMMENTARY_RE = re.compile(
    r"\b("
    r"the literature reviewed (?:here|above)?"
    r"|this (?:review|paper|essay|article|analysis|discussion|investigation)"
    r"|the (?:sources|works|studies|papers) (?:cited|reviewed|discussed|surveyed)"
    r"|as (?:discussed|noted|shown|argued|established) (?:above|earlier)"
    r"|in (?:this|the (?:foregoing|preceding)) (?:review|paper|essay|article|section|discussion)"
    r"|we (?:will|shall) (?:examine|explore|investigate|consider|argue|show|discuss|analyse|analyze)"
    r"|this (?:section|paragraph) (?:examines|explores|investigates|considers|argues|shows|discusses)"
    # v13: "the thesis" used as self-reference to the review's own argument
    # ("important qualification to the thesis", "applies the thesis to",
    # "illustrates how the thesis can be applied"). Two patterns:
    # - "the thesis" preceded by a self-referential preposition or noun
    # - "this is an important [noun] to the thesis" — the v12 audit's rule 6
    #   trailing-significance miss is really meta-commentary about the review.
    r"|(?:applies|illustrates how|qualification(?:s)? to|adds nuance to|in support of|against) the thesis"
    r"|this is (?:an? )?important (?:qualification|implication|consequence|insight|point) (?:to|for|of) the thesis"
    r")\b",
    re.IGNORECASE,
)


def _strip_meta_commentary(text: str) -> tuple:
    """v12: drop sentences that talk about the review itself rather than the
    substantive question. Returns (cleaned_text, list_of_removed_snippets)."""
    if not text:
        return text, []
    removed = []
    cleaned_paragraphs = []
    for para in text.split("\n\n"):
        kept_sents = []
        for sent in _split_sentences_for_cleanup(para):
            if _META_COMMENTARY_RE.search(sent):
                removed.append(_clip(sent, n=200))
            else:
                kept_sents.append(sent)
        if kept_sents:
            cleaned_paragraphs.append(" ".join(kept_sents))
    cleaned = "\n\n".join(cleaned_paragraphs)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(), removed


def _strip_orphaned_citations(text: str) -> str:
    # Remove lines that are ONLY a citation.
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\([A-Za-z0-9_&.\-]+:\s*p\.\d+\)$', stripped):
            continue
        if re.match(r'^\([A-Za-z0-9_&]+_\d{4}[a-z]?\)$', stripped):
            continue
        if re.match(r'^\([A-Za-z]+\s+et\s+al\.?,?\s*\d{4}\)$', stripped):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


def _strip_continuation_markers(text: str) -> str:
    # Remove to be continued and similar markers.
    text = re.sub(r'(?im)^\s*Coverage repair:\s*$', '', text)
    text = re.sub(r'(?i)\bThe previous draft failed the citation coverage rule\.\s*', '', text)
    text = re.sub(r'(?i)\bWrite the section again using only the allowed citations above\.\s*', '', text)
    text = re.sub(r'(?im)^\s*(Requirements|Previous draft|Rewrite):\s*$', '', text)
    patterns = [
        r'\.\.\.?\s*to be continued.*?\n*',
        r'\(to be continued.*?\)',
        r'The next section will.*?\.',
        r'In the next section.*?\.',
        r'we will delve deeper.*?\.',
        r'\.\.\.to be continued in the next section\.',
        r'Continued in next message\.\.\.?\s*',
        r'The discussion will continue.*?\.',
        r'In the following section[s,]*\s*$',
        r'This review will continue.*?\.',
        r'As we continue our exploration.*?\.',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _strip_tail_after(text: str, patterns, min_frac: float = 0.6) -> str:
    """v15.14: shared tail-stripper for conclusion/references removal.

    The old passes used DOTALL '.*$' with no position guard, so a single
    mid-essay "In summary," paragraph or an inline "References:" line
    deleted EVERYTHING after it — up to half the essay. Guard: only strip
    when the matched marker begins in the final (1 - min_frac) of the text,
    where a genuine closing/bibliography actually lives. Earlier matches
    are left alone (the meta-commentary and zero-cite passes still police
    mid-essay filler on their own terms).
    """
    for pattern in patterns:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            if m.start() >= len(text) * min_frac:
                text = text[:m.start()]
                break
    return text.strip()


def _strip_conclusion(text: str) -> str:
    # Remove conclusion paragraphs (tail-guarded since v15.14).
    return _strip_tail_after(text, [
        r'\n\s*In conclusion[,.].*$',
        r'\n\s*To conclude[,.].*$',
        r'\n\s*In summary[,.].*$',
    ])


def _strip_references_section(text: str) -> str:
    # Remove formal References/Bibliography sections (tail-guarded since v15.14).
    return _strip_tail_after(text, [
        r'\n\s*References\s*:?\s*\n.*$',
        r'\n\s*Bibliography\s*:?\s*\n.*$',
        r'\n\s*Works Cited\s*:?\s*\n.*$',
        r'\n\s*\(References:.*?\).*$',
    ])


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _writer_enforcement_enabled() -> bool:
    return _ENFORCE_COVERAGE and os.environ.get("RRR_BYPASS_VALIDATION", "0") != "1"


def _writer_parallel_workers(n_chunks: int) -> int:
    # v12: default flipped from "1" to "0". Sequential writing with the v11.2
    # claims-so-far context produces tighter cross-section coherence (0 twin
    # openers vs 3 in parallel) at minimal runtime cost (+10-15s). Set
    # RRR_WRITER_PARALLEL=1 to opt back into parallel.
    if n_chunks <= 1 or os.environ.get("RRR_WRITER_PARALLEL", "0") == "0":
        return 1
    raw = os.environ.get("RRR_WRITER_PARALLELISM") or os.environ.get("RRR_CONCURRENCY") or "2"
    try:
        workers = int(raw)
    except Exception:
        workers = 1
    return max(1, min(workers, n_chunks))


def _strict_cited_doc_ids(text: str, allowed_pairs=None, display_lookup=None) -> set:
    # v15.7: parse_citations now iterates all three surfaces. Pass
    # display_lookup so display-form cites resolve to a doc_id (without
    # it they yield doc_id=None and are invisible to the coverage audit
    # — exactly the v15.5 silent-failure mode that produced the writer
    # collapse). When allowed_pairs is empty all resolved cites are
    # counted; otherwise only (doc, page) pairs in allowed_pairs.
    allowed_pairs = set(allowed_pairs or [])
    cited = set()
    for c in parse_citations(text, display_lookup=display_lookup):
        if c["doc_id"] is None:
            continue
        pair = (c["doc_id"], c["page"])
        if not allowed_pairs or pair in allowed_pairs:
            cited.add(c["doc_id"])
    return cited


def _coverage_requirement(chunk_docs, section_kind: str) -> int:
    n_docs = len([d for d in chunk_docs if d.get("doc_id")])
    if n_docs <= 0:
        return 0
    if section_kind == "closing":
        return min(2, n_docs)
    return min(max(1, _MIN_SECTION_CITED_DOCS), n_docs)


def _audit_section_coverage(text: str, chunk_docs, section_kind: str):
    chunk_pairs, chunk_allowed_docs, _ = _build_allowed_citations(chunk_docs)
    # v15.7: pass a display_lookup built from chunk_allowed_docs so the
    # audit can resolve display-form cites natively (no canonical
    # conversion needed).
    chunk_display_lookup_local = _build_display_lookup(chunk_allowed_docs)
    cited = _strict_cited_doc_ids(
        text, allowed_pairs=chunk_pairs, display_lookup=chunk_display_lookup_local,
    )
    valid_citation_pairs = [
        (citation.get("doc_id"), int(citation.get("page", 0) or 0))
        for citation in parse_citations(
            text, display_lookup=chunk_display_lookup_local,
        )
        if (
            citation.get("doc_id") in chunk_allowed_docs
            and (
                citation.get("doc_id"),
                int(citation.get("page", 0) or 0),
            ) in chunk_pairs
        )
    ]
    pair_counts = Counter(valid_citation_pairs)
    repeated_pairs = [
        {
            "doc_id": doc_id,
            "page": page,
            "occurrences": count,
        }
        for (doc_id, page), count in sorted(pair_counts.items())
        if count > _MAX_CLOSING_IDENTITY_REPETITIONS
    ]
    closing_citation_repetition_ok = (
        section_kind != "closing" or not repeated_pairs
    )
    required = _coverage_requirement(chunk_docs, section_kind)
    structure = _audit_section_structure(text, section_kind)
    uncited_paragraphs = []
    for index, paragraph in enumerate(_prose_paragraphs(text)):
        paragraph_cited = _strict_cited_doc_ids(
            paragraph,
            allowed_pairs=chunk_pairs,
            display_lookup=chunk_display_lookup_local,
        )
        if not paragraph_cited:
            uncited_paragraphs.append(index)
    uncited_closing_paragraphs = (
        list(uncited_paragraphs) if section_kind == "closing" else []
    )
    coverage_ok = len(cited) >= required
    paragraph_citations_ok = not uncited_paragraphs
    closing_paragraph_citations_ok = not uncited_closing_paragraphs
    return {
        "section": section_kind,
        "required_cited_docs": required,
        "cited_doc_count": len(cited),
        "cited_docs": sorted(cited),
        "provided_doc_count": len(chunk_allowed_docs),
        "coverage_ok": coverage_ok,
        "paragraph_count": structure["paragraph_count"],
        "word_count": structure["word_count"],
        "incomplete_paragraph_count": structure["incomplete_paragraph_count"],
        "incomplete_paragraphs": structure["incomplete_paragraphs"],
        "closing_word_floor": structure["closing_word_floor"],
        "closing_word_floor_ok": structure["closing_word_floor_ok"],
        "closing_word_ceiling": structure["closing_word_ceiling"],
        "closing_word_ceiling_ok": structure["closing_word_ceiling_ok"],
        "closing_paragraph_count_ok": structure[
            "closing_paragraph_count_ok"
        ],
        "closing_citation_occurrences": (
            len(valid_citation_pairs) if section_kind == "closing" else 0
        ),
        "closing_unique_citation_pairs": (
            len(pair_counts) if section_kind == "closing" else 0
        ),
        "closing_repeated_citation_pairs": (
            repeated_pairs if section_kind == "closing" else []
        ),
        "closing_citation_repetition_ok": (
            closing_citation_repetition_ok
        ),
        "uncited_paragraphs": uncited_paragraphs,
        "paragraph_citations_ok": paragraph_citations_ok,
        "uncited_closing_paragraphs": uncited_closing_paragraphs,
        "closing_paragraph_citations_ok": closing_paragraph_citations_ok,
        "structure_ok": structure["ok"],
        "ok": (
            coverage_ok
            and structure["ok"]
            and paragraph_citations_ok
            and closing_paragraph_citations_ok
            and closing_citation_repetition_ok
        ),
    }


def _citation_surface_fault_count(render_stats: dict) -> int:
    return (
        int(render_stats.get("separated_parenthetical_groups", 0) or 0)
        + int(render_stats.get("redundant_author_parentheticals", 0) or 0)
        + int(render_stats.get("orphan_parenthetical_groups", 0) or 0)
    )


def _hard_candidate_fault_count(render_stats: dict) -> int:
    """Count integrity faults that prevent a section from being frozen."""
    return sum(
        int(render_stats.get(key, 0) or 0)
        for key in (
            "unknown_eids",
            "attribution_mismatches",
            "raw_citation_faults",
            "invalid_citations",
            "quote_faults",
        )
    ) + _citation_surface_fault_count(render_stats)


def _section_candidate_passes(audit: dict, render_stats: dict,
                              section_kind: str = "closing",
                              structural_only: bool = False) -> bool:
    """Apply the same hard integrity contract to every candidate section."""
    if structural_only:
        return bool(audit.get("structure_ok"))
    return bool(
        audit.get("ok")
        and _hard_candidate_fault_count(render_stats) == 0
    )


def _section_quality_faults(audit: dict, render_stats: dict,
                            section_kind: str,
                            structural_only: bool = False) -> list:
    """Describe every detected fault for a clean-room repair prompt."""
    faults = []
    if int(audit.get("paragraph_count", 0) or 0) == 0:
        faults.append("The output contained no prose paragraph.")
    incomplete = int(audit.get("incomplete_paragraph_count", 0) or 0)
    if incomplete:
        faults.append(
            f"{incomplete} paragraph(s) ended without a complete sentence."
        )
    if section_kind == "closing" and not audit.get(
        "closing_paragraph_count_ok", True
    ):
        faults.append(
            "The closing must contain exactly one paragraph; the output "
            f"contained {int(audit.get('paragraph_count', 0) or 0)}."
        )
    if section_kind == "closing" and not audit.get(
        "closing_word_floor_ok", True
    ):
        faults.append(
            "The closing was below the minimum length: "
            f"{int(audit.get('word_count', 0) or 0)} words supplied, "
            f"{int(audit.get('closing_word_floor', 0) or 0)} required."
        )
    if section_kind == "closing" and not audit.get(
        "closing_word_ceiling_ok", True
    ):
        faults.append(
            "The closing exceeded the maximum length: "
            f"{int(audit.get('word_count', 0) or 0)} words supplied, "
            f"{int(audit.get('closing_word_ceiling', 0) or 0)} allowed."
        )
    if section_kind == "closing" and not audit.get(
        "closing_citation_repetition_ok", True
    ):
        repeated = audit.get("closing_repeated_citation_pairs") or []
        faults.append(
            "The closing repeated "
            f"{len(repeated)} document-page citation identity or identities. "
            "Cite each allowed evidence marker at most once."
        )
    if not structural_only and not audit.get("coverage_ok", False):
        faults.append(
            "Citation coverage was insufficient: "
            f"{int(audit.get('cited_doc_count', 0) or 0)} distinct papers "
            f"cited, {int(audit.get('required_cited_docs', 0) or 0)} required."
        )
    uncited = (
        [] if structural_only
        else list(audit.get("uncited_closing_paragraphs") or [])
    )
    if uncited:
        faults.append(
            "Closing paragraph(s) without a valid citation: "
            + ", ".join(str(index + 1) for index in uncited) + "."
        )
    unknown = 0 if structural_only else int(
        render_stats.get("unknown_eids", 0) or 0
    )
    if unknown:
        faults.append(
            f"The output used {unknown} unknown or malformed evidence ID(s)."
        )
    mismatches = 0 if structural_only else int(
        render_stats.get("attribution_mismatches", 0) or 0
    )
    if mismatches:
        faults.append(
            f"The output contained {mismatches} author/evidence attribution "
            "mismatch(es)."
        )
    raw_citation_faults = 0 if structural_only else int(
        render_stats.get("raw_citation_faults", 0) or 0
    )
    if raw_citation_faults:
        faults.append(
            f"The output used {raw_citation_faults} citation surface(s) outside "
            "the [E####]-only contract."
        )
    invalid_citations = 0 if structural_only else int(
        render_stats.get("invalid_citations", 0) or 0
    )
    if invalid_citations:
        faults.append(
            f"The rendered output contained {invalid_citations} citation(s) "
            "outside the call-local document and page set."
        )
    quote_faults = 0 if structural_only else int(
        render_stats.get("quote_faults", 0) or 0
    )
    if quote_faults:
        faults.append(
            f"The output contained {quote_faults} quoted span(s) that could "
            "not be verified against the cited page."
        )
    separated_groups = 0 if structural_only else int(
        render_stats.get("separated_parenthetical_groups", 0) or 0
    )
    if separated_groups:
        faults.append(
            f"The output contained {separated_groups} separated parenthetical "
            "citation group(s). Group supporting [E####] markers with spaces "
            "only and leave them unwrapped."
        )
    redundant_author_cites = 0 if structural_only else int(
        render_stats.get("redundant_author_parentheticals", 0) or 0
    )
    if redundant_author_cites:
        faults.append(
            f"The output named an author and then repeated that author in "
            f"{redundant_author_cites} full parenthetical citation(s). Put the "
            "evidence marker directly after a narratively named author."
        )
    orphan_parentheticals = 0 if structural_only else int(
        render_stats.get("orphan_parenthetical_groups", 0) or 0
    )
    if orphan_parentheticals:
        faults.append(
            f"The output placed {orphan_parentheticals} parenthetical citation "
            "group(s) after completed sentences or at paragraph openings. Put "
            "the evidence markers before the sentence's terminal punctuation."
        )
    uncited_paragraphs = (
        [] if structural_only else list(audit.get("uncited_paragraphs") or [])
    )
    if uncited_paragraphs:
        faults.append(
            "Paragraph(s) without a valid call-local citation: "
            + ", ".join(str(index + 1) for index in uncited_paragraphs)
            + "."
        )
    return faults or ["The output failed the section quality contract."]


def _section_retry_reason(audit: dict, render_stats: dict,
                          section_kind: str,
                          structural_only: bool = False) -> str:
    if structural_only:
        return "structure"
    if int(render_stats.get("attribution_mismatches", 0) or 0):
        return "attribution"
    if any(
        int(render_stats.get(key, 0) or 0)
        for key in (
            "unknown_eids",
            "raw_citation_faults",
            "invalid_citations",
            "quote_faults",
        )
    ):
        return "evidence"
    if _citation_surface_fault_count(render_stats):
        return "citation_surface"
    if (
        section_kind == "closing"
        and not audit.get("closing_citation_repetition_ok", True)
    ):
        return "citation_surface"
    if (
        not audit.get("structure_ok", False)
        or (
            section_kind == "closing"
            and not audit.get("closing_paragraph_citations_ok", False)
        )
        or not audit.get("paragraph_citations_ok", False)
    ):
        return "structure"
    return "coverage"


def _section_retry_stage(stage: str, retry_reason: str, attempt: int,
                         section_kind: str) -> str:
    """Preserve historical stage names outside the two-retry closing path."""
    base = f"{stage}_{retry_reason}_retry"
    return f"{base}_{attempt}" if section_kind == "closing" else base


def _resolve_nonclosing_retry(prior_state: dict, retry_state: dict,
                               retry_reason: str,
                               section_kind: str = "stream"):
    """Accept only a retry that clears the complete hard contract."""
    del prior_state, retry_reason
    if section_kind == "closing":
        return retry_state, "unresolved"
    if _section_candidate_passes(
        retry_state["audit"],
        retry_state["render_stats"],
        section_kind,
    ):
        accepted = dict(retry_state)
        accepted["_nonclosing_retry_accepted"] = True
        return accepted, "retry_accepted"
    return retry_state, "unresolved"


def _run_bounded_section_retries(initial_state: dict, section_kind: str,
                                   retry_candidate,
                                   structural_only: bool = False):
    """Run at most one repair call after the initial section candidate."""
    state = initial_state
    if _section_candidate_passes(
        state["audit"], state["render_stats"], section_kind,
        structural_only,
    ):
        return state, 0, True
    max_retries = (
        _MAX_CLOSING_MODEL_CALLS - 1
        if section_kind == "closing"
        else 1
    )
    for attempt in range(1, max_retries + 1):
        state = retry_candidate(state, attempt)
        if _section_candidate_passes(
            state["audit"], state["render_stats"], section_kind,
            structural_only,
        ):
            return state, attempt, True
    return state, max_retries, False


_FROZEN_SECTION_SCHEMA_VERSION = "accepted-sections-v1"
_BODY_CHECKPOINT_SCHEMA_VERSION = "writer-body-checkpoint-v1"


def _text_sha256(text: str) -> str:
    """Return the SHA-256 identity of a section's exact UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_sha256(value) -> str:
    """Hash a JSON-compatible value with deterministic encoding."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _audit_frozen_section_identity(assembled_text: str, sections,
                                   section_spans) -> dict:
    """Verify that assembly retained every accepted section byte-for-byte."""
    frozen = tuple(sections)
    spans = list(section_spans)
    records = []
    ok = len(frozen) == len(spans)

    for index, accepted in enumerate(frozen):
        if index >= len(spans):
            records.append({
                "index": index,
                "ok": False,
                "reason": "missing_span",
                "accepted_sha256": _text_sha256(accepted),
            })
            continue

        span = spans[index]
        start = int(span.get("start", -1))
        end = int(span.get("end", -1))
        bounds_ok = 0 <= start <= end <= len(assembled_text)
        assembled_section = (
            assembled_text[start:end] if bounds_ok else ""
        )
        accepted_sha256 = _text_sha256(accepted)
        assembled_sha256 = _text_sha256(assembled_section)
        record_ok = (
            bounds_ok
            and assembled_section == accepted
            and accepted_sha256 == assembled_sha256
        )
        records.append({
            "index": index,
            "kind": span.get("kind", "section"),
            "start": start,
            "end": end,
            "utf8_bytes": len(accepted.encode("utf-8")),
            "accepted_sha256": accepted_sha256,
            "assembled_sha256": assembled_sha256,
            "ok": record_ok,
            "reason": "ok" if record_ok else (
                "content_changed" if bounds_ok else "invalid_span"
            ),
        })
        ok = ok and record_ok

    if len(spans) > len(frozen):
        ok = False
        for index in range(len(frozen), len(spans)):
            records.append({
                "index": index,
                "ok": False,
                "reason": "unexpected_span",
            })

    return {
        "schema_version": _FROZEN_SECTION_SCHEMA_VERSION,
        "section_count": len(frozen),
        "span_count": len(spans),
        "sections": records,
        "ok": bool(ok),
    }


def _assemble_frozen_sections(sections, closing_index=None) -> tuple:
    """Join accepted sections while retaining their exact byte identity.

    The private closing marker belongs to assembly metadata. It is inserted
    immediately before the accepted closing text and is excluded from that
    section's recorded span.
    """
    frozen = tuple(sections)
    if not frozen:
        raise ValueError("at least one accepted section is required")
    if any(not isinstance(section, str) or not section for section in frozen):
        raise ValueError("accepted sections must be non-empty strings")
    if any(_CLOSING_START_MARKER in section for section in frozen):
        raise ValueError("accepted section contains the private closing marker")
    if closing_index is not None and not (
        isinstance(closing_index, int)
        and 0 <= closing_index < len(frozen)
    ):
        raise ValueError("closing_index is outside the accepted section list")

    pieces = []
    spans = []
    cursor = 0
    for index, section in enumerate(frozen):
        separator = "" if index == 0 else "\n\n"
        pieces.append(separator)
        cursor += len(separator)

        kind = "closing" if index == closing_index else "section"
        if index == closing_index:
            marker_prefix = f"{_CLOSING_START_MARKER} "
            pieces.append(marker_prefix)
            cursor += len(marker_prefix)

        start = cursor
        pieces.append(section)
        cursor += len(section)
        spans.append({
            "index": index,
            "kind": kind,
            "start": start,
            "end": cursor,
        })

    assembled = "".join(pieces)
    audit = _audit_frozen_section_identity(assembled, frozen, spans)
    if not audit["ok"]:
        raise RuntimeError("accepted section identity changed during assembly")
    return assembled, audit


def _build_body_checkpoint_payload(
    sections,
    *,
    ledger_sha256: str,
    section_claims,
    section_coverage,
    call_contracts,
    diagnostics=None,
) -> tuple:
    """Serialize the accepted body before any closing call is attempted."""
    frozen = tuple(sections)
    body_text, freeze_audit = _assemble_frozen_sections(frozen)
    section_records = [
        {
            "index": index,
            "text": section,
            "sha256": _text_sha256(section),
            "utf8_bytes": len(section.encode("utf-8")),
        }
        for index, section in enumerate(frozen)
    ]
    payload = {
        "schema_version": _BODY_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_stage": "body_accepted_before_closing",
        "artifact_contract": _WRITER_ARTIFACT_CONTRACT,
        "execution_profile": _writer_execution_profile(),
        "citation_representation": _CITATION_REPRESENTATION_VERSION,
        "post_acceptance_mutation_policy": (
            _POST_ACCEPTANCE_MUTATION_POLICY
        ),
        "ledger_sha256": ledger_sha256,
        "body_sha256": _text_sha256(body_text),
        "body_utf8_bytes": len(body_text.encode("utf-8")),
        "body_section_count": len(frozen),
        "body_sections": section_records,
        "body_freeze": freeze_audit,
        "section_claims": list(section_claims or []),
        "section_coverage": list(section_coverage or []),
        "body_call_count": len(call_contracts or []),
        "body_call_contracts_sha256": _json_sha256(
            list(call_contracts or [])
        ),
        "diagnostics": dict(diagnostics or {}),
    }
    reconstructed = "\n\n".join(
        record["text"] for record in payload["body_sections"]
    )
    if reconstructed != body_text:
        raise RuntimeError("body checkpoint reconstruction changed accepted bytes")
    return payload, body_text, freeze_audit


def _split_marked_closing(text: str) -> dict:
    """Separate the explicitly marked closing from the assembled review."""
    source = text or ""
    marker_count = source.count(_CLOSING_START_MARKER)
    if marker_count != 1:
        return {
            "ok": False,
            "marker_count": marker_count,
            "marker_at_paragraph_start": False,
            "body": source,
            "closing": "",
            "without_marker": source,
        }
    marker_start = source.index(_CLOSING_START_MARKER)
    prefix = source[:marker_start]
    marker_at_paragraph_start = (
        not prefix.strip()
        or bool(re.search(r"\n[ \t]*\n[ \t]*$", prefix))
    )
    body = prefix.rstrip()
    closing = source[marker_start + len(_CLOSING_START_MARKER):].lstrip()
    if body and closing:
        without_marker = body + "\n\n" + closing
    else:
        without_marker = body or closing
    return {
        "ok": marker_at_paragraph_start and bool(closing),
        "marker_count": marker_count,
        "marker_at_paragraph_start": marker_at_paragraph_start,
        "body": body,
        "closing": closing,
        "without_marker": without_marker.strip(),
    }


def _audit_final_structure(text: str, closing_docs,
                           enforce_citation_integrity: bool = True) -> dict:
    """Validate every shipped paragraph and the identified closing section."""
    split = _split_marked_closing(text)
    whole = _audit_section_structure(split["without_marker"], "full_review")
    closing_audit = _audit_section_coverage(
        split["closing"], closing_docs, "closing",
    )
    closing_ok = (
        closing_audit["ok"]
        if enforce_citation_integrity
        else closing_audit["structure_ok"]
    )
    return {
        "ok": bool(split["ok"] and whole["ok"] and closing_ok),
        "citation_integrity_enforced": enforce_citation_integrity,
        "closing_marker_count": split["marker_count"],
        "closing_marker_at_paragraph_start": split["marker_at_paragraph_start"],
        "paragraph_count": whole["paragraph_count"],
        "incomplete_paragraph_count": whole["incomplete_paragraph_count"],
        "incomplete_paragraphs": whole["incomplete_paragraphs"],
        "closing": closing_audit,
        "text_without_marker": split["without_marker"],
    }


# Legacy coverage-patch prose is prohibited by the v19 acceptance contract.
_COVERAGE_PATCH_SENTINEL = "[[COV-PATCH]] "


def _coverage_retry_prompt(original_prompt: str, prior_chunk: str,
                           required_docs: int, section_kind: str = "",
                           retry_reason: str = "coverage",
                           faults=None) -> str:
    # v15.7: retry prompt now demands [E####] markers consistent with
    # _SYSTEM_CITATION_INSTRUCTION. The previous version forbade
    # author-year citations and demanded canonical '(DocId: p.N)' — both
    # contradicting the system prompt's '[E####] only' rule. The new
    # version restates the marker contract and asks for at least N
    # distinct DOCS-worth of markers (the audit checks doc coverage, not
    # marker count).
    reason_text = {
        "structure": (
            "The previous output contained an incomplete paragraph, an invalid "
            "paragraph boundary, or an undersized closing."
        ),
        "attribution": (
            "The previous output attached an evidence marker to an incompatible "
            "author attribution."
        ),
        "coverage": (
            "The previous output did not cite enough distinct papers from the "
            "allowed evidence."
        ),
        "evidence": (
            "The previous output used an unknown or malformed evidence ID."
        ),
        "citation_surface": (
            "The previous output used a redundant or incorrectly grouped "
            "citation surface."
        ),
    }.get(retry_reason, "The previous output failed a section quality rule.")
    fault_lines = "\n".join(
        f"- {fault}" for fault in (faults or [reason_text])
    )
    del prior_chunk
    clean_room = section_kind == "closing"
    prior_block = ""
    repair_mode = (
        "Generate a fresh closing from the allowed evidence above. Do not reuse "
        "the rejected output. Return exactly one uninterrupted paragraph, "
        "with no blank lines or paragraph breaks."
        if clean_room else
        "Generate a fresh section from the allowed evidence above. Do not reuse "
        "the rejected output."
    )
    return f"""{original_prompt}

Section repair:
The previous output failed the checks listed below:
{fault_lines}

{repair_mode} Use only [E####] markers drawn from the ALLOWED CITATIONS list at the top of this prompt.

Requirements:
- Cite at least {required_docs} DIFFERENT papers when that many are listed. (One paper can contribute multiple markers; the count is of distinct papers.)
- Every paragraph must include at least one [E####] marker.
- Put markers before the sentence's terminal punctuation. Never place markers after a completed sentence or at a paragraph opening.
- For a multi-source claim, copy two or more allowed markers and place them next to one another with spaces only. Do not put parentheses or semicolons around evidence markers.
- Keep the prose claim-led. If an author must be the grammatical subject, put that paper's allowed marker directly after the author name and before the verb. When the source supports the whole sentence, use its allowed marker without repeating the author in prose.
- Use ONLY [E####] markers from the ALLOWED CITATIONS list. Copy every allowed marker character-for-character. Do not invent IDs. Do not write '(Author Year, p.N)', 'Author (Year, p.N)', '(DocId: p.N)' or any other citation surface — the renderer expands every [E####] into the correct surface.
- Preserve the same substantive role and word range.
- Paraphrase the evidence. Do not use quotation marks in the repair.
- Finish every paragraph with a complete standalone sentence and terminal punctuation.
{f'- Aim for 140-180 words. The hard accepted range is {_MIN_CLOSING_WORDS}-{_MAX_CLOSING_WORDS} words. Use 2-3 citation groups total and cite each evidence marker at most once.' if section_kind == 'closing' else ''}
{prior_block}

{_OUTPUT_ONLY_DIRECTIVE}

Rewrite:"""


def _build_allowed_citations(docs):
    allowed_pairs = set()
    allowed_docs = set()
    allowed_pages_by_doc = defaultdict(set)
    for d in docs:
        did = str(d.get("doc_id", "")).strip()
        if not did:
            continue
        allowed_docs.add(did)
        for q in d.get("quotes", []) or []:
            qdid = str(q.get("doc_id", did)).strip() or did
            try:
                pg = int(q.get("page", 0) or 0)
            except Exception:
                pg = 0
            if qdid and pg > 0:
                allowed_pairs.add((qdid, pg))
                allowed_pages_by_doc[qdid].add(pg)
    return allowed_pairs, allowed_docs, allowed_pages_by_doc


# =============================================================================
# v7: LEANER PROMPTS - Claims about phenomena, not scholars
# =============================================================================

_PROSE_DIRECTIVE = (
    # v14.2.1 simplification: trimmed from ~190 words to ~80. Citation-format
    # material moved to _SYSTEM_CITATION_INSTRUCTION.
    # v14.4: reframed as a "literature review organised around streams of
    # related scholarship" — each section now presents one stream of work
    # and synthesises across the sources within it. The per-section builders
    # match the framing ("present this stream", "this stream challenges...",
    # "this stream introduces..."). Streams are a more natural unit for
    # literature-review prose than the older "chain of inference" metaphor.
    "Write a scholarly literature review organised around streams of "
    "related scholarship. Each section presents ONE stream and synthesises "
    "across its sources. Use historical phenomena, mechanisms, and conditions "
    "as grammatical subjects. Keep author-led sentences rare; whenever an "
    "author is named, place that paper's allowed evidence marker immediately "
    "after the name and before the verb. Argue in direct positive claims; avoid contrastive "
    "framings such as 'not X but Y', 'rather than', 'unlike', or 'in "
    "contrast'. Vary sentence length; introduce claims as full sentences "
    "rather than via colon setups. "
    # v12 multi-doc rule.
    "Each paragraph must integrate at least two DISTINCT documents; do not "
    "cite the same document for more than two consecutive citations. "
    "Finish every paragraph and section with a complete standalone sentence. "
    "Create continuity by carrying a substantive theme across sections. "
    # v12 anti-meta + topic restate.
    "Make claims directly about the substantive question. Do NOT write meta-"
    "statements about the review itself ('this review', 'the literature "
    "reviewed here', 'as discussed above', 'we will examine'). "
    "Do not restate the topic or the thesis."
)

# v8 (R4): single canonical set of prompt builders. The previous file had two
# definitions per builder — the first was dead code shadowed by the second.
# Each stance builder now has a structurally DISTINCT instruction so the
# resulting paragraphs read as substantively different arguments rather than
# three parallel templates with one verb swapped.

def _prev_tail_block(previous_tail: str) -> str:
    """v14.2 cond D: render the 'Previous ending: …' block with an optional
    strong directive (RRR_WRITER_PREV_TAIL_STRONG=1) that explicitly tells the
    writer NOT to restate the topic and instead continue from the tail. Returns
    the formatted block; safe to include even when previous_tail is empty (the
    opening section's first call has no prior chunk)."""
    if not previous_tail:
        return ""
    block = f"Previous ending:\n...{previous_tail}\n"
    # v14.2 default flipped 0 -> 1: the v14.2 cond D smoke showed the strong
    # directive cut mean seam count from 3.3 (cond A) to 2.3 with zero LLM
    # cost. Set RRR_WRITER_PREV_TAIL_STRONG=0 to reproduce v14.1 prompt shape.
    if os.environ.get("RRR_WRITER_PREV_TAIL_STRONG", "1") == "1":
        block += (
            "\nIMPORTANT: The Previous ending is a complete sentence. Open "
            "with another complete, standalone sentence that develops one of "
            "its themes, findings, or scope conditions. Do not continue its "
            "grammar, and do not restate the topic on its own.\n"
        )
    return block


def _closing_continuity_block(previous_tail: str) -> str:
    """Provide thematic continuity without expanding citation authority."""
    if not previous_tail:
        return ""
    return (
        "CONTINUITY CONTEXT ONLY. The excerpt below grants no citation or "
        "evidentiary authority. Do not copy its citations. Ground every closing "
        "claim only in the ALLOWED CITATIONS and CALL-LOCAL EVIDENCE that "
        "follow. Its sole purpose is to help the closing continue the prior "
        "section's theme.\n"
        f"Previous ending:\n...{previous_tail}\n"
    )


def _build_opening_prompt(topic: str, stance_summary: str, evidence: str, allowed_list: str):
    return f"""Literature review on: {topic}

{stance_summary}

ALLOWED CITATIONS:
{allowed_list}

Evidence:
{evidence}

{_PROSE_DIRECTIVE}

Open the review. State the substantive question and what is at stake intellectually (not in NGO terms). Define the live disagreement through its competing mechanisms, historical cases, and scope conditions without listing papers as a survey. Keep the exposition claim-led. 200-300 words. End with a complete standalone sentence whose substantive theme can carry into the next section.

{_OUTPUT_ONLY_DIRECTIVE}

Begin:"""


# v15.4.0 (Bug 4 half-fix): extract author surnames the writer has ALREADY
# named in prior chunk prose, so the next chunk's prompt can explicitly
# forbid re-introducing them with a fresh "X argues..." sentence.
# Pulls surnames from both display narrative cites "Surname (Year" and
# canonical "(Doc_Year: p.N)" tokens (via _doc_id_to_author_label).
_DISPLAY_NAMED_AUTHOR_RE = re.compile(
    r"\b((?:(?:van|von|de|del|der)\s+)?[A-Z][A-Za-z\-]+"
    r"(?:\s+and\s+(?:(?:van|von|de|del|der)\s+)?[A-Z][A-Za-z\-]+"
    r"|\s+et\s+al\.)?)\s+\(\d{4}[a-z]?(?:,|\s)"
)


def _extract_named_surnames(chunk: str) -> list:
    """Return the list of unique author labels named in prose (narrative
    cites and canonical cites resolved to author labels). Order-preserving.
    """
    if not chunk:
        return []
    seen = []
    for m in _DISPLAY_NAMED_AUTHOR_RE.finditer(chunk):
        label = m.group(1).strip()
        # heuristic guard: drop ones that are clearly sentence-initial
        # English words ("The (2020)" etc.). Common short labels are fine.
        if label.lower() in {"the", "this", "that", "these", "those", "many", "some",
                             "such", "their", "they", "his", "her", "our", "your"}:
            continue
        if label not in seen:
            seen.append(label)
    for m in CITE_RE.finditer(chunk):
        did = m.group(1)
        label = _doc_id_to_author_label(did)
        # strip the trailing "(Year)" if present in the label
        label = re.sub(r"\s*\(\d{4}[a-z]?\)\s*$", "", label).strip()
        if label and label not in seen:
            seen.append(label)
    return seen


def _format_claims_so_far(section_claims) -> str:
    """v11.2 lever 3: render the running record of prior sections into a prompt
    block so the next writer call knows what claims have already been made.
    Returns empty string when there are no prior sections (the opening, the
    first stance section, or the parallel writer path where every section
    starts simultaneously).
    v15.4.0 (Bug 4 half-fix): also surface 'Already named in prose: ...' so
    the next chunk's prompt knows which authors have been introduced
    narratively and should not be re-introduced with a fresh 'X argues...'.
    """
    if not section_claims:
        return ""
    lines = [
        "CLAIMS ALREADY MADE IN PRIOR SECTIONS (do NOT repeat these openers or "
        "the same mechanism phrasing, and do NOT re-introduce an author already "
        "named in prose below with a fresh \"X argues / X shows...\" sentence — "
        "cite them parenthetically instead):"
    ]
    all_named: list = []
    for i, c in enumerate(section_claims, 1):
        mechs = "; ".join((c.get("mechanisms") or [])[:2]) or "(no mechanism recorded)"
        docs = ", ".join(str(d) for d in (c.get("docs") or [])[:4])
        stance = (c.get("stance") or "").upper()
        cluster = c.get("cluster") or ""
        lines.append(f"  Section {i} [{stance}/{cluster}]: {mechs} (citing {docs})")
        for n in (c.get("surnames_named") or []):
            if n not in all_named:
                all_named.append(n)
    if all_named:
        lines.append(f"  Already named in prose (cite parenthetically, do not "
                     f"re-introduce): {', '.join(all_named[:20])}")
    return "\n".join(lines)


# =============================================================================
# v15: single stream-section builder. The cluster's posture, expressed as
# free-text elaboration plus a structural relation, is the
# section's substantive claim; the writer never sees "supports / critiques /
# complicates" vocabulary in its instructions.
# =============================================================================

# Human-readable, topic-agnostic glosses for each per-shape relation enum
# value. Used by the section prompt to tell the writer HOW this stream stands
# to the topic — structurally, never with bucket vocabulary.
_RELATION_OPENERS = {
    # causal shape (6 values)
    "same_as_topic_cause":
        "this stream develops the topic's claim using the same causal "
        "variable under different phrasing or as a more specific instance",
    "upstream_of_topic_cause":
        "this stream identifies an ANTECEDENT that flows INTO the topic's "
        "causal variable as a trigger; the stream's account complements "
        "rather than competes with the topic's claim",
    "downstream_of_topic_cause":
        "this stream studies a DOWNSTREAM consequence of the topic's causal "
        "variable; the topic's claim is taken as given and the stream adds "
        "what follows from it",
    "rival_to_topic_cause":
        "this stream proposes a RIVAL cause that does NOT operate through "
        "the topic's variable and offers a genuinely competing explanation",
    "scope_condition":
        "this stream identifies WHEN the topic's claim holds and when it "
        "does not — a contingency or scope restriction on the topic's account",
    "adjacent":
        "this stream addresses a related but distinct question and engages "
        "the topic only obliquely",
    # comparative shape (5 values)
    "endorses_topic":
        "this stream AFFIRMS the topic's comparative judgment with "
        "convergent evidence",
    "reverses_topic":
        "this stream reaches the OPPOSITE comparative verdict and reads the "
        "evidence as supporting the inverse claim",
    "qualifies_topic":
        "this stream ACCEPTS the topic's comparison only under specific "
        "conditions and identifies when the comparison breaks down",
    "methodological_critique":
        "this stream challenges the WAY the comparison is constructed — "
        "the measurement, sample, or framing — without directly endorsing "
        "or reversing the topic's verdict",
    # descriptive shape (4 values)
    "confirms_description":
        "this stream's account is CONSISTENT with the topic's description "
        "and adds corroborating evidence for the same pattern",
    "contradicts_description":
        "this stream's account is INCOMPATIBLE with the topic's "
        "description and reports a different pattern in the same subject",
    "adds_nuance":
        "this stream ACCEPTS the topic's description but refines its scope, "
        "magnitude, or mechanism",
}


def _format_outline_block(cluster: dict, doc_to_evidence_ids) -> str:
    """Render a claims-first section brief from a planned cluster."""
    if not cluster or not isinstance(cluster, dict):
        return ""

    def _eids(doc_ids):
        out = []
        for did in doc_ids or []:
            ids = doc_to_evidence_ids.get(did) or []
            if ids:
                out.append(ids[0])
        return out

    doc_ids = list(cluster.get("doc_ids", []) or [])
    lead_doc_id = (cluster.get("lead_doc_id") or "").strip()
    if lead_doc_id in doc_ids:
        doc_ids = [lead_doc_id] + [did for did in doc_ids if did != lead_doc_id]
    eids = _eids(doc_ids)
    elaboration = (cluster.get("elaboration") or "").strip()
    disagreement = (cluster.get("internal_disagreement") or "").strip()
    if not elaboration:
        return ""

    lines = [
        "SECTION ARGUMENT BRIEF:",
        "Historical claim:",
        f"  {elaboration}",
        "Build the section around this claim, its mechanism, and its scope "
        "conditions. Use papers as evidence for the argument. Organize the "
        "prose by claims and connections, not as one sentence per author.",
    ]
    if len(eids) >= 2:
        cite_block = " ".join(f"[{e}]" for e in eids[:5])
        lines.append(
            "Evidence coverage target: support the central historical claim "
            f"once with these adjacent markers: {cite_block}"
        )
        lines.append(
            "Keep grouped markers unwrapped and separated only by spaces. "
            "The renderer will create one parenthetical citation."
        )
    elif eids:
        lines.append(
            "Evidence coverage target: ground the central historical claim "
            f"with [{eids[0]}]."
        )
    if disagreement:
        lines.extend([
            "Qualification or internal disagreement:",
            f"  {disagreement}",
        ])
    return "\n".join(lines)


def _build_stream_prompt(topic: str, cluster_label: str, evidence: str,
                         allowed_list: str, previous_tail: str,
                         outline_block: str = "",
                         claims_so_far: str = "",
                         relation: str = "",
                         topic_shape: str = ""):
    """v15: single section builder. Same arity as the legacy per-stance
    builders so the dispatcher loop in compose_from_ledger does not need to
    branch on shape. `relation` and `topic_shape` are passed through so the
    prompt can name the stream's structural relation to the topic — without
    any supports/critiques/complicates vocabulary.

    The section brief carries evidence coverage targets while leaving the
    sentence subjects and paragraph order to the claims being developed.
    """
    outline_section = (outline_block + "\n\n") if outline_block else ""
    prior_section = (claims_so_far + "\n\n") if claims_so_far else ""
    prev_block = _prev_tail_block(previous_tail)
    relation_gloss = _RELATION_OPENERS.get(relation or "", "")
    relation_line = (
        f"Structural relation of this stream to the topic ({relation or 'unspecified'}): "
        f"{relation_gloss}.\n"
        if relation_gloss else ""
    )
    return f"""Continue this literature review on: {topic}

{prev_block}
{prior_section}Stream: {cluster_label}.
{relation_line}
ALLOWED CITATIONS:
{allowed_list}

{outline_section}Evidence:
{evidence}

{_PROSE_DIRECTIVE}

Present this stream as an argument organized around historical claims. Open with the central claim in the SECTION ARGUMENT BRIEF. The opening sentence does not need an author as its grammatical subject. Never name the stream's structural relation in the prose (no 'this stream supports/critiques/complicates'); express the relation through the argument and the sources cited together. Meet the Evidence coverage target in the brief. Develop one specific mechanism, condition, or pattern that the sources jointly establish, then explain one implication or qualification. Connect each paragraph's opening claim to the preceding section or paragraph. 220-300 words. End with a complete standalone sentence whose substantive theme can carry into the next section.

{_OUTPUT_ONLY_DIRECTIVE}

Continue:"""


def _claim_without_citation_surfaces(claim: str) -> str:
    """Remove citation text and punctuation-only shells from a digest claim."""
    claim = str(claim or "")
    for start, end in reversed(_parenthetical_citation_containers(claim)):
        claim = claim[:start] + claim[end:]
    citations = list(parse_citations(claim))
    for citation in sorted(citations, key=lambda item: item["start"], reverse=True):
        claim = claim[:citation["start"]] + claim[citation["end"]:]
    claim = re.sub(r"\[[Ee]\d{1,5}\]", "", claim)
    claim = re.sub(r"\([;,\s]*\)", "", claim)
    claim = re.sub(r"\s+([,.;:!?])", r"\1", claim)
    claim = re.sub(r"\s{2,}", " ", claim).strip(" ;,")
    return claim


def _accepted_claim_excerpt(section_text: str, max_chars: int = 220) -> str:
    """Extract a compact claim from an accepted section without cite surfaces."""
    paragraphs = _prose_paragraphs(section_text)
    if not paragraphs:
        return ""
    sentences = _split_sentences_for_cleanup(paragraphs[0])
    claim = sentences[0].strip() if sentences else paragraphs[0].strip()
    claim = _claim_without_citation_surfaces(claim)
    return _clip(claim, n=max_chars)


def _body_citation_trace(body_chunks, docs) -> list:
    """Return valid body citations in source order with section provenance."""
    allowed_pairs, allowed_docs, _ = _build_allowed_citations(docs)
    display_lookup = _build_display_lookup(allowed_docs)
    trace = []
    for section_index, section in enumerate(body_chunks or []):
        for citation in parse_citations(
            section or "", display_lookup=display_lookup,
        ):
            doc_id = citation.get("doc_id")
            page = int(citation.get("page", 0) or 0)
            if doc_id not in allowed_docs or (doc_id, page) not in allowed_pairs:
                continue
            trace.append({
                "section_index": section_index,
                "doc_id": doc_id,
                "page": page,
                "start": int(citation.get("start", 0) or 0),
            })
    return trace


def _select_closing_evidence_packet(docs, body_chunks,
                                    max_docs: int = _MAX_CLOSING_DOCS) -> tuple:
    """Select a small packet from sources actually cited by the frozen body.

    Selection first takes one source from each accepted stream in section
    order, then fills remaining slots by cross-section reach, citation
    frequency, document score, and first appearance. One passage is retained
    per document, preferring the page most often cited in the accepted body.
    """
    max_docs = max(1, int(max_docs))
    trace = _body_citation_trace(body_chunks, docs)
    docs_by_id = {
        str(doc.get("doc_id", "") or "").strip(): doc
        for doc in docs or []
        if str(doc.get("doc_id", "") or "").strip()
    }
    doc_counts = defaultdict(int)
    doc_sections = defaultdict(set)
    doc_first_seen = {}
    page_counts = defaultdict(lambda: defaultdict(int))
    page_first_seen = {}
    section_docs = defaultdict(list)

    for position, record in enumerate(trace):
        doc_id = record["doc_id"]
        page = record["page"]
        section_index = record["section_index"]
        doc_counts[doc_id] += 1
        doc_sections[doc_id].add(section_index)
        doc_first_seen.setdefault(doc_id, position)
        page_counts[doc_id][page] += 1
        page_first_seen.setdefault((doc_id, page), position)
        if doc_id not in section_docs[section_index]:
            section_docs[section_index].append(doc_id)

    ranked_docs = sorted(
        doc_counts,
        key=lambda doc_id: (
            -len(doc_sections[doc_id]),
            -doc_counts[doc_id],
            -float(docs_by_id.get(doc_id, {}).get("avg_score", 0) or 0),
            doc_first_seen[doc_id],
            doc_id,
        ),
    )
    selected_ids = []

    # The opening is index 0. Streams carry the body argument, so sample them
    # first and use the opening when capacity remains.
    stream_indices = sorted(index for index in section_docs if index != 0)
    section_order = stream_indices + ([0] if 0 in section_docs else [])
    for section_index in section_order:
        for doc_id in section_docs[section_index]:
            if doc_id not in selected_ids:
                selected_ids.append(doc_id)
                break
        if len(selected_ids) >= max_docs:
            break
    for doc_id in ranked_docs:
        if len(selected_ids) >= max_docs:
            break
        if doc_id not in selected_ids:
            selected_ids.append(doc_id)

    selected_docs = []
    selected_pages = {}
    fallback_passages = 0
    for doc_id in selected_ids:
        source_doc = docs_by_id.get(doc_id)
        if not source_doc:
            continue
        ranked_pages = sorted(
            page_counts[doc_id],
            key=lambda page: (
                -page_counts[doc_id][page],
                page_first_seen[(doc_id, page)],
                page,
            ),
        )
        selected_quote = None
        for page in ranked_pages:
            selected_quote = next(
                (
                    quote for quote in (source_doc.get("quotes") or [])
                    if str(quote.get("doc_id", doc_id) or "").strip() == doc_id
                    and int(quote.get("page", 0) or 0) == page
                ),
                None,
            )
            if selected_quote:
                selected_pages[doc_id] = page
                break
        if selected_quote is None:
            quotes = list(source_doc.get("quotes") or [])
            if not quotes:
                continue
            selected_quote = quotes[0]
            selected_pages[doc_id] = int(
                selected_quote.get("page", 0) or 0
            )
            fallback_passages += 1
        selected_doc = dict(source_doc)
        selected_doc["quotes"] = [selected_quote]
        selected_docs.append(selected_doc)

    packet = _select_call_evidence(selected_docs, quotes_per_doc=1)
    stats = {
        "body_section_count": len(body_chunks or []),
        "body_citation_count": len(trace),
        "body_cited_doc_ids": ranked_docs,
        "body_cited_doc_count": len(ranked_docs),
        "selected_doc_ids": [
            str(doc.get("doc_id", "") or "").strip() for doc in packet
        ],
        "selected_doc_count": len(packet),
        "selected_pages": selected_pages,
        "selected_evidence_ids": sorted(_build_evidence_id_map(packet)),
        "passage_fallbacks": fallback_passages,
        "max_docs": max_docs,
    }
    return packet, stats


def _build_closing_digest(section_claims, body_chunks, closing_docs) -> str:
    """Render accepted claims and body-cited evidence into a compact digest."""
    selected_ids = {
        str(doc.get("doc_id", "") or "").strip()
        for doc in closing_docs or []
    }
    evidence_ids_by_doc = _group_evidence_ids_by_doc(closing_docs)
    allowed_pairs, allowed_docs, _ = _build_allowed_citations(closing_docs)
    display_lookup = _build_display_lookup(allowed_docs)
    lines = [
        "ACCEPTED BODY DIGEST. The body is final. Synthesize its claims without "
        "copying its sentences.",
    ]

    claims = list(section_claims or [])
    for index, record in enumerate(claims, 1):
        claim = str(record.get("accepted_claim", "") or "").strip()
        if not claim:
            for key in (
                "historical_claim", "claim", "posture", "elaboration",
            ):
                claim = str(record.get(key, "") or "").strip()
                if claim:
                    break
        if not claim:
            mechanisms = [
                str(value).strip()
                for value in (record.get("mechanisms") or [])
                if str(value).strip()
            ]
            claim = "; ".join(mechanisms[:2])
        claim = _claim_without_citation_surfaces(
            claim or str(record.get("cluster", "") or "")
        )
        claim = _clip(claim, n=220)
        claim = claim.rstrip(".!?")

        body_index = index
        cited_ids = []
        if body_index < len(body_chunks or []):
            for citation in parse_citations(
                body_chunks[body_index] or "",
                display_lookup=display_lookup,
            ):
                doc_id = citation.get("doc_id")
                page = int(citation.get("page", 0) or 0)
                if (
                    doc_id in selected_ids
                    and (doc_id, page) in allowed_pairs
                    and doc_id not in cited_ids
                ):
                    cited_ids.append(doc_id)
        markers = [
            f"[{evidence_ids_by_doc[doc_id][0]}]"
            for doc_id in cited_ids
            if evidence_ids_by_doc.get(doc_id)
        ]
        label = str(record.get("cluster", "") or f"stream {index}").strip()
        grounding = (
            " Grounding retained for the closing: " + " ".join(markers) + "."
            if markers else ""
        )
        lines.append(f"- Accepted stream {index}, {label}: {claim}.{grounding}")

    lines.append("SELECTED BODY-CITED EVIDENCE:")
    lines.extend(_format_doc_entry(doc) for doc in closing_docs or [])
    return "\n".join(lines)


def _build_closing_prompt(topic: str, digest: str, allowed_list: str,
                          previous_tail: str = ""):
    # v8 (R4/R6): closing must synthesise without restating prior section claims;
    # do not paraphrase the same two mechanisms that opened.
    # v13: the v12 smoke produced a methodology-stub closing ("This would
    # involve measuring..."), zero citations, no synthesis. The prompt now
    # forbids the future-research / methodology-stub register explicitly and
    # demands the closing cite at least two distinct documents from the
    # allowed list. The synthesis structure (agreement, disagreement,
    # resolution) is preserved but reframed as ACTIVE statements grounded in
    # the cited sources, not as hypothetical assessments of what would settle
    # the question.
    # The fourth argument remains for compatibility with older direct callers.
    # v19 derives continuity from accepted claims and does not expose a prose
    # tail that could be copied or treated as fresh evidentiary authority.
    del previous_tail
    return f"""Close this literature review on: {topic}

ALLOWED CITATIONS:
{allowed_list}

{digest}

Write exactly one continuous, claim-led synthesis paragraph. Aim for 140-180
words. The hard accepted range is {_MIN_CLOSING_WORDS}-{_MAX_CLOSING_WORDS}
words.

Begin with the underlying point of agreement. State the precise remaining
disagreement. Finish with the specific historical case, comparison, or
measurement already present in the selected evidence that distinguishes the
competing explanations.

Use 2-3 citation groups total and cite at least two distinct documents. Cite
each [E####] marker at most once. When several sources support one move, place
their markers next to one another before the terminal punctuation. Ground the
three analytic moves, not every sentence.

Use direct present-tense claims about historical phenomena, mechanisms, and
conditions. Do not write future-research or methodology-stub prose. Do not
write "In conclusion" or "To summarize". Return one complete paragraph with
no heading, blank line, or commentary.

{_OUTPUT_ONLY_DIRECTIVE}

Continue:"""


def _dump_writer_prompt(stage: str, system: str, user: str):
    """v10: when RRR_DEBUG_WRITER_PROMPTS=1 (or an explicit dir), write the
    exact system+user pair sent to the model into runs/prompts/. One file per
    stage call; later calls of the same stage suffix with a counter.
    """
    target = os.environ.get("RRR_WRITER_PROMPT_DUMP_DIR")
    if not target and os.environ.get("RRR_DEBUG_WRITER_PROMPTS", "0") == "1":
        target = str(runs_path("prompts"))
    if not target:
        return
    try:
        os.makedirs(target, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", stage)
        # disambiguate repeated stages (e.g. writer_complicates fires N times)
        idx = 0
        while True:
            fn = os.path.join(target, f"{slug}_{idx:02d}.txt")
            if not os.path.exists(fn):
                break
            idx += 1
        payload = (
            "===== SYSTEM =====\n" + (system or "") + "\n\n"
            "===== USER =====\n" + (user or "") + "\n"
        )
        with open(fn, "w", encoding="utf-8") as f:
            f.write(payload)
        return fn
    except Exception:
        # debug dump must never break a writer call
        return None


def _dump_writer_response(prompt_path, response: str = "", error=None) -> None:
    """Store the exact response paired with a dumped writer prompt."""
    if not prompt_path:
        return
    try:
        response_path = os.path.splitext(prompt_path)[0] + "_response.txt"
        payload = response or ""
        if error is not None:
            payload += f"\n\n===== ERROR =====\n{error}\n"
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(payload)
    except Exception:
        pass


def _ollama_chat(prompt: str, metrics=None, stage="writer"):
    import ollama
    import time
    _system = _writer_system_prompt()
    qwen_prose_transport = _uses_qwen_prose_transport(_MODEL)
    if qwen_prose_transport:
        _system += "\n\n" + _QWEN_PROSE_TRANSPORT_INSTRUCTION
    prompt_path = _dump_writer_prompt(stage, _system, prompt)
    start = time.perf_counter()
    chat_kwargs = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _system},
            {"role": "user", "content": prompt},
        ],
        "options": _DEFAULT_CHAT_OPTIONS,
        "keep_alive": _KEEP_ALIVE,
        "stream": False,
    }
    if qwen_prose_transport:
        chat_kwargs["format"] = _QWEN_PROSE_SCHEMA
    try:
        res = ollama.chat(**chat_kwargs)
    except Exception as e:
        _dump_writer_response(prompt_path, error=e)
        if metrics:
            metrics.record_llm(stage, _MODEL, options=_DEFAULT_CHAT_OPTIONS,
                               success=False, duration_s=time.perf_counter() - start,
                               prompt_chars=len(prompt), error=e)
        raise
    raw_out = (res.get("message", {}).get("content") or "").strip()
    try:
        out = (
            _decode_qwen_prose_transport(raw_out)
            if qwen_prose_transport
            else raw_out
        )
    except ValueError as exc:
        _dump_writer_response(prompt_path, response=raw_out, error=exc)
        if metrics:
            metrics.record_llm(
                stage,
                _MODEL,
                options=_DEFAULT_CHAT_OPTIONS,
                success=False,
                duration_s=time.perf_counter() - start,
                prompt_chars=len(prompt),
                response_chars=len(raw_out),
                error=exc,
            )
        raise
    _dump_writer_response(prompt_path, response=raw_out)
    if metrics:
        metrics.record_llm(stage, _MODEL, options=_DEFAULT_CHAT_OPTIONS,
                           duration_s=time.perf_counter() - start,
                           prompt_chars=len(prompt), response_chars=len(raw_out))
    return out


_FABRICATED_QUOTE_RE = re.compile(r'"([^"\n]{20,})"')


def _is_period_inside_paren(text: str, idx: int) -> bool:
    """v15.7: True if the '.' at text[idx] sits inside an unclosed '(' on the
    same line. Used by the sentence-boundary walker in _strip_fabricated_quotes
    so a period inside '(Author 2005, p.49)' is NOT treated as a sentence
    terminator — that was the v15.6 audit's broken-cite-stub bug producing
    '(Ogilvie_2007: p.' floating in prose after a quote-strip.
    """
    if idx < 0 or idx >= len(text):
        return False
    # Walk back to the previous newline or start; count parens.
    depth = 0
    i = idx - 1
    while i >= 0 and text[i] != "\n":
        ch = text[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            if depth == 0:
                # Unclosed open-paren before this period → period is inside it.
                return True
            depth -= 1
        i -= 1
    return False


def _normalise_for_quote_match(s: str) -> str:
    """Normalise for verbatim-quote substring matching.

    Lowercases, drops quote marks (straight + smart) and ellipsis markers,
    joins OCR hyphen-line-breaks (`deteriora-tion` from the PDF extractor
    becomes `deterioration` so a modern-prose quote of `deterioration` can
    match the original page text), and collapses whitespace.

    v14.2.2: added ellipsis-strip + hyphen-join after the v14.2.1 smoke
    showed two false-positive strips — Nunn p.4 had `deteriora-tion` in
    the page text vs `deterioration` in the model output; Acemoglu p.33
    had a real first-half quote that the model truncated with `...`. Both
    are now matched correctly.
    """
    if not s:
        return ""
    s = s.lower()
    # Drop quote marks (straight + curly/smart).
    s = re.sub(r"[\"'`“”‘’«»]", "", s)
    # Drop ellipsis markers (the model often inserts `...` or `…` to mark
    # truncation inside what is otherwise a verbatim quote; the cited
    # text after the ellipsis IS on the page, but the strict substring
    # check fails because the literal `...` is not).
    s = s.replace("…", " ")
    s = re.sub(r"\.{2,}", " ", s)
    # Join OCR hyphen-line-breaks: `word-\nword`, `word- word`, and the
    # in-text `word-word` artifact where pdfminer inserted a hyphen at a
    # line-end without preserving the newline. Conservative: only join
    # when the hyphen sits between two lowercase alphabetic chars (so
    # legit compounds like `long-run`, `cross-section` survive — they
    # have lowercase on both sides too, hmm, see note). The pdfminer
    # OCR pattern is `letter-letter` mid-word; the model paraphrasing
    # writes the dehyphenated form. We rejoin to make them match.
    s = re.sub(r"([a-z])-\s*([a-z])", r"\1\2", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _strip_fabricated_quotes(full_text: str, allowed_docs, metrics=None,
                              display_lookup=None):
    """v14.2 fix #1: scan the assembled essay for every '"..."' span >=20 chars,
    locate the nearest canonical (doc_id: p.N) citation within +/-200 chars,
    and verify the normalised quoted text is a substring of
    data/page_text/{doc_id}_page_{N}.txt. Sentences containing an
    UNVERIFIED quote are stripped.

    The check normalises both sides (lowercase, drop quote marks, collapse
    whitespace) so OCR-spaced page text and the model's tidy prose can still
    match. Runs AFTER _display_to_canonical (so the cite form is canonical and
    doc_id is directly resolvable) before later cleanup. Pre-existing E1/E2 checks operate on the
    same canonical form so the verifier shares the lookup contract.

    Returns (text, stats_dict). stats keys:
      checked_quotes, verified_real, fabricated_stripped,
      fabricated_kept_no_citation, fabricated_kept_doc_not_in_corpus,
      fabricated_kept_page_not_readable, fabrications[], duration_s.

    The `fabrications` list records (quote, doc_id, page, action) for each
    strip so the metric is auditable.

    Disabled when RRR_QUOTE_VERIFY=0 (default 1). Defensive: any unexpected
    error returns the original text unchanged with fallback_reason set.
    """
    import time
    from rrr.paths import page_text_path

    stats = {
        "enabled": True,
        "checked_quotes": 0,
        "verified_real": 0,
        "fabricated_stripped": 0,
        "fabricated_kept_no_citation": 0,
        "fabricated_kept_doc_not_in_corpus": 0,
        "fabricated_kept_page_not_readable": 0,
        "fabrications": [],
        "duration_s": 0.0,
        "fallback_reason": "ok",
    }
    if os.environ.get("RRR_QUOTE_VERIFY", "1") != "1":
        stats["enabled"] = False
        stats["fallback_reason"] = "disabled"
        return full_text, stats
    if not full_text:
        stats["fallback_reason"] = "empty_text"
        return full_text, stats

    allowed_doc_set = set(allowed_docs or ())
    start = time.perf_counter()
    page_text_cache: dict = {}

    def _load_page_text(doc_id: str, page: int):
        key = (doc_id, page)
        if key in page_text_cache:
            return page_text_cache[key]
        try:
            p = page_text_path(doc_id, page)
            if p.is_file():
                txt = p.read_text(encoding="utf-8", errors="replace")
                page_text_cache[key] = txt
                return txt
        except Exception:
            pass
        page_text_cache[key] = None
        return None

    spans_to_strip = []
    for qm in _FABRICATED_QUOTE_RE.finditer(full_text):
        stats["checked_quotes"] += 1
        quote = qm.group(1)
        q_start, q_end = qm.span()

        win_start = max(0, q_start - 200)
        win_end = min(len(full_text), q_end + 200)
        window = full_text[win_start:win_end]
        # v15.7: find nearest cite across ALL surfaces using parse_citations.
        # Previously CITE_RE-only — missed display-form cites entirely,
        # which after the surface flip is all of them.
        cites = list(parse_citations(window, display_lookup=display_lookup))
        cites = [c for c in cites if c["doc_id"] is not None]
        if not cites:
            stats["fabricated_kept_no_citation"] += 1
            continue

        q_center = (q_start + q_end) / 2 - win_start

        def _dist(c, _q=q_center):
            return abs((c["start"] + c["end"]) / 2 - _q)

        nearest = min(cites, key=_dist)
        doc_id = nearest["doc_id"]
        page = int(nearest["page"])

        if doc_id not in allowed_doc_set:
            stats["fabricated_kept_doc_not_in_corpus"] += 1
            continue

        page_txt = _load_page_text(doc_id, page)
        if not page_txt:
            stats["fabricated_kept_page_not_readable"] += 1
            continue

        norm_q = _normalise_for_quote_match(quote)
        norm_page = _normalise_for_quote_match(page_txt)
        if norm_q and norm_q in norm_page:
            stats["verified_real"] += 1
            continue

        # Fabricated quote. Strip the SENTENCE containing it.
        # v15.7: previous boundary walk halted on '.' inside '(Doc: p.N)'
        # producing stubs like '(Ogilvie_2007: p.' floating in the prose.
        # Use a smarter walk that treats a '.' inside an unclosed '('
        # as not a sentence terminator.
        sent_start = q_start
        while sent_start > 0:
            ch = full_text[sent_start - 1]
            if ch == "\n":
                break
            if ch in ".?!" and not _is_period_inside_paren(full_text, sent_start - 1):
                break
            sent_start -= 1
        while sent_start < q_start and full_text[sent_start] in " \t":
            sent_start += 1
        sent_end = q_end
        while sent_end < len(full_text):
            prev_ch = full_text[sent_end - 1]
            if prev_ch == "\n":
                break
            if prev_ch in ".?!" and not _is_period_inside_paren(full_text, sent_end - 1):
                break
            sent_end += 1
        # v15.7: drop-aware-of-surname guard. If stripping this sentence
        # would strand a surname in the prior 120 chars (the cite that
        # gave the surname semantic cover lives in `sent`), keep the
        # sentence — better to ship a fabricated-quote sentence than a
        # confidently-wrong attribution. Record the skip so the quality
        # manifest can flag it.
        sent_text = full_text[sent_start:sent_end]
        prev_text = full_text[max(0, sent_start - 200):sent_start]
        if _drop_would_strand_surname(
            prev_text, sent_text, allowed_doc_set, display_lookup=display_lookup,
        ):
            stats.setdefault("strand_guard_kept", 0)
            stats["strand_guard_kept"] += 1
            continue
        spans_to_strip.append((sent_start, sent_end, quote, doc_id, page))

    if spans_to_strip:
        # Strip in reverse order so earlier strips do not shift later spans.
        spans_to_strip.sort(key=lambda x: -x[0])
        out = full_text
        for sent_start, sent_end, quote, doc_id, page in spans_to_strip:
            out = out[:sent_start] + out[sent_end:]
            stats["fabricated_stripped"] += 1
            stats["fabrications"].append({
                "quote": quote[:240],
                "doc_id": doc_id,
                "page": page,
                "action": "sentence_stripped",
            })
        out = re.sub(r"  +", " ", out)
        out = re.sub(r"\s+\.", ".", out)
        out = re.sub(r"\n[ \t]+\n", "\n\n", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
    else:
        out = full_text

    stats["duration_s"] = round(time.perf_counter() - start, 3)
    return out, stats


def _first_paragraph_sentence_span(text: str, sentence_count: int):
    """Locate the first N sentences in the first paragraph."""
    source = text or ""
    paragraph_break = _PARAGRAPH_BREAK_RE.search(source)
    first_paragraph_end = (
        paragraph_break.start() if paragraph_break else len(source)
    )
    first_paragraph = source[:first_paragraph_end]
    sentences = _split_sentences_for_cleanup(first_paragraph)
    selected = sentences[:max(1, sentence_count)]
    if not selected:
        return None

    cursor = 0
    first_start = None
    last_end = None
    for sentence in selected:
        needle = sentence.strip()
        start = first_paragraph.find(needle, cursor)
        if start < 0:
            return None
        if first_start is None:
            first_start = start
        last_end = start + len(needle)
        cursor = last_end
    return {
        "start": first_start,
        "end": last_end,
        "text": first_paragraph[first_start:last_end],
        "sentence_count": len(selected),
    }


def _replace_first_paragraph_opener(text: str, sentence_count: int,
                                    rewritten: str) -> tuple:
    """Replace an opener while retaining every later source byte."""
    span = _first_paragraph_sentence_span(text, sentence_count)
    if not span:
        return text, "opener_span_missing"
    candidate = (rewritten or "").strip()
    if not candidate or "\n" in candidate or "\r" in candidate:
        return text, "paragraph_injection"
    original_length = max(1, len(span["text"].strip()))
    if len(candidate) > int(original_length * 1.2) + 10:
        return text, "length_growth"
    candidate_sentences = _split_sentences_for_cleanup(candidate)
    if len(candidate_sentences) != span["sentence_count"]:
        return text, "sentence_count_mismatch"
    if any(
        not _ends_with_terminal_punctuation(sentence)
        for sentence in candidate_sentences
    ):
        return text, "incomplete_rewrite"
    rewritten_text = text[:span["start"]] + candidate + text[span["end"]:]
    if len(_prose_paragraphs(rewritten_text)) != len(_prose_paragraphs(text)):
        return text, "paragraph_count_mismatch"
    return rewritten_text, "ok"


def _apply_cross_section_stitch(chunks, topic: str, metrics=None):
    """v11-B: rewrite the first sentence(s) of each interior section so the
    review flows from section to section instead of each restating the topic
    framing. One batched LLM call. Falls back to the original chunks on any
    parse / verification failure.

    v14.2 cond D: RRR_WRITER_STITCH_SENTENCES (default 1) controls how many
    opening sentences are rewritten together as a block. With N=3 the model
    has more room to set up a transition that does not restate the topic,
    instead of being forced to do all the work in a single sentence.

    Verification: the rewritten opener block must contain the SAME set of
    canonical citations (Doc_Year: p.N) as the original opener block; any
    addition or removal causes that block to be discarded. Only the first N
    sentences are replaced; the rest of each section is untouched.
    """
    import time
    # v13: RRR_WRITER_STITCH retired (always on). The v11-B stitch + v12
    # tail-echo guard are core to the cross-section flow story.
    # Need at least opening + 2 interior sections + closing to make stitching
    # meaningful. (Opening alone doesn't need stitching; closing already
    # synthesises across sections.)
    if not chunks or len(chunks) < 4:
        return chunks

    try:
        stitch_n = max(1, int(os.environ.get("RRR_WRITER_STITCH_SENTENCES", "1")))
    except ValueError:
        stitch_n = 1

    interior_indices = list(range(1, len(chunks) - 1))
    extracts = []
    for i in interior_indices:
        span = _first_paragraph_sentence_span(chunks[i], stitch_n)
        if not span:
            continue
        opener = span["text"].strip()
        prev_sents = _sentences_paragraph_local(chunks[i - 1])
        prev_tail = prev_sents[-1].strip() if prev_sents else ""
        if not opener:
            continue
        extracts.append({
            "index": i,
            "opener": opener,
            "opener_sent_count": span["sentence_count"],
            "prev_tail": prev_tail,
        })

    if len(extracts) < 2:
        return chunks

    opener_unit = "sentence" if stitch_n == 1 else f"first {stitch_n} sentences"
    parts = [
        f"Topic of the literature review: {topic}",
        "",
        f"Below are the OPENING {opener_unit} of several adjacent sections in "
        "the same review. Each currently restates the topic framing on its "
        "own, which reads as several mini-essays stitched together. Rewrite "
        f"each OPENING ({opener_unit}) as a complete standalone sentence "
        "that develops a theme, finding, or scope condition from the "
        "previous section. The previous section already ends with a complete "
        "sentence. Do not continue its grammar and do not re-introduce the topic.",
        "",
        "STRICT RULES:",
        "- Keep every citation token — 'Author (Year, p.N)', '(Author Year, p.N)', "
        "'(Doc_Year: p.N)' or '[E####]' — exactly UNCHANGED, character for character.",
        "- Do NOT add or remove any citation.",
        "- Do NOT exceed the original block length by more than ~20%.",
        "- Keep the original sentence count. Every rewritten sentence must "
        "end with terminal punctuation.",
        "- Return each rewritten opener as one line with no paragraph break.",
        f"- Return ONLY a single JSON object whose 'rewritten' field is the "
        f"{opener_unit} (joined into one string, original sentence-boundary "
        f"order preserved): "
        '{"openings": [{"index": <int>, "rewritten": "<text>"}, ...]}',
        "",
        "Sections to stitch:",
    ]
    for e in extracts:
        parts.append(f"\n--- section index={e['index']} ---")
        if e["prev_tail"]:
            parts.append(f"Previous section ended: {e['prev_tail']}")
        parts.append(f"Current opening (rewrite this): {e['opener']}")
    prompt = "\n".join(parts)

    try:
        import ollama
        start = time.perf_counter()
        prompt_path = _dump_writer_prompt("cross_section_stitch", "", prompt)
        # v13: RRR_WRITER_STITCH_T/CTX/PRED retired; tuning frozen.
        res = ollama.chat(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_ctx": 8192, "num_predict": 1500},
            keep_alive=_KEEP_ALIVE,
            format="json",
            stream=False,
        )
        raw = (res.get("message", {}).get("content") or "").strip()
        _dump_writer_response(prompt_path, response=raw)
        if metrics:
            metrics.record_llm("cross_section_stitch", _MODEL,
                               duration_s=time.perf_counter() - start,
                               prompt_chars=len(prompt), response_chars=len(raw))
    except Exception as e:
        if "prompt_path" in locals():
            _dump_writer_response(prompt_path, error=e)
        if metrics:
            metrics.record_llm("cross_section_stitch", _MODEL,
                               success=False, error=e)
        return chunks

    try:
        start_idx = raw.find("{")
        end_idx = raw.rfind("}")
        if start_idx < 0 or end_idx <= start_idx:
            return chunks
        obj = json.loads(raw[start_idx:end_idx + 1])
    except Exception:
        return chunks
    rewrites = obj.get("openings") if isinstance(obj, dict) else None
    if not isinstance(rewrites, list):
        return chunks

    new_chunks = list(chunks)
    applied = 0
    skipped_citation = 0
    skipped_empty = 0
    skipped_topic_paraphrase = 0
    skipped_tail_echo = 0
    skipped_layout_change = 0
    skipped_incomplete = 0
    interior_set = set(interior_indices)
    prev_tail_by_index = {e["index"]: e["prev_tail"] for e in extracts}

    # v15.14: the drift check was canonical-only ('(Doc_Year: p.N)'), but
    # post-v15.5 chunks reach the stitch ALREADY RENDERED to display form —
    # both sides matched zero tokens and the guard passed vacuously, letting
    # the stitch LLM drop or invent citations undetected. Fingerprint across
    # all three surfaces via parse_citations, plus any unrendered [E####]
    # markers, and require an exact multiset match.
    # v11.1: topic-paraphrase guard. Compute the topic's content-token set once,
    # then reject any stitch rewrite that pulls back to a near-restatement of
    # the topic line. Closes the v11 failure on section 4 where the LLM
    # produced "The question of whether institutions are indeed the fundamental
    # cause of long-run economic growth remains a pivotal area of inquiry..."
    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "of", "to", "in", "for", "on", "with", "as", "by", "at", "from",
        "this", "that", "these", "those", "and", "or", "but", "if", "then",
        "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
        "how", "their", "there", "here", "it", "its", "such", "any", "all",
        "no", "not", "do", "does", "did", "have", "has", "had", "can", "could",
        "may", "might", "will", "would", "shall", "should", "must",
        "more", "most", "less", "least", "very", "much", "many", "some",
        "about", "into", "through", "during", "before", "after", "above",
        "below", "between", "among", "across", "over", "under",
        "we", "you", "they", "i", "me", "us", "them", "he", "she", "him", "her",
        "our", "your", "my",
    }

    def _content_tokens(s: str) -> set:
        toks = re.findall(r"[A-Za-z][A-Za-z\-]+", (s or "").lower())
        return {t for t in toks if t not in _STOPWORDS and len(t) >= 3}

    topic_tokens = _content_tokens(topic)
    # Threshold: if the rewrite shares >= 2/3 of the topic's content tokens, it's
    # a paraphrase. Floor at 4 so a 5-word topic doesn't trip on coincidence.
    paraphrase_threshold = max(4, int(0.66 * len(topic_tokens)))

    opener_sent_count_by_index = {e["index"]: e["opener_sent_count"] for e in extracts}
    for r in rewrites:
        if not isinstance(r, dict):
            continue
        try:
            idx = int(r.get("index"))
        except (TypeError, ValueError):
            continue
        rewritten = str(r.get("rewritten", "") or "").strip()
        if not rewritten or idx not in interior_set:
            skipped_empty += 1
            continue
        span = _first_paragraph_sentence_span(
            new_chunks[idx], opener_sent_count_by_index.get(idx, 1),
        )
        if not span:
            skipped_layout_change += 1
            continue
        n_open = opener_sent_count_by_index.get(idx, 1)
        orig_opener_block = span["text"].strip()
        orig_cites = _citation_fingerprints(orig_opener_block)
        new_cites = _citation_fingerprints(rewritten)
        if orig_cites != new_cites:
            skipped_citation += 1
            continue
        # v11.1: reject if the rewrite paraphrases the topic line.
        rewrite_tokens = _content_tokens(rewritten)
        overlap = len(rewrite_tokens & topic_tokens)
        if topic_tokens and overlap >= paraphrase_threshold:
            skipped_topic_paraphrase += 1
            continue
        # v12: reject if the rewrite is an echo of the previous section's tail.
        # The original failure: the LLM "flows from prev_tail" by literally
        # copying it as the new opener, producing visible duplicate sentences
        # at every section boundary. Same shape as the topic-paraphrase guard
        # but referenced against prev_tail.
        prev_tail = prev_tail_by_index.get(idx, "")
        if prev_tail:
            prev_tail_tokens = _content_tokens(prev_tail)
            tail_overlap = len(rewrite_tokens & prev_tail_tokens)
            # Threshold: shared >= 6 content tokens, OR rewrite tokens are a
            # strict subset of prev_tail tokens with at least 4 shared. The
            # subset check catches paraphrases that change a few words.
            if (tail_overlap >= 6) or (
                tail_overlap >= 4 and rewrite_tokens and rewrite_tokens.issubset(prev_tail_tokens)
            ):
                skipped_tail_echo += 1
                continue
        rewritten_chunk, layout_reason = _replace_first_paragraph_opener(
            new_chunks[idx], n_open, rewritten,
        )
        if layout_reason != "ok":
            if layout_reason == "incomplete_rewrite":
                skipped_incomplete += 1
            else:
                skipped_layout_change += 1
            continue
        new_chunks[idx] = rewritten_chunk
        applied += 1

    if metrics:
        metrics.set("writer_stitch_applied", applied)
        metrics.set("writer_stitch_skipped_citation_change", skipped_citation)
        metrics.set("writer_stitch_skipped_empty", skipped_empty)
        metrics.set("writer_stitch_skipped_topic_paraphrase", skipped_topic_paraphrase)
        metrics.set("writer_stitch_skipped_tail_echo", skipped_tail_echo)
        metrics.set("writer_stitch_skipped_layout_change", skipped_layout_change)
        metrics.set("writer_stitch_skipped_incomplete", skipped_incomplete)
        metrics.set(
            "writer_stitch_paragraph_counts_preserved",
            all(
                len(_prose_paragraphs(before)) == len(_prose_paragraphs(after))
                for before, after in zip(chunks, new_chunks)
            ),
        )
    if applied or skipped_topic_paraphrase or skipped_tail_echo:
        print(f"[Writer] Cross-section stitch: rewrote {applied} section "
              f"opener(s) (skipped {skipped_citation} citation drift, "
              f"{skipped_topic_paraphrase} topic paraphrase, "
              f"{skipped_tail_echo} tail echo)")
    return new_chunks


# v13: _build_author_year_lookup promoted to render.py; we import it via the
# top-of-file import block. The writer-side _collect_cited_docs extends the
# shared one with display-form scans (DISPLAY_CITE_RE / DISPLAY_PAREN_CITE_RE)
# that the reasoner-side caller doesn't need.

def _collect_cited_docs(text: str, allowed_docs, author_year_to_docid):
    """Writer-side collector: shared canonical / bare-canonical / legacy
    author-year scans (via render._collect_cited_docs) PLUS the v10 display
    surfaces. The display branch lives here because reasoner's fallback path
    doesn't emit display forms.
    """
    cited_docs = _shared_collect_cited_docs(text, allowed_docs, author_year_to_docid)
    display_lookup = _build_display_lookup(allowed_docs)
    for m in DISPLAY_CITE_RE.finditer(text or ""):
        key = (m.group(1).strip().lower(), m.group(2))
        did = display_lookup.get(key)
        if did:
            cited_docs.add(did)
    for m in DISPLAY_PAREN_CITE_RE.finditer(text or ""):
        key = (m.group(1).strip().lower(), m.group(2))
        did = display_lookup.get(key)
        if did:
            cited_docs.add(did)
    return cited_docs


def compose_from_ledger(ledger_path=None, metrics=None):
    writer_started_at = time.perf_counter()
    ledger_path = ledger_path or str(runs_path("review_ledger.json"))
    if not os.path.isfile(ledger_path):
        raise SystemExit(f"Ledger not found: {ledger_path}")

    with open(ledger_path, encoding="utf-8") as f:
        data = json.load(f)
    with open(ledger_path, "rb") as f:
        ledger_sha256 = hashlib.sha256(f.read()).hexdigest()

    topic_display = data.get("topic", "(no topic)")
    # v12: writer prompts use the reformulated topic question. Falls back to
    # raw topic when the planner didn't (or couldn't) reformulate.
    topic = (data.get("plan", {}) or {}).get("topic_question") or topic_display
    topic_dimensions = (data.get("plan", {}) or {}).get("topic_dimensions") or []
    if topic != topic_display:
        print(f"[Writer] using reformulated topic: '{topic}'")
        if topic_dimensions:
            print(f"[Writer] topic dimensions: {topic_dimensions}")
    docs = data.get("docs", [])
    if not isinstance(docs, list) or not docs:
        raise SystemExit("Ledger empty or malformed (no docs).")

    # v15.9 (#6): provenance data flowed in through the ledger from the
    # reasoner (which read it from metadata.csv). Absent in older ledgers →
    # citations.json still emits (with pdf_path=null) so downstream tools
    # can detect the missing-provenance case.
    pdf_paths_by_docid = data.get("pdf_paths_by_docid", {}) or {}
    pdf_page_offsets = data.get("pdf_page_offsets", {}) or {}
    dois_by_docid = data.get("dois_by_docid", {}) or {}

    ledger_allowed_pairs, _, _ = _build_allowed_citations(docs)
    if not ledger_allowed_pairs:
        raise SystemExit("No allowed citations found in ledger.")

    # v15: read the corpus-level OUTLINE PLAN instead of stance-bucketed
    # cluster_syntheses. Each cluster is top-level (no nested-under-stance
    # structure), has a relation tag + free-text elaboration, and a
    # pre-computed section order.
    outline_plan = data.get("outline_plan") or {}
    if not outline_plan or not outline_plan.get("clusters"):
        raise SystemExit("Ledger missing outline_plan or outline produced no clusters.")
    topic_shape = outline_plan.get("topic_shape") or "causal"
    clusters_by_id = {c["cluster_id"]: c for c in outline_plan.get("clusters", [])}
    ordered_cluster_ids = (
        outline_plan.get("ordered_cluster_ids")
        or list(clusters_by_id.keys())
    )

    docs_by_id = {d.get("doc_id"): d for d in docs}

    # Relation distribution summary for the opening prompt — describes the
    # corpus's structural shape without any supports/critiques vocabulary.
    relation_dist = outline_plan.get("relation_distribution") or {}
    relation_summary_parts = []
    for relation in sorted(relation_dist.keys()):
        n = relation_dist[relation]
        gloss = _RELATION_OPENERS.get(relation, relation.replace("_", " "))
        relation_summary_parts.append(
            f"{n} source(s) in streams whose relation to the topic is: {gloss}"
        )
    stance_summary = (
        f"Topic shape: {topic_shape}. Of {len(docs)} sources, " +
        "; ".join(relation_summary_parts) + "." if relation_summary_parts
        else f"Topic shape: {topic_shape}. {len(docs)} sources across "
             f"{len(clusters_by_id)} streams."
    )
    print(f"[Writer] topic_shape={topic_shape} clusters={len(clusters_by_id)} "
          f"relations={relation_dist}")

    # Build chunk_plan in the cluster order Stage 3 picked. Each entry uses
    # the single v15 stream-section builder, partial-applied with the
    # cluster's relation + topic_shape.
    # The lead document remains first in each claims-first evidence target.
    # It no longer dictates the grammatical subject of the section opener.
    from functools import partial

    chunk_plan = []
    for cid in ordered_cluster_ids:
        c = clusters_by_id.get(cid)
        if not c or not c.get("doc_ids"):
            continue
        cluster_docs = [docs_by_id[did] for did in c["doc_ids"] if did in docs_by_id]
        if not cluster_docs:
            continue
        builder = partial(_build_stream_prompt,
                          relation=c.get("relation", ""),
                          topic_shape=topic_shape)
        # Reuse the legacy 5-tuple shape so the dispatcher loop below works
        # unchanged. Position 0 ("stance") now carries the relation; position
        # 1 ("cluster") carries the cluster's shared_thread label.
        chunk_plan.append((
            c.get("relation", "stream"),
            c.get("shared_thread") or cid,
            cluster_docs,
            builder,
            c,
        ))

    if not chunk_plan:
        raise SystemExit("No clusters produced any documents to write about.")

    print(f"[Writer] Generating {len(chunk_plan) + 2} sections "
          f"(opening + {len(chunk_plan)} stream sections + closing)...")
    if metrics:
        metrics.set("writer_chunk_plan", {
            "topic_shape": topic_shape,
            "clusters_total": len(chunk_plan),
            "relation_distribution": dict(relation_dist),
        })

    chunks = []
    section_claims = []
    total_evidence_id_renders = 0
    # v8 (R5): observability for new postproc surfaces
    total_double_paren_collapsed = 0
    total_author_led_openings = 0
    # v15.7: track renderer-side attribution gate + unknown-eid signal across
    # chunks so the quality manifest and finalize_covered_chunk can act on them.
    total_unknown_eids = 0
    total_attribution_mismatches = 0
    total_attribution_retries = 0
    total_structure_retries = 0
    total_section_model_retries = 0
    total_retry_candidates_accepted = 0
    total_sections_failed_after_retry = 0
    total_closing_model_retries = 0
    closing_retry_routes: list = []
    section_retry_routes = defaultdict(int)
    closing_digest_stats = {}
    total_invalid_citation_faults = 0
    total_raw_citation_faults = 0
    total_quote_faults = 0
    total_style_violations_detected = 0
    total_meta_commentary_detected = 0
    total_adjacent_paren_merges = 0
    total_duplicate_citation_identities_removed = 0
    total_post_sentence_marker_reattachments = 0
    quote_audit_records: list = []
    citation_surface_records: list = []
    unknown_eid_snippets: list = []
    attribution_mismatch_snippets: list = []
    structure_failure_records: list = []
    author_year_repair_stats = {
        "attempts": 0,
        "accepted": 0,
        "rejected": 0,
        "replacements_proposed": 0,
        "replacements_accepted": 0,
        "distinct_docs_rescued": 0,
        "collision_skips": 0,
    }
    author_year_repair_records: list = []
    section_coverage = []
    call_contracts = []
    call_allowed_pairs = set()
    call_allowed_docs = set()
    call_evidence_id_map = {}

    def register_call(stage, call_docs, allowed_list, prompt):
        contract = _build_call_contract(stage, call_docs, allowed_list, prompt)
        contract["call_index"] = len(call_contracts)
        call_contracts.append(contract)
        pairs, doc_ids, _ = _build_allowed_citations(call_docs)
        call_allowed_pairs.update(pairs)
        call_allowed_docs.update(doc_ids)
        call_evidence_id_map.update(_build_evidence_id_map(call_docs))
        ensure_dir(str(runs_path()))
        with open(runs_path("writer_call_contracts.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": _WRITER_ARTIFACT_CONTRACT,
                    "artifact_contract": _WRITER_ARTIFACT_CONTRACT,
                    "execution_profile": _writer_execution_profile(),
                    "citation_representation": _CITATION_REPRESENTATION_VERSION,
                    "post_acceptance_mutation_policy": (
                        _POST_ACCEPTANCE_MUTATION_POLICY
                    ),
                    "quotes_per_doc": _writer_quotes_per_doc(),
                    "calls": call_contracts,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def register_retry_call(stage, call_docs, prompt):
        _, _, pages_by_doc = _build_allowed_citations(call_docs)
        retry_allowed_list = _list_allowed_citations(call_docs, pages_by_doc)
        register_call(stage, call_docs, retry_allowed_list, prompt)

    def postprocess_chunk(chunk, chunk_docs):
        nonlocal total_evidence_id_renders
        nonlocal total_double_paren_collapsed, total_author_led_openings
        nonlocal total_unknown_eids, total_attribution_mismatches
        nonlocal total_invalid_citation_faults, total_raw_citation_faults
        nonlocal total_quote_faults, total_style_violations_detected
        nonlocal total_meta_commentary_detected, total_adjacent_paren_merges
        nonlocal total_duplicate_citation_identities_removed
        nonlocal total_post_sentence_marker_reattachments

        # v19 candidate transaction. Transport cleanup and deterministic
        # citation rendering happen before validation. Semantic detectors
        # inspect copies and never delete prose.
        chunk = _strip_wrapping(chunk).replace("\r\n", "\n").replace("\r", "\n")
        raw_citation_stats = _audit_raw_citation_contract(chunk, chunk_docs)
        total_raw_citation_faults += raw_citation_stats["raw_citation_faults"]
        chunk, marker_reattachments = (
            _reattach_post_sentence_evidence_markers(chunk)
        )
        total_post_sentence_marker_reattachments += marker_reattachments

        chunk_evidence_id_map = _build_evidence_id_map(chunk_docs)
        chunk, render_stats = _render_evidence_id_citations(
            chunk, chunk_evidence_id_map
        )
        render_stats.update(raw_citation_stats)
        render_stats["post_sentence_marker_reattachments"] = (
            marker_reattachments
        )
        total_evidence_id_renders += render_stats["replacements"]
        total_unknown_eids += render_stats["unknown_eids"]
        unknown_eid_snippets.extend(render_stats["unknown_eid_snippets"])
        total_attribution_mismatches += render_stats["attribution_mismatches"]
        attribution_mismatch_snippets.extend(
            render_stats["attribution_mismatch_snippets"]
        )

        chunk, dp_collapsed = _collapse_double_parens(chunk)
        total_double_paren_collapsed += dp_collapsed

        chunk_allowed_pairs, chunk_allowed_docs, _ = _build_allowed_citations(
            chunk_docs
        )
        chunk_display_lookup_local = _build_display_lookup(chunk_allowed_docs)
        citation_merge_stats = {}
        chunk, adjacent_merges = _merge_adjacent_paren_cites(
            chunk,
            display_lookup=chunk_display_lookup_local,
            stats=citation_merge_stats,
        )
        total_adjacent_paren_merges += adjacent_merges
        duplicates_removed = int(
            citation_merge_stats.get("duplicate_identities_removed", 0) or 0
        )
        total_duplicate_citation_identities_removed += duplicates_removed

        invalid_citations = []
        if _writer_enforcement_enabled():
            _discarded_invalid_cleanup, invalid_citations = (
                _remove_invalid_citations(
                    chunk,
                    chunk_allowed_docs,
                    allowed_pairs=chunk_allowed_pairs,
                    display_lookup=chunk_display_lookup_local,
                )
            )
        render_stats["invalid_citations"] = len(invalid_citations)
        render_stats["invalid_citation_records"] = invalid_citations[:10]
        total_invalid_citation_faults += len(invalid_citations)

        quote_stats = _audit_quote_contract(
            chunk,
            chunk_allowed_docs,
            display_lookup=chunk_display_lookup_local,
        )
        render_stats["quote_faults"] = quote_stats["quote_faults"]
        render_stats["quote_audit"] = quote_stats
        total_quote_faults += quote_stats["quote_faults"]
        if quote_stats["checked_quotes"] or quote_stats["quote_faults"]:
            quote_audit_records.append(quote_stats)

        surface_stats = _audit_citation_surface_contract(chunk)
        render_stats.update(surface_stats)
        if _citation_surface_fault_count(surface_stats):
            citation_surface_records.append(surface_stats)

        _sentences, style_violations = _collect_style_violations(chunk)
        render_stats["style_violations_detected"] = len(style_violations)
        total_style_violations_detected += len(style_violations)
        _discarded_meta_cleanup, meta_records = _strip_meta_commentary(chunk)
        render_stats["meta_commentary_detected"] = len(meta_records)
        render_stats["meta_commentary_snippets"] = meta_records[:5]
        total_meta_commentary_detected += len(meta_records)
        author_led_openings = _count_author_led_openings(chunk)
        render_stats["author_led_openings"] = author_led_openings
        total_author_led_openings += author_led_openings

        return chunk, 0, 0, 0, 0, render_stats

    def finalize_covered_chunk(raw, prompt, chunk_docs, section_kind, stage):
        nonlocal total_attribution_retries
        nonlocal total_structure_retries, total_closing_model_retries
        nonlocal total_section_model_retries
        nonlocal total_retry_candidates_accepted
        nonlocal total_sections_failed_after_retry
        local_failure_records = []
        chunk, repairs, placeholders, ajr, style_removed, render_stats = postprocess_chunk(raw, chunk_docs)
        audit = _audit_section_coverage(chunk, chunk_docs, section_kind)
        structural_only = not _writer_enforcement_enabled()

        def record_failed_candidate(state, phase, attempt):
            current_audit = state["audit"]
            current_render = state["render_stats"]
            record = {
                "stage": stage,
                "phase": phase,
                "attempt": attempt,
                "section": section_kind,
                "word_count": current_audit["word_count"],
                "paragraph_count": current_audit["paragraph_count"],
                "closing_paragraph_count_ok": current_audit[
                    "closing_paragraph_count_ok"
                ],
                "cited_doc_count": current_audit["cited_doc_count"],
                "required_cited_docs": current_audit["required_cited_docs"],
                "incomplete_paragraph_count": current_audit[
                    "incomplete_paragraph_count"
                ],
                "closing_word_floor_ok": current_audit[
                    "closing_word_floor_ok"
                ],
                "closing_word_ceiling_ok": current_audit[
                    "closing_word_ceiling_ok"
                ],
                "uncited_closing_paragraphs": current_audit[
                    "uncited_closing_paragraphs"
                ],
                "uncited_paragraphs": current_audit["uncited_paragraphs"],
                "unknown_eids": current_render["unknown_eids"],
                "attribution_mismatches": current_render[
                    "attribution_mismatches"
                ],
                "separated_parenthetical_groups": current_render.get(
                    "separated_parenthetical_groups", 0
                ),
                "redundant_author_parentheticals": current_render.get(
                    "redundant_author_parentheticals", 0
                ),
                "orphan_parenthetical_groups": current_render.get(
                    "orphan_parenthetical_groups", 0
                ),
                "raw_citation_faults": current_render.get(
                    "raw_citation_faults", 0
                ),
                "invalid_citations": current_render.get(
                    "invalid_citations", 0
                ),
                "quote_faults": current_render.get("quote_faults", 0),
                "faults": _section_quality_faults(
                    current_audit, current_render, section_kind,
                    structural_only,
                ),
            }
            local_failure_records.append(record)
            structure_failure_records.append(record)

        state = {
            "chunk": chunk,
            "repairs": repairs,
            "placeholders": placeholders,
            "ajr": ajr,
            "style_removed": style_removed,
            "audit": audit,
            "render_stats": render_stats,
        }
        if not _section_candidate_passes(
            state["audit"], state["render_stats"], section_kind,
            structural_only,
        ):
            record_failed_candidate(state, "generated", 0)

        def retry_candidate(prior_state, attempt):
            nonlocal total_attribution_retries, total_structure_retries
            nonlocal total_closing_model_retries
            nonlocal total_section_model_retries
            retry_reason = _section_retry_reason(
                prior_state["audit"], prior_state["render_stats"], section_kind,
                structural_only,
            )
            retry_stage = _section_retry_stage(
                stage, retry_reason, attempt, section_kind,
            )
            total_section_model_retries += 1
            section_retry_routes[retry_reason] += 1
            if section_kind == "closing":
                total_closing_model_retries += 1
                closing_retry_routes.append(retry_reason)
                if metrics:
                    metrics.inc("writer_closing_model_retries")
            if retry_reason == "structure":
                total_structure_retries += 1
                if metrics:
                    metrics.inc("writer_structure_retries")
            elif retry_reason == "attribution":
                total_attribution_retries += 1
                if metrics:
                    metrics.inc("writer_attribution_retries")
            elif retry_reason == "evidence":
                if metrics:
                    metrics.inc("writer_evidence_integrity_retries")
            elif metrics:
                metrics.inc("writer_section_coverage_retries")

            retry_prompt = _coverage_retry_prompt(
                prompt,
                prior_state["chunk"],
                prior_state["audit"]["required_cited_docs"],
                section_kind=section_kind,
                retry_reason=retry_reason,
                faults=_section_quality_faults(
                    prior_state["audit"],
                    prior_state["render_stats"],
                    section_kind,
                    structural_only,
                ),
            )
            register_retry_call(retry_stage, chunk_docs, retry_prompt)
            raw_retry = _ollama_chat(
                retry_prompt, metrics=metrics, stage=retry_stage,
            )
            (
                retry_chunk, repairs2, placeholders2, ajr2, style_removed2,
                retry_render_stats,
            ) = postprocess_chunk(raw_retry, chunk_docs)
            retry_state = {
                "chunk": retry_chunk,
                "repairs": prior_state["repairs"] + repairs2,
                "placeholders": prior_state["placeholders"] + placeholders2,
                "ajr": prior_state["ajr"] + ajr2,
                "style_removed": prior_state["style_removed"] + style_removed2,
                "audit": _audit_section_coverage(
                    retry_chunk, chunk_docs, section_kind,
                ),
                "render_stats": retry_render_stats,
            }
            nonclosing_resolution = None
            if section_kind != "closing" and not structural_only:
                retry_state, nonclosing_resolution = _resolve_nonclosing_retry(
                    prior_state, retry_state, retry_reason, section_kind,
                )

            if (
                metrics and retry_reason == "attribution"
                and (
                    nonclosing_resolution == "retry_accepted"
                    or _section_candidate_passes(
                        retry_state["audit"],
                        retry_state["render_stats"],
                        section_kind,
                        structural_only,
                    )
                )
            ):
                metrics.inc("writer_attribution_retry_accepted")
            elif metrics and retry_reason == "attribution":
                metrics.inc("writer_attribution_retry_rejected")
            if not _section_candidate_passes(
                retry_state["audit"],
                retry_state["render_stats"],
                section_kind,
                structural_only,
            ):
                record_failed_candidate(retry_state, "after_retry", attempt)
            return retry_state

        state, retries_used, accepted = _run_bounded_section_retries(
            state, section_kind, retry_candidate, structural_only,
        )
        if retries_used and accepted:
            total_retry_candidates_accepted += 1
        if not accepted:
            if retries_used:
                total_sections_failed_after_retry += 1
            failure_record = {
                "schema_version": _WRITER_ARTIFACT_CONTRACT,
                "execution_profile": _writer_execution_profile(),
                "error": "section_quality_validation_failed",
                "stage": stage,
                "section": section_kind,
                "model_retries": retries_used,
                "audit": state["audit"],
                "render_stats": state["render_stats"],
                "candidate_failures": local_failure_records,
                "faults": _section_quality_faults(
                    state["audit"], state["render_stats"], section_kind,
                    structural_only,
                ),
            }
            try:
                ensure_dir(str(runs_path()))
                with open(
                    runs_path("writer_section_failure.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(failure_record, f, indent=2, ensure_ascii=False)
            except Exception as artifact_error:
                print(
                    "[Writer] Could not write writer_section_failure.json: "
                    f"{artifact_error}"
                )
            raise ValueError(
                f"section quality failed for {section_kind}: "
                f"coverage={state['audit']['cited_doc_count']}/"
                f"{state['audit']['required_cited_docs']}, "
                f"paragraphs={state['audit']['paragraph_count']}, "
                f"incomplete_paragraphs="
                f"{state['audit']['incomplete_paragraph_count']}, "
                f"unknown_eids={state['render_stats']['unknown_eids']}, "
                f"attribution_mismatches="
                f"{state['render_stats']['attribution_mismatches']}, "
                f"citation_surface_faults="
                f"{_citation_surface_fault_count(state['render_stats'])}, "
                f"hard_integrity_faults="
                f"{_hard_candidate_fault_count(state['render_stats'])}, "
                f"words={state['audit']['word_count']}"
            )
        return (
            state["chunk"], state["repairs"], state["placeholders"],
            state["ajr"], state["style_removed"], state["audit"],
        )

    def generate_covered_chunk(prompt, chunk_docs, section_kind, stage):
        raw = _ollama_chat(prompt, metrics=metrics, stage=stage)
        return finalize_covered_chunk(raw, prompt, chunk_docs, section_kind, stage)

    # v15: opening docs draw the top-2 by score from EACH cluster in the
    # chunk plan, then the top 6 overall. This keeps the opening's evidence
    # diversified across streams without referencing any stance bucket.
    opening_docs_unbounded = []
    for relation, cluster_label, cluster_docs, _builder, _block in chunk_plan:
        opening_docs_unbounded.extend(
            sorted(cluster_docs, key=_score_doc, reverse=True)[:2]
        )
    opening_docs_unbounded = sorted(
        opening_docs_unbounded, key=_score_doc, reverse=True
    )[:6]
    opening_docs = _select_call_evidence(opening_docs_unbounded)

    _, _, opening_pages_by_doc = _build_allowed_citations(opening_docs)
    allowed_list = _list_allowed_citations(opening_docs, opening_pages_by_doc)
    evidence = "\n\n".join(_format_doc_entry(d) for d in opening_docs)

    prompt = _build_opening_prompt(topic, stance_summary, evidence, allowed_list)
    register_call("writer_opening", opening_docs, allowed_list, prompt)

    try:
        chunk, repairs, placeholders, ajr, style_removed, audit = generate_covered_chunk(
            prompt,
            opening_docs,
            "opening",
            "writer_opening",
        )
        word_count = _count_words(chunk)
        print(f"[Writer] Opening: {word_count} words; cited_docs={audit['cited_doc_count']}")
        section_coverage.append(audit)
        chunks.append(chunk)
        if metrics:
            metrics.inc("writer_sections_succeeded")
    except Exception as e:
        print(f"[Writer] Opening failed: {e}")
        if metrics:
            metrics.inc("writer_sections_failed")
        raise RuntimeError(
            "writer opening failed structural or evidence validation"
        ) from e

    # Generate stance sections
    stance_jobs = []
    parallel_tail = (
        chunks[-1][-_TAIL_CHARS:]
        if chunks and _writer_uses_raw_previous_tail()
        else ""
    )
    for i, (stance, cluster, cluster_docs, prompt_builder, cluster_plan) in enumerate(chunk_plan):
        cluster_docs_sorted = _select_call_evidence(
            sorted(cluster_docs, key=_score_doc, reverse=True)[:6]
        )
        _, _, cluster_pages_by_doc = _build_allowed_citations(cluster_docs_sorted)
        allowed_list = _list_allowed_citations(
            cluster_docs_sorted, cluster_pages_by_doc
        )
        evidence = "\n\n".join(_format_doc_entry(d) for d in cluster_docs_sorted)
        synthesis_block = _format_outline_block(
            cluster_plan, _group_evidence_ids_by_doc(cluster_docs_sorted)
        )
        prompt = prompt_builder(topic, cluster, evidence, allowed_list, parallel_tail,
                                synthesis_block)
        job = {
            "index": i,
            "stance": stance,
            "cluster": cluster,
            "docs": cluster_docs_sorted,
            "prompt": prompt,
            "stage": f"writer_{stance}",
            "allowed_list": allowed_list,
        }
        stance_jobs.append(job)

    def record_stance_chunk(job, chunk, word_count):
        top_mechs = []
        for d in job["docs"]:
            for m in d.get("mechanisms", []) or []:
                m = str(m).strip()
                if m and m not in top_mechs:
                    top_mechs.append(m)
        section_claims.append({
            "stance": job["stance"],
            "cluster": job["cluster"],
            "docs": [d.get("doc_id") for d in job["docs"]],
            "mechanisms": top_mechs[:4],
            "accepted_claim": _accepted_claim_excerpt(chunk),
            "word_count": word_count,
            # v15.4.0 (Bug 4 half-fix): extract author surnames the chunk
            # named in prose, so the next section's claims_so_far block can
            # forbid re-introducing them.
            "surnames_named": _extract_named_surnames(chunk),
        })
        chunks.append(chunk)

    def log_stance_chunk(job, word_count, repairs, placeholders, ajr, style_removed, audit):
        # v13: repairs / placeholders / ajr are vestigial (the helpers that
        # produced them were retired); keep the signature for caller stability
        # but only log the live signals (style sentences removed, citation
        # count). The unused parameters are accepted to keep the call sites
        # untouched.
        del repairs, placeholders, ajr
        notes = []
        if style_removed > 0:
            notes.append(f"style stripped {style_removed}")
        if audit.get("cited_doc_count", 0) > 0:
            notes.append(f"cited {audit['cited_doc_count']}")
        note_str = f" ({', '.join(notes)})" if notes else ""
        print(f"[Writer] {job['stance'].upper()}/{job['cluster']}: {word_count} words{note_str}")

    parallel_workers = _writer_parallel_workers(len(stance_jobs))
    if metrics:
        metrics.set("writer_parallel_workers", parallel_workers)

    if parallel_workers > 1:
        print(f"[Writer] Parallel stance chunks: workers={parallel_workers}")
        raw_by_index = {}
        for job in stance_jobs:
            register_call(
                job["stage"], job["docs"], job["allowed_list"], job["prompt"]
            )
        with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
            futures = {
                pool.submit(_ollama_chat, job["prompt"], metrics=metrics, stage=job["stage"]): job["index"]
                for job in stance_jobs
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    raw_by_index[idx] = future.result()
                except Exception as e:
                    raw_by_index[idx] = e

        for job in stance_jobs:
            try:
                raw = raw_by_index.get(job["index"])
                if isinstance(raw, Exception):
                    raise raw
                chunk, repairs, placeholders, ajr, style_removed, audit = finalize_covered_chunk(
                    raw,
                    job["prompt"],
                    job["docs"],
                    job["stance"],
                    job["stage"],
                )
                section_coverage.append(audit)
                word_count = _count_words(chunk)
                log_stance_chunk(job, word_count, repairs, placeholders, ajr, style_removed, audit)
                record_stance_chunk(job, chunk, word_count)
                if metrics:
                    metrics.inc("writer_sections_succeeded")
            except Exception as e:
                print(f"[Writer] {job['stance']}/{job['cluster']} failed: {e}")
                if metrics:
                    metrics.inc("writer_sections_failed")
                raise RuntimeError(
                    f"writer section failed for {job['stance']}/{job['cluster']}"
                ) from e
    else:
        # v11.2 lever 3: when sequential, build the accumulated claims summary
        # right before each section's prompt so it sees what's already been
        # argued. Empty for the first stance section (only the opening exists),
        # then grows as each section completes.
        print(f"[Writer] Sequential stance chunks: claims-so-far context enabled")
        for i, (stance, cluster, cluster_docs, prompt_builder, cluster_plan) in enumerate(chunk_plan):
            cluster_docs_sorted = _select_call_evidence(
                sorted(cluster_docs, key=_score_doc, reverse=True)[:6]
            )
            _, _, cluster_pages_by_doc = _build_allowed_citations(cluster_docs_sorted)
            allowed_list = _list_allowed_citations(
                cluster_docs_sorted, cluster_pages_by_doc
            )
            evidence = "\n\n".join(_format_doc_entry(d) for d in cluster_docs_sorted)
            synthesis_block = _format_outline_block(
                cluster_plan, _group_evidence_ids_by_doc(cluster_docs_sorted)
            )
            previous_tail = (
                chunks[-1][-_TAIL_CHARS:]
                if chunks and _writer_uses_raw_previous_tail()
                else ""
            )
            claims_so_far = _format_claims_so_far(section_claims)
            prompt = prompt_builder(topic, cluster, evidence, allowed_list, previous_tail,
                                    synthesis_block, claims_so_far)
            job = {
                "index": i,
                "stance": stance,
                "cluster": cluster,
                "docs": cluster_docs_sorted,
                "prompt": prompt,
                "stage": f"writer_{stance}",
            }
            register_call(job["stage"], job["docs"], allowed_list, prompt)
            try:
                chunk, repairs, placeholders, ajr, style_removed, audit = generate_covered_chunk(
                    job["prompt"],
                    job["docs"],
                    job["stance"],
                    job["stage"],
                )
                section_coverage.append(audit)
                word_count = _count_words(chunk)
                log_stance_chunk(job, word_count, repairs, placeholders, ajr, style_removed, audit)
                record_stance_chunk(job, chunk, word_count)
                if metrics:
                    metrics.inc("writer_sections_succeeded")
            except Exception as e:
                print(f"[Writer] {job['stance']}/{job['cluster']} failed: {e}")
                if metrics:
                    metrics.inc("writer_sections_failed")
                raise RuntimeError(
                    f"writer section failed for {job['stance']}/{job['cluster']}"
                ) from e

    # Generate the closing from a deterministic digest of the accepted body.
    # The packet contains one passage for at most four sources that the body
    # actually cited. It can therefore be reconstructed from a body checkpoint
    # without rerunning any accepted section.
    body_chunks = tuple(chunks)
    (
        body_checkpoint_payload,
        body_checkpoint,
        body_freeze_audit,
    ) = _build_body_checkpoint_payload(
        body_chunks,
        ledger_sha256=ledger_sha256,
        section_claims=section_claims,
        section_coverage=section_coverage,
        call_contracts=call_contracts,
        diagnostics={
            "section_model_retries": total_section_model_retries,
            "retry_candidates_accepted": (
                total_retry_candidates_accepted
            ),
            "sections_failed_after_retry": (
                total_sections_failed_after_retry
            ),
            "retry_routes": dict(section_retry_routes),
            "attribution_retries": total_attribution_retries,
            "structure_retries": total_structure_retries,
            "unknown_evidence_ids": total_unknown_eids,
            "attribution_mismatches": total_attribution_mismatches,
            "raw_citation_faults": total_raw_citation_faults,
            "invalid_citation_faults": total_invalid_citation_faults,
            "quote_faults": total_quote_faults,
            "post_sentence_marker_reattachments": (
                total_post_sentence_marker_reattachments
            ),
        },
    )
    ensure_dir(str(runs_path()))
    body_checkpoint_path = runs_path("writer_body_checkpoint.json")
    body_checkpoint_bytes = json.dumps(
        body_checkpoint_payload,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")
    body_checkpoint_path.write_bytes(body_checkpoint_bytes)
    body_checkpoint_artifact_sha256 = hashlib.sha256(
        body_checkpoint_bytes
    ).hexdigest()
    if metrics:
        metrics.set("writer_body_freeze", body_freeze_audit)
        metrics.set(
            "writer_body_checkpoint_sha256",
            _text_sha256(body_checkpoint),
        )
    closing_docs, closing_digest_stats = _select_closing_evidence_packet(
        docs, body_chunks,
    )
    available_doc_count = len({
        str(doc.get("doc_id", "") or "").strip()
        for doc in docs
        if str(doc.get("doc_id", "") or "").strip()
    })
    required_body_cited_docs = min(2, available_doc_count)
    if (
        _writer_enforcement_enabled()
        and len(closing_docs) < required_body_cited_docs
    ):
        raise RuntimeError(
            "accepted body did not retain enough valid cited sources for "
            f"closing generation: {len(closing_docs)}/"
            f"{required_body_cited_docs}"
        )

    _, _, closing_pages_by_doc = _build_allowed_citations(closing_docs)
    allowed_list = _list_allowed_citations(closing_docs, closing_pages_by_doc)
    if not allowed_list:
        allowed_list = "(No call-local citations are available.)"
    closing_digest = _build_closing_digest(
        section_claims, body_chunks, closing_docs,
    )
    closing_digest_stats = {
        **closing_digest_stats,
        "digest_chars": len(closing_digest),
        "digest_sha256": hashlib.sha256(
            closing_digest.encode("utf-8")
        ).hexdigest(),
    }
    prompt = _build_closing_prompt(topic, closing_digest, allowed_list)
    register_call("writer_closing", closing_docs, allowed_list, prompt)

    closing_generated_audit = None
    try:
        chunk, repairs, placeholders, ajr, style_removed, audit = generate_covered_chunk(
            prompt,
            closing_docs,
            "closing",
            "writer_closing",
        )
        word_count = _count_words(chunk)
        print(f"[Writer] Closing: {word_count} words; cited_docs={audit['cited_doc_count']}")
        section_coverage.append(audit)
        closing_generated_audit = dict(audit)
        chunks.append(chunk)
        closing_model_calls = 1 + total_closing_model_retries
        if closing_model_calls > _MAX_CLOSING_MODEL_CALLS:
            raise RuntimeError(
                "closing generation exceeded its model-call cap: "
                f"{closing_model_calls}/{_MAX_CLOSING_MODEL_CALLS}"
            )
        if metrics:
            metrics.set("writer_closing_model_calls_total", closing_model_calls)
            metrics.set(
                "writer_closing_first_pass_accepted",
                total_closing_model_retries == 0,
            )
            metrics.set("writer_closing_retry_routes", closing_retry_routes)
            metrics.set("writer_closing_digest", closing_digest_stats)
        if metrics:
            metrics.inc("writer_sections_succeeded")
    except Exception as e:
        print(f"[Writer] Closing failed: {e}")
        if metrics:
            metrics.inc("writer_sections_failed")
            metrics.set(
                "writer_closing_model_calls_total",
                1 + total_closing_model_retries,
            )
            metrics.set("writer_closing_first_pass_accepted", False)
            metrics.set("writer_closing_retry_routes", closing_retry_routes)
            metrics.set("writer_closing_digest", closing_digest_stats)
        raise RuntimeError(
            "writer closing failed structural or evidence validation"
        ) from e

    expected_chunk_count = len(chunk_plan) + 2
    if len(chunks) != expected_chunk_count:
        raise RuntimeError(
            "writer section count mismatch: "
            f"expected {expected_chunk_count}, found {len(chunks)}"
        )

    # Accepted section identities are now fixed. Continuity is created in each
    # section prompt; final assembly does not run a stitch call.
    if closing_generated_audit is None or not chunks:
        raise RuntimeError("writer lost closing section identity before assembly")

    # v15.14: refuse to ship an empty essay. Previously, if every section
    # generation failed (opening, all streams, closing), chunks == [] and an
    # EMPTY review_composed.md was written with exit code 0 — downstream
    # automation read that as a healthy run. Raising here lets the reasoner's
    # writer-error path record the failure honestly.
    if not chunks or not any((c or "").strip() for c in chunks):
        raise RuntimeError(
            "writer produced no usable sections (all chunk generations failed); "
            "refusing to write an empty review_composed.md"
        )

    # Final assembly
    accepted_sections = tuple(chunks)
    full_text, frozen_assembly_audit = _assemble_frozen_sections(
        accepted_sections,
        closing_index=len(accepted_sections) - 1,
    )
    # Each accepted response has already been rendered against its call-local
    # evidence map. Final assembly records zero renderer activity and rejects
    # any unresolved marker below.
    final_render_stats = {
        "replacements": 0,
        "unknown_eids": 0,
        "attribution_mismatches": 0,
    }

    # Bracketed document identifiers are rejected before section acceptance.
    # This retired-operation counter remains in the manifest for comparison.
    final_bracket_rewrites = 0

    # v15.7: _display_to_canonical retired at final assembly. The user-facing
    # surface is now display ('Author (Year, p.N)' / '(Author Year, p.N)')
    # end-to-end. Downstream validators consume display surfaces natively via
    # parse_citations + display_lookup. The display_lookup itself is still
    # built here because validators want it for doc_id resolution.
    allowed_pairs = call_allowed_pairs
    allowed_docs = call_allowed_docs
    display_lookup = _build_display_lookup(allowed_docs)

    # Quote verification runs on each candidate before acceptance. These
    # aggregate figures describe candidate faults; no shipped sentence is
    # deleted here.
    quote_verify_stats = {
        "checked_quotes": sum(
            int(record.get("checked_quotes", 0) or 0)
            for record in quote_audit_records
        ),
        "verified_real": sum(
            int(record.get("verified_real", 0) or 0)
            for record in quote_audit_records
        ),
        "quote_faults_detected": total_quote_faults,
        "fabricated_stripped": 0,
        "strand_guard_kept": 0,
        "post_acceptance_mutation": False,
        "candidate_audits": quote_audit_records[:20],
    }
    if metrics:
        metrics.set("writer_quote_verification", quote_verify_stats)

    # v13: removed the legacy final-assembly arms (_repair_year_only_citations,
    # _fix_ajr_abbreviation, _strip_placeholder_citations, _extract_citation_dumps,
    # _normalize_citation_case). All five had 0-hit metrics on the v12 smoke
    # and target shapes the v13 prompt+display surface no longer produces.

    removed_citations = []
    final_style_removed = []
    if metrics and os.environ.get("RRR_BYPASS_VALIDATION", "0") == "1":
        metrics.set("writer_bypass_validation", True)

    # The historical redundancy detector now runs on a copy for diagnostics.
    # Its proposed deletions never reach accepted prose.
    (
        _discarded_redundancy_cleanup,
        redundancy_candidates,
        closing_words_before_redundancy,
    ) = _drop_body_redundancy_preserving_closing(
        full_text,
        min_token_overlap=4,
        display_lookup=display_lookup,
        allowed_docs=allowed_docs,
    )
    redundancy_drops = []
    if metrics:
        metrics.set(
            "writer_redundancy_candidates",
            redundancy_candidates[:20],
        )

    # Meta-commentary is observed on a copy. Candidate prompts and local
    # retries carry the writing rule; final assembly does not delete prose.
    _discarded_meta_cleanup, meta_removed = _strip_meta_commentary(full_text)
    if metrics:
        metrics.set("writer_meta_commentary_detected", len(meta_removed))

    # Style rules are diagnostic after acceptance. The retired whole-review
    # rewriter makes no model call and applies no replacement.
    _style_sentences, final_style_violations = _collect_style_violations(
        full_text
    )
    style_stats = {
        "violations": len(final_style_violations),
        "rewrites_applied": 0,
        "trailing_stripped": 0,
        "fallback_reason": "diagnostic_only_v19",
        "post_acceptance_mutation": False,
    }
    if metrics:
        metrics.set("writer_style_enforcement", style_stats)

    # v15.5: placeholder strip retired — the writer no longer copies the
    # "(Author, Year)" exemplar because the prompt no longer SHOWS that
    # exemplar (replaced with "[E####]" usage examples).

    # Coverage fallback prose is retired. A sentinel in accepted text is an
    # invariant failure.
    n_patches_dropped = 0
    n_patches_kept = full_text.count(_COVERAGE_PATCH_SENTINEL)
    if metrics:
        metrics.set("writer_coverage_patches_dropped", n_patches_dropped)
        metrics.set("writer_coverage_patches_shipped", n_patches_kept)
    if n_patches_kept:
        raise RuntimeError("frozen section contains a legacy coverage patch")

    # Closing identity is explicit and revalidated below. Zero-citation
    # deletion is retired; every paragraph passed the section-local gate.
    citation_integrity_enforced = _writer_enforcement_enabled()
    zero_cite_dropped = []
    zc_kept_closing = None

    # Adjacent citation merging already ran inside each candidate transaction.
    # Final assembly only reports those deterministic pre-acceptance changes.
    adjacent_paren_merges = total_adjacent_paren_merges
    duplicate_citations_removed = (
        total_duplicate_citation_identities_removed
    )
    if metrics:
        metrics.set("writer_adjacent_paren_merges", adjacent_paren_merges)
        metrics.set(
            "writer_duplicate_citation_identities_removed",
            duplicate_citations_removed,
        )

    raw_eid_residue = _EVIDENCE_ID_LIKE_RE.findall(full_text)
    bracketed_doc_id_residue = [
        match.group(0) for match in _BRACKETED_DOC_ID_RE.finditer(full_text)
    ]
    pre_ship_surface_stats = _audit_citation_surface_contract(full_text)
    if raw_eid_residue or bracketed_doc_id_residue:
        raise RuntimeError(
            "frozen review retained an unresolved evidence identifier"
        )
    if _citation_surface_fault_count(pre_ship_surface_stats):
        raise RuntimeError(
            "frozen review retained a rejected citation surface"
        )

    # Final structural gate. Every paragraph must be complete, and the exact
    # generated closing must retain its word floor and document coverage.
    # A second dedupe scan is a no-residue assertion.
    final_structure = _audit_final_structure(
        full_text,
        closing_docs,
        enforce_citation_integrity=citation_integrity_enforced,
    )
    _deduped_check, residual_duplicate_citations = (
        _dedupe_grouped_citation_identities(
            full_text, display_lookup=display_lookup,
        )
    )
    final_structure = _apply_residual_duplicate_gate(
        final_structure,
        residual_duplicate_citations,
        citation_integrity_enforced,
    )
    final_structure_manifest = {
        key: value
        for key, value in final_structure.items()
        if key != "text_without_marker"
    }
    if metrics:
        metrics.set("writer_final_structure", final_structure_manifest)
        metrics.set("writer_structure_retries_total", total_structure_retries)
        metrics.set(
            "writer_closing_model_retries_total",
            total_closing_model_retries,
        )
        metrics.set(
            "writer_structure_failure_records",
            structure_failure_records[:20],
        )
    if not final_structure["ok"]:
        ensure_dir(str(runs_path()))
        failure_record = {
            "schema_version": _WRITER_ARTIFACT_CONTRACT,
            "execution_profile": _writer_execution_profile(),
            "error": "final_writer_structure_validation_failed",
            "final_structure": final_structure_manifest,
            "generated_closing": closing_generated_audit,
            "structure_retries": total_structure_retries,
            "structure_failures": structure_failure_records,
            "duplicate_citation_identities_removed": duplicate_citations_removed,
        }
        with open(
            runs_path("writer_structure_failure.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(failure_record, f, indent=2, ensure_ascii=False)
        raise RuntimeError(
            "final writer structure validation failed; refusing to write review"
        )

    full_text, shipped_freeze_audit = _assemble_frozen_sections(
        accepted_sections
    )
    if final_structure["text_without_marker"] != full_text:
        raise RuntimeError(
            "closing-marker removal changed the frozen review surface"
        )
    cited_docs = _strict_cited_doc_ids(
        full_text,
        allowed_pairs=allowed_pairs,
        display_lookup=display_lookup,
    )
    cited_docids = sorted(cited_docs)

    # v15.9 (#6): build citations.json provenance manifest AND, if
    # RRR_LINKIFY=1, rewrite in-text citations as markdown links pointing at
    # the source PDF at the right page. Runs BEFORE the final write so the
    # markdown that lands on disk is the linkified version when opted in.
    frozen_review_text = full_text
    manifest_text, citations_manifest = _emit_citations_manifest(
        full_text, allowed_docs, display_lookup,
        pdf_paths_by_docid, pdf_page_offsets, dois_by_docid,
        linkify=False,
    )
    if manifest_text != frozen_review_text:
        raise RuntimeError("citation manifest generation changed frozen prose")
    if citations_manifest:
        with open(runs_path("citations.json"), "w", encoding="utf-8") as f:
            json.dump(citations_manifest, f, indent=2, ensure_ascii=False)
        print(f"[Writer] citations.json written ({len(citations_manifest['citations'])} cites, "
              f"{citations_manifest['distinct_docs']} distinct docs)")

    if os.environ.get("RRR_LINKIFY", "0") == "1":
        linked_text, _linked_manifest = _emit_citations_manifest(
            frozen_review_text,
            allowed_docs,
            display_lookup,
            pdf_paths_by_docid,
            pdf_page_offsets,
            dois_by_docid,
            linkify=True,
        )
        linked_path = runs_path("review_composed_linked.md")
        linked_path.write_bytes(linked_text.encode("utf-8"))

    total_words = _count_words(frozen_review_text)

    ensure_dir(str(runs_path()))
    out_path = runs_path("review_composed.md")
    out_path.write_bytes(frozen_review_text.encode("utf-8"))
    output_sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    expected_output_sha256 = _text_sha256(frozen_review_text)
    if output_sha256 != expected_output_sha256:
        raise RuntimeError("review file bytes differ from the frozen review")
    full_text = frozen_review_text

    with open(runs_path("review_cited_docs.json"), "w", encoding="utf-8") as f:
        json.dump(cited_docids, f, indent=2)

    shipped_surface_stats = _audit_citation_surface_contract(full_text)
    shipped_source_parades = _count_source_parade_paragraphs(full_text)
    writer_runtime_seconds = round(
        time.perf_counter() - writer_started_at,
        6,
    )

    # v15.7: quality_manifest.json — surfaces every observable quality
    # signal the writer produced, plus snippets where applicable, so an
    # operator (or the 9-battery harness) can grade a review without
    # re-running the audit pipeline. NOT a runtime gate — the precheck
    # decision stays one-shot at Stage 0 — but a transparent record of
    # what the writer detected and self-healed.
    quality_manifest = {
        "schema_version": _WRITER_ARTIFACT_CONTRACT,
        "artifact_contract": _WRITER_ARTIFACT_CONTRACT,
        "execution_profile": _writer_execution_profile(),
        "citation_representation": _CITATION_REPRESENTATION_VERSION,
        "post_acceptance_mutation_policy": (
            _POST_ACCEPTANCE_MUTATION_POLICY
        ),
        "output_sha256": output_sha256,
        "output_bytes": len(out_path.read_bytes()),
        "body_checkpoint": {
            "schema_version": _BODY_CHECKPOINT_SCHEMA_VERSION,
            "path": body_checkpoint_path.name,
            "artifact_sha256": body_checkpoint_artifact_sha256,
            "body_sha256": body_checkpoint_payload["body_sha256"],
            "ledger_sha256": ledger_sha256,
            "reconstructs_exact_body": True,
        },
        "body_freeze": body_freeze_audit,
        "marked_assembly_freeze": frozen_assembly_audit,
        "shipped_assembly_freeze": shipped_freeze_audit,
        "accepted_sections_shipped_byte_identical": True,
        "writer_runtime_seconds": writer_runtime_seconds,
        "word_count": total_words,
        "distinct_docs_cited": len(cited_docids),
        "chunks_written": len(chunks),
        "chunks_expected": expected_chunk_count,
        "all_planned_sections_present": len(chunks) == expected_chunk_count,
        # Attribution gate (renderer-as-gate; coverage retry on mismatch)
        "attribution_mismatches": total_attribution_mismatches,
        "attribution_retries": total_attribution_retries,
        "attribution_mismatch_snippets": attribution_mismatch_snippets[:20],
        # Evidence-id integrity
        "unknown_evidence_ids": total_unknown_eids,
        "unknown_evidence_id_snippets": unknown_eid_snippets[:10],
        "author_year_repair": {
            **author_year_repair_stats,
            "records": author_year_repair_records[:20],
        },
        # Surface coherence (display-form leaks should now be 0 with the
        # ALLOWED CITATIONS prompt fix; tracked here to detect regressions)
        "invalid_citations_removed": 0,
        "invalid_citation_faults_detected": total_invalid_citation_faults,
        "raw_citation_faults_detected": total_raw_citation_faults,
        "citation_surface_candidate_faults": citation_surface_records[:20],
        "shipped_citation_surface": shipped_surface_stats,
        "double_paren_collapsed": total_double_paren_collapsed,
        "post_sentence_marker_reattachments": (
            total_post_sentence_marker_reattachments
        ),
        "adjacent_paren_merges": adjacent_paren_merges,
        "duplicate_citation_identities_removed": duplicate_citations_removed,
        "residual_duplicate_citation_identities": residual_duplicate_citations,
        "source_parade_paragraphs": shipped_source_parades,
        "author_led_openings_detected": total_author_led_openings,
        # Structural completeness and closing lineage
        "structure_retries": total_structure_retries,
        "closing_model_retries": total_closing_model_retries,
        "structure_failures": structure_failure_records,
        "final_structure": final_structure_manifest,
        "closing": {
            "model_calls": 1 + total_closing_model_retries,
            "maximum_model_calls": _MAX_CLOSING_MODEL_CALLS,
            "first_pass_accepted": total_closing_model_retries == 0,
            "retry_routes": closing_retry_routes,
            "digest": closing_digest_stats,
            "generated_words": closing_generated_audit["word_count"],
            "generated_cited_doc_count": closing_generated_audit["cited_doc_count"],
            "generated_cited_docs": closing_generated_audit["cited_docs"],
            "generated_paragraph_count": closing_generated_audit["paragraph_count"],
            "generated_complete": closing_generated_audit["structure_ok"],
            "generated_within_word_ceiling": closing_generated_audit[
                "closing_word_ceiling_ok"
            ],
            "generated_citation_occurrences": closing_generated_audit[
                "closing_citation_occurrences"
            ],
            "generated_unique_citation_pairs": closing_generated_audit[
                "closing_unique_citation_pairs"
            ],
            "generated_repeated_citation_pairs": closing_generated_audit[
                "closing_repeated_citation_pairs"
            ],
            "generated_citation_repetition_ok": closing_generated_audit[
                "closing_citation_repetition_ok"
            ],
            "words_before_redundancy": closing_words_before_redundancy,
            "shipped_words": final_structure_manifest["closing"]["word_count"],
            "shipped_cited_doc_count": final_structure_manifest["closing"]["cited_doc_count"],
            "shipped_cited_docs": final_structure_manifest["closing"]["cited_docs"],
            "shipped_paragraph_count": final_structure_manifest["closing"]["paragraph_count"],
            "shipped_complete": final_structure_manifest["closing"]["structure_ok"],
            "shipped_citation_occurrences": final_structure_manifest[
                "closing"
            ]["closing_citation_occurrences"],
            "shipped_unique_citation_pairs": final_structure_manifest[
                "closing"
            ]["closing_unique_citation_pairs"],
            "shipped_repeated_citation_pairs": final_structure_manifest[
                "closing"
            ]["closing_repeated_citation_pairs"],
            "shipped_citation_repetition_ok": final_structure_manifest[
                "closing"
            ]["closing_citation_repetition_ok"],
            "minimum_words": _MIN_CLOSING_WORDS,
            "maximum_words": _MAX_CLOSING_WORDS,
        },
        # Drop validators
        "zero_cite_paragraphs_dropped": len(zero_cite_dropped),
        "zero_cite_closing_kept": 1 if zc_kept_closing else 0,
        "redundancy_drops": len(redundancy_drops),
        "redundancy_candidates": len(redundancy_candidates),
        "style_sentences_removed": 0,
        "style_violations_detected": len(final_style_violations),
        "meta_commentary_detected": len(meta_removed),
        "style_enforcement": style_stats,
        "model_calls": {
            "total_writer_calls": len(call_contracts),
            "stitch_calls": 0,
            "whole_review_style_calls": 0,
            "closing_calls": 1 + total_closing_model_retries,
        },
        "candidate_transactions": {
            "accepted_sections": len(chunks),
            "rejected_candidates": len(structure_failure_records),
            "candidate_rollbacks": len(structure_failure_records),
            "model_retries": total_section_model_retries,
            "retry_candidates_accepted": (
                total_retry_candidates_accepted
            ),
            "sections_failed_after_retry": (
                total_sections_failed_after_retry
            ),
            "retry_routes": dict(section_retry_routes),
        },
        "post_acceptance_mutations": {
            "sentences_deleted": 0,
            "stitch_rewrites": 0,
            "style_rewrites": 0,
            "redundancy_deletions": 0,
            "citation_repairs": 0,
            "paragraphs_deleted": 0,
        },
        # Coverage fallback
        "coverage_fallbacks": 0,
        "coverage_patches_dropped_at_final": n_patches_dropped,
        "coverage_patches_shipped_to_user": n_patches_kept,
        # Quote verification
        "quote_verify": {
            "checked": quote_verify_stats.get("checked_quotes", 0),
            "verified_real": quote_verify_stats.get("verified_real", 0),
            "fabricated_stripped": quote_verify_stats.get("fabricated_stripped", 0),
            "strand_guard_kept": quote_verify_stats.get("strand_guard_kept", 0),
            "candidate_faults_detected": total_quote_faults,
            "post_acceptance_mutation": False,
        },
        # Outline (carried up from Stage 0/1/2 for one-stop quality view)
        "outline": {
            "topic_shape": (outline_plan or {}).get("topic_shape", ""),
            "topic_cause": (outline_plan or {}).get("topic_cause", ""),
            "topic_outcome": (outline_plan or {}).get("topic_outcome", ""),
            "clusters_count": len((outline_plan or {}).get("clusters", [])),
            "unassigned_share": (outline_plan or {}).get("unassigned_share"),
            "relation_distribution": (outline_plan or {}).get("relation_distribution", {}),
        },
    }
    with open(runs_path("quality_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(quality_manifest, f, indent=2)
    print(
        f"[Writer] quality_manifest.json written "
        f"(attribution_mismatches={total_attribution_mismatches}, "
        f"unknown_eids={total_unknown_eids}, "
        f"raw_citation_faults={total_raw_citation_faults}, "
        f"post_acceptance_mutations=0)"
    )

    print(f"[Writer] review_composed.md written ({total_words} words).")
    print(
        f"[Writer] stats: chunks={len(chunks)} distinct_docs={len(cited_docids)} "
        f"writer_calls={len(call_contracts)} evidence_id_renders={total_evidence_id_renders} "
        f"marker_reattachments={total_post_sentence_marker_reattachments} "
        f"author_year_repairs={author_year_repair_stats['replacements_accepted']} "
        f"double_paren_collapsed={total_double_paren_collapsed} author_led_openings={total_author_led_openings} "
        f"redundancy_candidates={len(redundancy_candidates)} source_parades={shipped_source_parades} "
        f"unknown_eids={total_unknown_eids} attribution_mismatches={total_attribution_mismatches} "
        f"attribution_retries={total_attribution_retries} "
        f"section_retries={total_section_model_retries} "
        f"retries_accepted={total_retry_candidates_accepted} "
        f"closing_model_retries={total_closing_model_retries} "
        f"runtime_s={writer_runtime_seconds:.3f} "
        f"post_acceptance_mutations=0"
    )
    if metrics:
        metrics.set("writer_stats", {
            "artifact_contract": _WRITER_ARTIFACT_CONTRACT,
            "execution_profile": _writer_execution_profile(),
            "citation_representation": _CITATION_REPRESENTATION_VERSION,
            "post_acceptance_mutation_policy": (
                _POST_ACCEPTANCE_MUTATION_POLICY
            ),
            "chunks_written": len(chunks),
            "distinct_docs_cited": len(cited_docids),
            "word_count": total_words,
            "output_sha256": output_sha256,
            "writer_runtime_seconds": writer_runtime_seconds,
            "writer_model_calls": len(call_contracts),
            "stitch_model_calls": 0,
            "whole_review_style_model_calls": 0,
            "post_acceptance_mutations": 0,
            "body_checkpoint_artifact_sha256": (
                body_checkpoint_artifact_sha256
            ),
            "body_checkpoint_sha256": body_checkpoint_payload[
                "body_sha256"
            ],
            "body_freeze": body_freeze_audit,
            "shipped_freeze": shipped_freeze_audit,
            "raw_citation_faults": total_raw_citation_faults,
            "invalid_citation_faults": total_invalid_citation_faults,
            "quote_faults": total_quote_faults,
            "rejected_candidates": len(structure_failure_records),
            "candidate_rollbacks": len(structure_failure_records),
            "section_model_retries": total_section_model_retries,
            "retry_candidates_accepted": (
                total_retry_candidates_accepted
            ),
            "sections_failed_after_retry": (
                total_sections_failed_after_retry
            ),
            "section_retry_routes": dict(section_retry_routes),
            "style_violations_detected": len(final_style_violations),
            "meta_commentary_detected": len(meta_removed),
            "citation_surface_candidate_faults": len(
                citation_surface_records
            ),
            "post_sentence_marker_reattachments": (
                total_post_sentence_marker_reattachments
            ),
            "source_parade_paragraphs": shipped_source_parades,
            "redundancy_candidates": len(redundancy_candidates),
            "removed_citations": 0,
            "style_sentences_removed": 0,
            "coverage_fallbacks": 0,
            "evidence_id_renders": total_evidence_id_renders,
            "author_year_repair": {
                **author_year_repair_stats,
                "records": author_year_repair_records[:20],
            },
            "double_paren_collapsed": total_double_paren_collapsed,
            "author_led_openings": total_author_led_openings,
            "redundancy_drops": len(redundancy_drops),
            "bracket_id_rewrites": 0,
            "unknown_eids": total_unknown_eids,
            "attribution_mismatches": total_attribution_mismatches,
            "attribution_retries": total_attribution_retries,
            "structure_retries": total_structure_retries,
            "closing_model_retries": total_closing_model_retries,
            "duplicate_citation_identities_removed": duplicate_citations_removed,
            "final_structure": final_structure_manifest,
            "section_claims": section_claims,
            "section_coverage": section_coverage,
        })
        metrics.set("writer_mode", "chunked")
        metrics.set(
            "writer_execution_profile", _writer_execution_profile()
        )
        metrics.inc("writer_removed_citations", 0)
        metrics.inc("writer_style_sentences_removed", 0)
        metrics.inc("writer_evidence_id_renders", total_evidence_id_renders)
        metrics.set(
            "writer_allowed_author_year_repairs",
            author_year_repair_stats["replacements_accepted"],
        )
        metrics.inc("writer_double_paren_collapsed", total_double_paren_collapsed)
        metrics.inc(
            "writer_post_sentence_marker_reattachments",
            total_post_sentence_marker_reattachments,
        )
        metrics.inc("writer_author_led_openings", total_author_led_openings)
        # v15.7: always emit attribution counters (even at 0) so the 9-battery
        # can distinguish "fired with zero hits" from "never fired".
        metrics.set("writer_unknown_evidence_id", total_unknown_eids)
        metrics.set("writer_attribution_mismatches", total_attribution_mismatches)
        metrics.set("writer_attribution_retries_total", total_attribution_retries)
        if unknown_eid_snippets:
            metrics.set("writer_unknown_evidence_id_snippets", unknown_eid_snippets[:10])
        if attribution_mismatch_snippets:
            metrics.set(
                "writer_attribution_mismatch_snippets",
                attribution_mismatch_snippets[:20],
            )
        metrics.inc("writer_bracket_id_rewrites", 0)

    return str(out_path)


def compose_review(ledger_path=None, metrics=None):
    """Run the current corpus-outline, multi-section writer."""
    mode = os.environ.get("RRR_WRITER_MODE", "chunked").strip().lower()
    if mode != "chunked":
        raise SystemExit(
            f"[Writer] unsupported RRR_WRITER_MODE={mode!r}; the current "
            "architecture uses the chunked corpus-outline writer"
        )
    return compose_from_ledger(ledger_path=ledger_path, metrics=metrics)


if __name__ == "__main__":
    compose_review()
