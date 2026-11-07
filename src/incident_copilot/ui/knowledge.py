"""Knowledge Base page: browse what is indexed, read a document, and (only on request) ask
the LLM to explain it.

Read-only. Everything comes from the API (GET /documents, GET /documents/{path}, POST
/documents/explain); the page never touches the database or the data directory and never
triggers ingestion. Lists and documents are cached for a minute so reruns do not query the
API again. Opening a document never calls the model; "Explain with AI" does, once per document
version, and the explanation is kept for this session keyed by the document's content hash.

Each document is a card; a click anywhere on it opens the document viewer, a large dialog
(st.dialog: Escape, the backdrop and its X close it). The viewer's header has "Explain with
AI" on the left and the title in the middle; the explanation opens as a panel beside the
document (below it on narrow screens), and the document pane stays the same element, so its
scroll position is kept. Which document is open is session state (`OPEN`), so a rerun
reopens the same one and closing it returns to the list with its filters as they were.
"""

import streamlit as st

from incident_copilot.ui.chat import render_failure
from incident_copilot.ui.client import AskResult, CausoraClient
from incident_copilot.ui.view import (
    DOC_TYPE_NAMES,
    DOC_TYPES,
    TOPIC_LABELS,
    escape_md,
    filter_documents,
    group_points,
)

CACHE_SECONDS = 60
OPEN = "kb_open"  # path of the document in the viewer (absent: closed)
EXPLAIN_OPEN = "kb_explain_open"  # path whose explanation panel is open
EXPLANATIONS = "explanations"  # "{path}@{content_hash}" -> AskResult, for this session


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def _documents(base_url: str) -> AskResult:
    return CausoraClient(base_url).list_documents()


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def _document(base_url: str, path: str) -> AskResult:
    return CausoraClient(base_url).get_document(path)


def _render_explanation(exp: dict, sections_by_id: dict) -> None:
    st.markdown(f"**Summary**  \n{escape_md(exp.get('summary'))}")
    for topic, points in group_points(exp.get("points")):
        st.markdown(f"**{escape_md(TOPIC_LABELS.get(topic, topic))}**")
        for p in points:
            badge = ":green-badge[Documented]" if p.get("basis") == "documented" \
                else ":orange-badge[Inferred]"
            refs = "; ".join(f"{escape_md(i)} ({escape_md(sections_by_id.get(i) or 'section')})"
                             for i in p.get("section_ids") or [])
            st.markdown(f"- {badge} {escape_md(p.get('text'))}"
                        + (f"  \n  :gray[Source: {refs}]" if refs else ""))
    missing = [m for m in exp.get("not_documented") or [] if m]
    if missing:
        st.markdown("**Not covered by this document**")
        for m in missing:
            # The model sometimes answers with a topic key; show its label then.
            st.markdown(f"- {escape_md(TOPIC_LABELS.get(m.strip(), m))}")
    if exp.get("truncated"):
        st.caption("The document is long: only its first part was sent to the model.")
    if exp.get("dropped_points"):
        st.caption(f"{exp['dropped_points']} point(s) were removed because they cited sections "
                   "that do not exist, or claimed to be documented without citing a section.")
    meta = [m for m in (exp.get("model") and f"model {exp['model']}",
                        exp.get("cached") and "from cache",
                        exp.get("generation_ms") is not None
                        and f"{exp['generation_ms']} ms",
                        (exp.get("llm_attempts") or 1) > 1
                        and f"{exp['llm_attempts']} LLM attempts") if m]
    st.caption("Uses only this document's own sections. Documented points restate the cited "
               "sections; inferred points are the model's reading of them and are not stated "
               "in the document. " + escape_md(" · ".join(meta)))


# Button callbacks: they change the state before the viewer is drawn again, whether that
# run is the dialog's own (a click inside it) or the whole page's.
def _toggle_panel(path: str) -> None:
    if st.session_state.get(EXPLAIN_OPEN) == path:
        st.session_state.pop(EXPLAIN_OPEN, None)
    else:
        st.session_state[EXPLAIN_OPEN] = path


def _forget_explanation(key: str) -> None:
    st.session_state.get(EXPLANATIONS, {}).pop(key, None)


def _explanation_panel(client: CausoraClient, doc: dict, key: str) -> None:
    """The AI panel for `doc`: asks the API once per document version, then keeps the
    result (or the failure, which can be retried) for this session. The header's toggle
    ("Hide explanation") closes it."""
    explanations = st.session_state.setdefault(EXPLANATIONS, {})
    st.markdown("#### AI explanation")
    if key not in explanations:
        with st.spinner("Asking the model to explain this document…"):
            explanations[key] = client.explain(doc["path"])
    result: AskResult = explanations[key]
    if result.ok:
        by_id = {s.get("id"): s.get("section") for s in result.data.get("sections") or []}
        _render_explanation(result.data, by_id)
        return
    render_failure(result.failure)
    st.caption("The document stays readable. Nothing was cached, so you can try again.")
    st.button("Try again", icon=":material/refresh:", key=f"explain_retry::{key}",
              on_click=_forget_explanation, args=(key,))


def _document_body(doc: dict, chunks: list, original, note: str) -> None:
    tabs = st.tabs(["Document", f"Indexed chunks ({len(chunks)})", "Details"])
    with tabs[0]:
        if original is not None:
            if doc["path"].endswith(".md"):
                with st.container(key="kb_doc_text"):
                    st.markdown(original)
            else:
                st.code(original, language=None, wrap_lines=True)
        st.caption(escape_md(note))
    with tabs[1]:
        st.caption("The chunks exactly as stored in the search index (sensitive values masked). "
                   "Each chunk repeats the document heading for context, so together they are "
                   "not identical to the original file.")
        for c in chunks:
            title = f"Chunk {c.get('chunk_index')}" + (f" · {c['section']}"
                                                         if c.get("section") else "")
            with st.expander(title):
                st.code(c.get("text") or "", language=None, wrap_lines=True)
    with tabs[2]:
        if doc.get("sections"):
            st.markdown("**Sections:** " + escape_md(", ".join(doc["sections"])))
        meta = {k: v for k, v in (doc.get("metadata") or {}).items()}
        meta |= {"pipeline": doc.get("pipeline"), "content_hash": doc.get("content_hash"),
                 "indexed_at": doc.get("updated_at")}
        st.table({"field": list(meta), "value": [str(v) for v in meta.values()]})


def _render_document(client: CausoraClient, path: str) -> None:
    """The viewer's content: header (Explain with AI | title and metadata), then the document
    pane and, when asked for, the explanation pane beside it."""
    result = _document(client.base_url, path)
    if not result.ok:
        _document.clear()  # do not keep a failure cached; the next rerun asks again
        render_failure(result.failure)
        return
    doc, chunks = result.data["document"], result.data.get("chunks") or []
    key = f"{doc['path']}@{doc.get('content_hash')}"
    explaining = st.session_state.get(EXPLAIN_OPEN) == doc["path"]

    with st.container(key="kb_viewer_head", horizontal=True, vertical_alignment="center",
                      gap="medium"):
        # One toggle with a fixed key, never disabled: the clicked button stays the focused
        # element, so Escape still reaches the dialog afterwards.
        st.button("Hide explanation" if explaining else "Explain with AI",
                  icon=":material/right_panel_close:" if explaining
                  else ":material/auto_awesome:",
                  type="secondary" if explaining else "primary", key=f"explain::{key}",
                  on_click=_toggle_panel, args=(doc["path"],),
                  help="Hide the explanation; the document stays open." if explaining else
                  "Asks the LLM to explain this document from its own indexed sections. "
                  "Nothing is sent until you click.")
        with st.container(key="kb_viewer_title", gap="xxsmall"):
            st.subheader(doc.get("title") or doc["path"], anchor=False)
            st.caption(f"{escape_md(DOC_TYPE_NAMES.get(doc['source_type'], doc['source_type']))}"
                       f" · `{doc['path']}` · {doc.get('chunk_count', len(chunks))} indexed "
                       "chunks" + (f" · pipeline `{doc['pipeline']}`" if doc.get("pipeline")
                                   else ""))

    # One horizontal row: the document pane is always its first element, so opening or
    # closing the explanation does not rebuild it (its scroll position stays).
    with st.container(key="kb_viewer_body", horizontal=True, gap="medium"):
        with st.container(key="kb_doc_pane"):
            _document_body(doc, chunks, result.data.get("original"),
                           result.data.get("original_note") or "")
        if explaining:
            with st.container(key="kb_ai_pane", border=True):
                _explanation_panel(client, doc, key)


def _close_viewer() -> None:
    st.session_state.pop(OPEN, None)
    st.session_state.pop(EXPLAIN_OPEN, None)


def _open_viewer(path: str) -> None:
    if st.session_state.get(OPEN) != path:
        st.session_state.pop(EXPLAIN_OPEN, None)  # a new document starts without the panel
    st.session_state[OPEN] = path


@st.dialog("Document viewer", width="large", on_dismiss=_close_viewer)
def _viewer(client: CausoraClient, path: str) -> None:
    _render_document(client, path)


def _card(d: dict, i: int) -> None:
    """One document as a card; the transparent button over the whole card opens it."""
    with st.container(key=f"kbcard_{i}", border=True):
        kind = DOC_TYPE_NAMES.get(d.get("source_type"), d.get("source_type"))
        st.markdown(f"**{escape_md(d.get('title') or d['path'])}**")
        facts = [kind, d.get("pipeline") and f"`{d['pipeline']}`",
                 f"{len(d.get('sections') or [])} sections",
                 f"{d.get('chunk_count')} chunks"]
        st.caption(" · ".join(escape_md(f) if not str(f).startswith("`") else f
                              for f in facts if f) + f"  \n`{d['path']}`")
        st.button(f"Open {d.get('title') or d['path']}", key=f"kbopen_{i}",
                  on_click=_open_viewer, args=(d["path"],), width="stretch")


def render(client: CausoraClient) -> None:
    st.title("Knowledge Base")
    st.caption("Everything Causora can cite: troubleshooting guides, task logs, incident "
               "reports (their timelines are sections inside them) and table schemas, as "
               "indexed. Read-only; browsing never re-indexes anything or calls the model.")
    listing = _documents(client.base_url)
    if not listing.ok:
        render_failure(listing.failure)
        if st.button("Try again", icon=":material/refresh:"):
            _documents.clear()
            st.rerun()
        return
    docs = listing.data
    if not docs:
        st.info("Nothing is indexed yet. Index the corpus first (see the README, section 6a).")
        return

    # Only the types the API returned, in DOC_TYPES order (unknown types last, as they are).
    counts: dict[str, int] = {}
    for d in docs:
        if isinstance(d, dict) and d.get("source_type"):
            counts[d["source_type"]] = counts.get(d["source_type"], 0) + 1
    order = list(DOC_TYPES)
    present = sorted(counts, key=lambda t: order.index(t) if t in order else len(order))
    c1, c2 = st.columns([2, 3])
    with c1:
        types = st.multiselect(
            "Type", present, default=[t for t in ("runbook",) if t in counts],
            format_func=lambda t: f"{DOC_TYPES.get(t, t)} ({counts.get(t, 0)})",
            help="Empty: all types.", key="kb_types")
    with c2:
        query = st.text_input("Search", placeholder="title, path, pipeline, error, ID…",
                              help="Every word must appear in the title, path, pipeline, "
                                   "sections or metadata (for example: rate limit, deadlock, "
                                   "INC-2043, orders_etl).", key="kb_query")
    shown = sorted(filter_documents(docs, types, query),  # grouped by type, stable within
                   key=lambda d: present.index(d["source_type"])
                   if d.get("source_type") in present else len(present))
    st.caption(f"{len(shown)} of {len(docs)} documents · click a document to open it")
    if not shown:
        st.info("No document matches. Clear the search or add types.")
        return
    with st.container(key="kb_cards"):
        for i, d in enumerate(shown):
            _card(d, i)

    path = st.session_state.get(OPEN)
    if path:
        _viewer(client, path)
