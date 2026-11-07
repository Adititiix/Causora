"""The chat page: questions go to POST /ask; this page renders what the API returned.

State (st.session_state is per browser session, so nothing is shared between users):
  messages        the conversation, oldest first: {"role": "user", "content"} and
                  {"role": "assistant", "result": AskResult, "question"}; an assistant message
                  the page made itself (choices narrowed from a typed reply, no API call) has
                  "local": True and "choices" instead of "result"
  pending_request a submitted question that has no answer yet: question, history, options
  conversation_id identifies this conversation's widgets (choice lists); a new one only on
                  "Clear conversation", never on navigation

How a question flows, and why (the earlier page showed stale elements and lost answers):
  1. The chat box, the example buttons and "Continue with this failure" call `queue_question`
     as an on_submit / on_click CALLBACK. It appends the user message and sets pending_request
     exactly once, before the rerun, so a rerun can never add a duplicate.
  2. Every run renders the whole conversation with ONE loop, in the same place, followed by
     the in-flight answer's placeholder. The newest messages are never drawn somewhere else
     first, so element positions are stable between runs and Streamlit never shows a
     previous answer's stale elements in the new answer's slot.
  3. While a request is pending, the chat box and the sidebar options are disabled, so the
     run that is waiting for the API is not interrupted by another question or option change.
     If it is interrupted anyway (e.g. by switching page), pending_request is still set and
     the next run of this page sends it again, so a question is never left without a reply.
  4. When the API answers (or fails), the assistant message is stored, pending_request is
     cleared and the page reruns to show it in its final place.

Switching to the Knowledge Base and back keeps everything: messages and pending_request are
plain session state, and every widget here keeps its value with persist_state="session"
(Streamlit otherwise drops a widget's value when its page is not shown). Text typed in the
chat box but not sent is the exception: st.chat_input keeps it only in the browser.

Clarifications ("which failure do you mean?"): the choices are a radio list; nothing is sent
until "Continue with this failure", which asks the same question again with the picked
failure's subject. A short typed reply instead ("i mean about the shipment") is matched
against the open choices (view.match_choices): one match is sent as that pick, several are
shown again as a narrowed list, none is sent as a new question.

The mascot: answers use the open-mouth cat as their avatar; the answer being waited for uses
the thinking cat, in a container keyed `causora_pending` that app.py's CSS animates. A new
answer "speaks" for SPEAK_SECONDS: its avatar is a two-frame sprite (open | closed mouth)
that app.py's CSS alternates. The time is counted from when the answer arrived, so a rerun or
coming back to the page never starts it again. User messages show the frog.
"""

import math
import time
import uuid
from pathlib import Path

import streamlit as st

from incident_copilot.ui.client import TOP_K_MAX, TOP_K_MIN, ApiFailure, AskResult, CausoraClient
from incident_copilot.ui.view import (
    EXAMPLE_QUESTIONS,
    OUTCOMES,
    AnswerView,
    Choice,
    build_view,
    escape_md,
    evidence_label,
    failure_hint,
    followup_hint,
    history_for_request,
    match_choices,
    open_clarification,
    outcome_kind,
)

PENDING = "pending_request"
CONVERSATION = "conversation_id"
DEFAULT_TOP_K = 8  # what the slider shows first; the server's ASK_TOP_K applies unless overridden

# Made by scripts/make_cat_icons.py. Streamlit's default icons if one is missing.
_ASSETS = Path(__file__).resolve().parent / "assets"


def _asset(name: str, fallback=None):
    return str(_ASSETS / name) if (_ASSETS / name).exists() else fallback


AVATAR = _asset("cat_avatar.png")
THINKING = _asset("cat_thinking.png", AVATAR)
SPEAKING = _asset("cat_speaking.png")  # open | closed mouth, side by side
USER_AVATAR = _asset("frog_avatar.png", ":material/person:")
SPEAK_SECONDS = 5.0
SPEAK_CYCLE_SECONDS = 1.0  # one open + one closed frame (app.py: causora-speak)

# Badge icon and callout per kind of result (view.OUTCOMES has the label and explanation).
_OUTCOME_ICONS = {"answered": ":material/check:", "retrieval_only": ":material/search:",
                  "gate": ":material/filter_alt:", "model": ":material/report:",
                  "withheld": ":material/block:", "clarify": ":material/live_help:"}


def queue_question(question: str | None, subject: dict | None = None,
                   choice_label: str | None = None, shown_as: str | None = None) -> None:
    """Callback: record a submitted question once. Ignored while another is pending.
    `subject` / `choice_label`: a failure picked from a clarification; the same question is
    asked again with that subject, and the user message shows what was picked. `shown_as`:
    the user's own words when they differ from the question sent (a typed pick)."""
    ss = st.session_state
    ss.setdefault("messages", [])
    text = (question or "").strip()
    if not text or ss.get(PENDING):
        return
    ss[PENDING] = {
        "question": text,
        # Earlier turns only, taken before this question is appended.
        "history": history_for_request(ss.messages),
        "retrieval_only": bool(ss.get("opt_retrieval_only")),
        "top_k": int(ss.get("opt_top_k", DEFAULT_TOP_K)) if ss.get("opt_override_k") else None,
    }
    message = {"role": "user", "content": (shown_as or "").strip() or text}
    if subject:
        ss[PENDING]["subject"] = dict(subject)
        message["selection"] = choice_label or ""
    ss.messages.append(message)


def _on_chat_submit() -> None:
    ss = st.session_state
    text = (ss.get("chat_box") or "").strip()
    if not text or ss.get(PENDING):
        return
    open_ = open_clarification(ss.get("messages", []))
    matches = match_choices(text, open_[1]) if open_ else []
    if len(matches) == 1:  # "i mean the column rename": unambiguous, so it is the pick
        queue_question(open_[0], subject=matches[0].subject, choice_label=matches[0].label,
                       shown_as=text)
    elif matches:  # still more than one: ask again, only among those (no API call)
        ss.messages.append({"role": "user", "content": text})
        ss.messages.append({"role": "assistant", "local": True, "question": open_[0],
                            "choices": matches})
    else:
        queue_question(text)


def _continue(question: str, radio_key: str, choices: list[Choice]) -> None:
    """Callback of "Continue with this failure": the picked choice, nothing if none is."""
    picked = st.session_state.get(radio_key)
    if isinstance(picked, int) and 0 <= picked < len(choices):
        queue_question(question, subject=choices[picked].subject,
                       choice_label=choices[picked].label)


def _clear() -> None:
    """A new conversation: messages, the pending request and the choice lists' state go;
    the sidebar options stay."""
    ss = st.session_state
    ss.messages = []
    ss.pop(PENDING, None)
    for key in [k for k in ss if isinstance(k, str) and k.startswith("choice::")]:
        del ss[key]
    ss[CONVERSATION] = uuid.uuid4().hex[:12]


def _sidebar_options(busy: bool) -> bool:
    """Options for the next question. Disabled while a request runs. Returns `debug`."""
    keep = {"persist_state": "session"}  # survives a visit to the Knowledge Base
    with st.sidebar:
        st.button("Clear conversation", icon=":material/delete:", on_click=_clear,
                  disabled=busy)
        with st.expander("Advanced", icon=":material/tune:"):
            st.toggle("Retrieval only (no LLM)", key="opt_retrieval_only", disabled=busy,
                      help="Show the evidence /ask retrieves without calling the model.", **keep)
            st.toggle("Override top_k", key="opt_override_k", disabled=busy,
                      help="top_k is the maximum number of evidence CHUNKS (passages) "
                           "retrieved for one question, not the number of documents: several "
                           "chunks can come from the same document. Off: the server's "
                           "ASK_TOP_K is used.", **keep)
            st.slider("top_k (chunks)", TOP_K_MIN, TOP_K_MAX, DEFAULT_TOP_K, key="opt_top_k",
                      disabled=busy or not st.session_state.get("opt_override_k"), **keep)
            debug = st.toggle("Show raw API response", key="opt_debug", **keep)
    return debug


def _source_line(view: AnswerView, ids) -> str:
    """`E5 (Task log · logs/... · run summary)` for each id, the id as a badge; ids the
    response does not contain are shown as they are, without a label."""
    parts = []
    for i in ids:
        chunk = view.chunk(i)
        parts.append(f":violet-badge[{escape_md(i)}] {escape_md(chunk.label)}" if chunk
                     else f":violet-badge[{escape_md(i)}]")
    return "; ".join(parts)


def render_choices(choices: list[Choice], question: str | None, key: str, busy: bool) -> None:
    """The failures as a radio list (whole rows clickable, mauve when picked; app.py) and a
    confirm button. Picking only selects; "Continue with this failure" sends the question
    again with that failure's subject. Labels and descriptions are shown literally."""
    if not choices or not question:
        return
    radio_key = f"choice::{key}"
    with st.container(key=f"clarify_{key}"):
        picked = st.radio(
            "Which failure do you mean?", list(range(len(choices))), index=None, key=radio_key,
            format_func=lambda i: escape_md(choices[i].label),
            captions=[escape_md(c.description) for c in choices], disabled=busy,
            persist_state="session", width="stretch")
        st.button("Continue with this failure", key=f"continue::{key}", type="primary",
                  icon=":material/arrow_forward:", disabled=busy or picked is None,
                  on_click=_continue, args=(question, radio_key, choices))


STEP_GROUPS = (("recovery", "Next steps: recovery"), ("prevention", "Next steps: prevention"))
STEP_SUPPORT = {
    "incident_report": ":green-badge[From the incident report]",
    "general_guidance": ":orange-badge[General guidance] :gray[from a runbook or log, not "
                        "documented as this incident's fix; check that it applies]",
}


def render_next_steps(view: AnswerView) -> None:
    """Recovery, then prevention (an older API sends no kinds: one list). Each step says where
    it comes from; one citing nothing is marked unsupported."""
    grouped = [(title, [s for s in view.next_steps if (s.kind or "recovery") == kind])
               for kind, title in STEP_GROUPS]
    if all(s.kind is None for s in view.next_steps):
        grouped = [("Next steps", list(view.next_steps))]
    for title, steps in grouped:
        if not steps:
            continue
        st.markdown(f"**{title}**")
        for n, step in enumerate(steps, 1):
            # Only the ids the model cited for the step (validated by the API) are a source.
            if step.evidence_ids:
                note = f":gray[Source:] {_source_line(view, step.evidence_ids)}"
                if step.support in STEP_SUPPORT:
                    note = f"{STEP_SUPPORT[step.support]}  \n  {note}"
            else:
                note = ":orange[Unsupported: no evidence was cited for this step]"
            st.markdown(f"{n}\\. {escape_md(step.text)}  \n  {note}")


def render_answer(view: AnswerView, raw: dict, debug: bool, question: str | None,
                  key: str = "", busy: bool = False) -> None:
    if view.interpreted_as:
        st.caption("Follow-up: evidence was searched for "
                   + escape_md("; ".join(view.interpreted_as))
                   + ". Earlier messages only set the topic; every fact below comes from the "
                   "retrieved documents.")
    kind = outcome_kind(view)
    label, color, explanation = OUTCOMES[kind]
    st.badge(label, color=color, icon=_OUTCOME_ICONS.get(kind, ":material/help:"))
    if kind == "retrieval_only":
        st.caption("No LLM was called. Below is the evidence /ask would give the model.")
    elif kind == "answered":
        if view.matched_failure:
            similarity = (f" (best similarity {view.best_similarity:.3f}, below the "
                          f"{view.min_similarity:.2f} threshold)"
                          if view.best_similarity is not None
                          and view.min_similarity is not None else "")
            st.caption("Answered about the only indexed failure this question matches: "
                       f"{escape_md(view.matched_failure)}{similarity}. Evidence about other "
                       "failures was left out.")
        st.markdown(f"**Summary**  \n{escape_md(view.summary)}")
    elif kind == "clarify" and view.choices:
        # A question back with choices: short, then the choices.
        st.info(escape_md(view.summary) or "Which failure do you mean?",
                icon=_OUTCOME_ICONS[kind])
        render_choices(view.choices, question, key, busy)
    else:
        # A refusal or a question back: calm, and says who stopped and why.
        callout = st.info if kind == "clarify" else st.warning
        callout(f"{escape_md(view.summary) or 'Not enough evidence to answer.'}  \n"
                f"**Why:** {escape_md(explanation)}", icon=_OUTCOME_ICONS[kind])
        if view.best_similarity is not None and view.min_similarity is not None:
            st.caption(f"Best similarity {view.best_similarity:.3f} "
                       f"(threshold {view.min_similarity:.2f}).")
        hint = followup_hint(question, view)
        if hint:
            st.info(hint, icon=":material/lightbulb:")

    if view.facts:
        st.markdown("**Stated in the evidence**")
        for c in view.facts:
            st.markdown(f"- {escape_md(c.text)}  \n  :gray[Source:] {_source_line(view, c.ids)}")
    if view.inferences:
        st.markdown("**Likely cause and other inferences** :gray[(the model's conclusions, "
                    "not stated in the documents)]")
        for c in view.inferences:
            conf = f"{escape_md(c.confidence)} confidence; " if c.confidence else ""
            st.markdown(f"- {escape_md(c.text)}  \n  :gray[{conf}based on] "
                        f"{_source_line(view, c.ids) or ':gray[no ids]'}")
    if view.unknowns:
        st.markdown("**Not established by the evidence**")
        for u in view.unknowns:
            st.markdown(f"- {escape_md(u)}")
    if view.next_steps:
        render_next_steps(view)
    elif kind == "answered":
        st.caption("No next steps were returned.")

    if view.chunks:
        with st.expander(evidence_label(view)):
            for c in view.chunks:
                tags = []
                if c.similarity is not None:
                    tags.append(f"similarity {c.similarity:.3f}")
                if c.origin and c.origin != "vector":
                    tags.append(f"added by {c.origin}")
                if c.linked_from:
                    tags.append(f"linked from {c.linked_from}")
                cited = ":green-badge[cited]" if c.used_in_answer else ":gray-badge[not cited]"
                st.markdown(f":violet-badge[{escape_md(c.evidence_id)}] {cited} "
                            f"{escape_md(c.label)}"
                            + (f"  \n:gray[{escape_md(' · '.join(tags))}]" if tags else ""))
                st.code(c.text, language=None, wrap_lines=True)
    attempts = view.timings.get("llm_attempts")
    meta = [m for m in (view.model and f"model {view.model}",
                        view.retriever and f"retriever {view.retriever}",
                        view.top_k is not None and f"top_k {view.top_k}",
                        view.latency_ms is not None and f"{view.latency_ms} ms",
                        isinstance(attempts, int) and attempts > 1
                        and f"{attempts} LLM attempts",
                        view.query_id and f"query {view.query_id}") if m]
    if meta:
        st.caption(escape_md(" · ".join(meta)))
    if debug:
        with st.expander("Raw API response"):
            st.json(raw, expanded=False)


def render_failure(failure: ApiFailure) -> None:
    """An error, never an answer: what to do first, then the technical detail."""
    status = f" (HTTP {failure.status})" if failure.status else ""
    hint = failure_hint(failure.code)
    # The detail is a code span: shown literally (no Markdown, no auto-linked URLs).
    detail = f"{failure.code}{status}: {failure.message}".replace("`", "'")
    st.error(f"**Causora could not complete this request.** {hint or ''}  \n"
             f":small[`{detail}`]", icon=":material/error:")
    if failure.query_id:
        st.caption(f"Recorded as query {escape_md(failure.query_id)}.")


# ---- the speaking cat ---------------------------------------------------------------------


def speaking_cycles(msg: dict, now: float) -> int:
    """How many open/closed cycles an answer still has to speak (0 = idle): counted from
    when it arrived, so neither a rerun nor coming back to the page starts it over."""
    started = msg.get("answered_at")
    if not isinstance(started, int | float):
        return 0
    remaining = SPEAK_SECONDS - (now - started)
    return max(0, math.ceil(remaining / SPEAK_CYCLE_SECONDS)) if remaining > 0 else 0


def speaking_css(key: str, cycles: int) -> str:
    """The rule that animates one answer's avatar (the sprite) for `cycles` cycles."""
    return (f".st-key-amsg_{key} [data-testid=\"stChatMessageAvatarImage\"] "
            f"{{ object-fit: cover; object-position: 0% 0; "
            f"animation: causora-speak {SPEAK_CYCLE_SECONDS:g}s steps(1, end) {cycles}; }}")


def _render_message(msg: dict, debug: bool, key: str, busy: bool, now: float) -> str | None:
    """Draws one message; returns its speaking CSS rule if it is still speaking."""
    role = msg["role"]
    if role == "user":
        with st.container(key=f"umsg_{key}"), st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(escape_md(msg.get("content", "")))
            if msg.get("selection"):
                st.caption("Selected: " + escape_md(msg["selection"]))
        return None
    cycles = speaking_cycles(msg, now) if SPEAKING else 0
    with st.container(key=f"amsg_{key}"), \
            st.chat_message("assistant", avatar=SPEAKING if cycles else AVATAR):
        if msg.get("local"):
            st.info("Which of these failures do you mean?", icon=_OUTCOME_ICONS["clarify"])
            render_choices(msg.get("choices") or [], msg.get("question"), key, busy)
            return None
        result: AskResult = msg["result"]
        if result.ok:
            render_answer(build_view(result.data), result.data, debug, msg.get("question"),
                          key=key, busy=busy)
        else:
            render_failure(result.failure)
    return speaking_css(key, cycles) if cycles else None


def _answer_pending(client: CausoraClient) -> None:
    """Call the API for the pending question, in the slot its answer will occupy."""
    pending = st.session_state[PENDING]
    # Static text, no spinner: the only motion is the avatar's bob, which app.py turns off
    # for prefers-reduced-motion.
    with st.container(key="causora_pending"), st.chat_message("assistant", avatar=THINKING):
        st.markdown("**Investigating…**")
        st.caption("Searching the indexed documents"
                   + ("." if pending["retrieval_only"] else " and asking the model. This can "
                      "take a few seconds."))
        try:
            # `subject` only for a picked clarification, so other requests are unchanged.
            extra = {"subject": pending["subject"]} if pending.get("subject") else {}
            result = client.ask(pending["question"],
                                retrieval_only=pending["retrieval_only"],
                                top_k=pending["top_k"], history=pending["history"], **extra)
        except Exception as exc:  # noqa: BLE001 - the client returns failures; be safe
            result = AskResult(failure=ApiFailure(
                "ui_error", f"the page could not complete the request "
                            f"({type(exc).__name__})"))
    message = {"role": "assistant", "result": result, "question": pending["question"]}
    if result.ok and isinstance(result.data, dict) and result.data.get("status") == "answered":
        message["answered_at"] = time.time()  # a new answer speaks (speaking_cycles)
    st.session_state.messages.append(message)
    st.session_state.pop(PENDING, None)


def render(client: CausoraClient) -> None:
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault(CONVERSATION, uuid.uuid4().hex[:12])
    busy = bool(ss.get(PENDING))
    debug = _sidebar_options(busy)

    with st.container(key="causora_hero", gap=None):  # brand type: app.py
        st.title("Causora", anchor=False)
        st.markdown("Answers questions about data-pipeline failures from the indexed task "
                    "logs, incident reports, troubleshooting guides and table schemas, with "
                    "citations, and says so when the evidence is insufficient.")
    st.caption("Ask why a task failed, how to fix an error, or what happened in a past "
               "incident. The documents are a synthetic demo corpus. Follow-ups such as "
               "\"how do we prevent it?\" use your earlier questions in this browser session "
               "to work out the subject; the facts still come only from the documents.")
    st.chat_input("Describe the failure or ask a question", key="chat_box",
                  on_submit=_on_chat_submit, disabled=busy)

    # One slot, emptied explicitly once there is a conversation: otherwise, while the first
    # question waits for the API, the examples drawn by the previous run stay on screen (faded)
    # after the new messages, until the run ends.
    examples = st.empty()
    if not ss.messages and not busy:
        with examples.container():
            st.markdown("**Try an example**")
            with st.container(horizontal=True, gap="small"):
                for q in EXAMPLE_QUESTIONS:
                    st.button(q, on_click=queue_question, args=(q,))
    else:
        examples.empty()

    now = time.time()
    speaking = [rule for i, msg in enumerate(ss.messages)
                if (rule := _render_message(msg, debug, f"{ss[CONVERSATION]}_{i}", busy, now))]
    if speaking:
        st.html("<style>" + "\n".join(speaking) + "</style>")
    if busy:
        _answer_pending(client)
        st.rerun()
