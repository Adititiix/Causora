"""The Streamlit pages themselves, run headless with streamlit.testing (AppTest).

The API client is replaced by a recording fake, so no API, network or LLM is used. Skipped
only if streamlit (requirements-ui.txt) is not installed.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from incident_copilot.ui.client import (  # noqa: E402
    ApiFailure,
    AskResult,
    CausoraClient,
    HealthResult,
)

APP = str(Path(__file__).resolve().parents[1] / "src" / "incident_copilot" / "ui" / "app.py")
Q1 = "Which past incident was caused by the payments provider API rate limiting our requests?"
Q2 = "What should we check first to prevent this happening again?"


def answered(question, path="incidents/INC-2043.md", text="INC-2043 lasted 205 minutes."):
    return {"query_id": f"q-{len(question)}", "status": "answered", "answer": text,
            "insufficient_evidence": False, "insufficient_reason": None,
            "evidence": [{"claim": text, "evidence_ids": ["E1"]}], "inference": [],
            "next_steps": [], "citations": [{"evidence_id": "E1", "path": path}],
            "retrieved": [{"evidence_id": "E1", "path": path, "source_type": "incident",
                           "section": "Overview", "text": text, "used_in_answer": True,
                           "scores": {"vector_similarity": 0.9}, "origin": "vector"}],
            "model": "fake", "retriever": "fake", "gate": {"best_similarity": 0.9,
                                                           "min_similarity": 0.72,
                                                           "passed": True},
            "validation": {}, "latency_ms": 10, "retrieval_info": {}, "timings": {},
            "top_k": 8}


def refused(reason, best=0.6):
    return {"query_id": "q-r", "status": "insufficient_evidence",
            "answer": "Not enough evidence to determine the cause",
            "insufficient_evidence": True, "insufficient_reason": reason, "evidence": [],
            "inference": [], "next_steps": [], "citations": [],
            "retrieved": [{"evidence_id": "E1", "path": "runbooks/x.md", "source_type": "runbook",
                           "section": "Symptoms", "text": "t", "used_in_answer": False,
                           "scores": {"vector_similarity": best}, "origin": "vector"}],
            "gate": {"best_similarity": best, "min_similarity": 0.72, "passed": best >= 0.72},
            "retrieval_info": {}, "timings": {}, "top_k": 8}


@pytest.fixture
def api(monkeypatch):
    """Patch the client class (the app builds its own instances) and record the calls."""
    calls = {"ask": [], "explain": [], "get_document": []}
    script: list = []

    def ask(self, question, *, retrieval_only=False, top_k=None, history=None, subject=None):
        calls["ask"].append({"question": question, "retrieval_only": retrieval_only,
                             "top_k": top_k, "history": history, "subject": subject})
        item = script.pop(0) if script else answered(question)
        return item if isinstance(item, AskResult) else AskResult(data=item)

    monkeypatch.setattr(CausoraClient, "ask", ask)
    monkeypatch.setattr(CausoraClient, "health",
                        lambda self, timeout=5.0: HealthResult(True, True, "ok", 104, 324))
    st.cache_data.clear()
    return calls, script


def run_app():
    return AppTest.from_file(APP, default_timeout=30).run()


def roles(at):
    return [m["role"] for m in at.session_state.messages]


def test_question_is_answered_once_and_history_survives_reruns(api):
    calls, _ = api
    at = run_app()
    at.chat_input[0].set_value(Q1).run()
    assert not at.exception
    assert roles(at) == ["user", "assistant"]
    assert [c["question"] for c in calls["ask"]] == [Q1] and calls["ask"][0]["history"] == []
    assert "pending_request" not in at.session_state
    at.run()
    at.run()  # plain reruns neither duplicate messages nor resend the question
    assert roles(at) == ["user", "assistant"] and len(calls["ask"]) == 1
    assert not at.chat_input[0].disabled


def test_follow_up_sends_the_earlier_turn_as_history(api):
    calls, _ = api
    at = run_app()
    at.chat_input[0].set_value(Q1).run()
    at.chat_input[0].set_value(Q2).run()
    assert roles(at) == ["user", "assistant", "user", "assistant"]
    history = calls["ask"][1]["history"]
    assert history == [{"question": Q1, "answer": "INC-2043 lasted 205 minutes.",
                        "cited_paths": ["incidents/INC-2043.md"]}]
    # Each answer stays attached to its own question.
    assert [m.get("question") for m in at.session_state.messages if m["role"] == "assistant"] \
        == [Q1, Q2]


def test_a_failure_keeps_the_conversation_and_the_next_question_works(api):
    calls, script = api
    script.append(AskResult(failure=ApiFailure("llm_upstream_error",
                                               "Gemini API error 504 (after 4 attempts)", 502)))
    at = run_app()
    at.chat_input[0].set_value(Q1).run()
    assert roles(at) == ["user", "assistant"]
    assert any("504" in e.value for e in at.error)
    at.chat_input[0].set_value(Q2).run()
    assert roles(at) == ["user", "assistant", "user", "assistant"]
    # The failed turn is still sent back as context (question only).
    assert calls["ask"][1]["history"] == [{"question": Q1}]


def test_an_interrupted_request_is_resumed_not_lost(api):
    calls, _ = api
    at = AppTest.from_file(APP, default_timeout=30)
    # State as left by a run that was stopped while waiting for the API.
    at.session_state["messages"] = [{"role": "user", "content": Q1}]
    at.session_state["pending_request"] = {"question": Q1, "history": [],
                                           "retrieval_only": False, "top_k": None}
    at.run()
    assert roles(at) == ["user", "assistant"] and len(calls["ask"]) == 1


def test_example_button_queues_its_question(api):
    calls, _ = api
    at = run_app()
    example = next(b for b in at.button if b.label.startswith("A metric task"))
    example.click().run()
    assert calls["ask"][0]["question"].startswith("A metric task fails with ZeroDivisionError")
    assert roles(at) == ["user", "assistant"]


def test_top_k_override_is_sent_only_when_enabled(api):
    calls, _ = api
    at = run_app()
    at.chat_input[0].set_value(Q1).run()
    at.toggle(key="opt_override_k").set_value(True).run()
    at.slider(key="opt_top_k").set_value(5).run()
    at.chat_input[0].set_value(Q2).run()
    assert [c["top_k"] for c in calls["ask"]] == [None, 5]


@pytest.mark.parametrize("reason, label, evidence", [
    ("below_min_similarity", "Refused: evidence too weak", "below the similarity threshold"),
    ("model_insufficient_evidence", "Refused by the model", "sent to the model; no answer"),
])
def test_refusals_say_who_refused_and_never_show_0_cited(api, reason, label, evidence):
    _, script = api
    script.append(refused(reason, best=0.6 if reason == "below_min_similarity" else 0.82))
    at = run_app()
    at.chat_input[0].set_value("Some question?").run()
    texts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    assert any(label in t for t in texts) or any(label in str(b.proto) for b in at.get("badge"))
    labels = [e.label for e in at.expander]
    assert any(evidence in lbl for lbl in labels) and not any("0 cited" in lbl for lbl in labels)


# ---- Knowledge Base -----------------------------------------------------------------------

DOCS = [
    {"path": "runbooks/api-rate-limited.md", "source_type": "runbook",
     "title": "Runbook: Upstream API rate limit (HTTP 429)", "pipeline": None,
     "sections": ["Symptoms", "Resolution"], "chunk_count": 2, "content_hash": "h1",
     "updated_at": None, "metadata": {"runbook_id": "RB-007"}},
    {"path": "runbooks/deadlock-detected.md", "source_type": "runbook",
     "title": "Runbook: Database deadlock during load", "pipeline": None,
     "sections": ["Symptoms"], "chunk_count": 1, "content_hash": "h2", "updated_at": None,
     "metadata": {}},
    {"path": "incidents/INC-2043.md", "source_type": "incident",
     "title": "INC-2043: Upstream API rate limit (HTTP 429) in payments_sync",
     "pipeline": "payments_sync", "sections": ["Summary"], "chunk_count": 1,
     "content_hash": "h3", "updated_at": None, "metadata": {"incident_id": "INC-2043"}},
]
DETAIL = {"document": DOCS[0], "original": "# Runbook\n\n## Resolution\n1. Wait.",
          "original_note": "Original source file from the data directory.",
          "chunks": [{"chunk_index": 0, "section": "Symptoms", "text": "429", "ts_start": None,
                      "ts_end": None},
                     {"chunk_index": 1, "section": "Resolution", "text": "Wait.",
                      "ts_start": None, "ts_end": None}]}


def _kb_script():
    from incident_copilot.ui import knowledge
    from incident_copilot.ui.client import CausoraClient

    knowledge._render_document(CausoraClient("http://fake"), "runbooks/api-rate-limited.md")


@pytest.fixture
def kb(monkeypatch, api):
    calls, _ = api

    def get_document(self, path):
        calls["get_document"].append(path)
        return AskResult(data=DETAIL)

    def explain(self, path):
        calls["explain"].append(path)
        return AskResult(data={
            "path": path, "summary": "How to handle HTTP 429.", "model": "fake",
            "points": [{"topic": "remediation", "text": "Wait for the window to reset.",
                        "basis": "documented", "section_ids": ["S2"]},
                       {"topic": "dependencies", "text": "Depends on the provider quota.",
                        "basis": "inferred", "section_ids": ["S1"]}],
            "not_documented": ["No owner escalation path is documented."],
            "sections": [{"id": "S1", "chunk_index": 0, "section": "Symptoms"},
                         {"id": "S2", "chunk_index": 1, "section": "Resolution"}],
            "dropped_points": 0, "truncated": False, "cached": False, "generation_ms": 5,
            "llm_attempts": 1})

    monkeypatch.setattr(CausoraClient, "get_document", get_document)
    monkeypatch.setattr(CausoraClient, "explain", explain)
    monkeypatch.setattr(CausoraClient, "list_documents", lambda self: AskResult(data=DOCS))
    return calls


def test_opening_a_document_shows_it_without_calling_the_model(kb):
    at = AppTest.from_function(_kb_script, default_timeout=30).run()
    assert not at.exception
    assert kb["explain"] == [] and kb["get_document"] == ["runbooks/api-rate-limited.md"]
    assert any("Runbook: Upstream API rate limit" in h.value for h in at.subheader)
    assert [e.label for e in at.expander if e.label.startswith("Chunk")] == \
        ["Chunk 0 · Symptoms", "Chunk 1 · Resolution"]
    at.run()  # a rerun is served from the cache and still never calls the model
    assert kb["explain"] == [] and len(kb["get_document"]) == 1


def test_explain_with_ai_runs_once_per_click_and_labels_documented_and_inferred(kb):
    at = AppTest.from_function(_kb_script, default_timeout=30).run()
    next(b for b in at.button if b.label == "Explain with AI").click().run()
    assert kb["explain"] == ["runbooks/api-rate-limited.md"]
    text = " ".join(m.value for m in at.markdown).replace("\\", "")  # undo Markdown escaping
    assert "Documented" in text and "Inferred" in text
    assert "S2 (Resolution)" in text and "No owner escalation path is documented." in text
    at.run()  # the explanation stays for this session without another call
    assert kb["explain"] == ["runbooks/api-rate-limited.md"]


def test_explain_failure_is_shown_as_an_error(kb, monkeypatch):
    monkeypatch.setattr(CausoraClient, "explain", lambda self, path: AskResult(
        failure=ApiFailure("llm_upstream_error", "Gemini API error 504", 502)))
    at = AppTest.from_function(_kb_script, default_timeout=30).run()
    next(b for b in at.button if b.label == "Explain with AI").click().run()
    assert any("504" in e.value for e in at.error) and not at.exception


def _kb_page():
    from incident_copilot.ui import knowledge
    from incident_copilot.ui.client import CausoraClient

    knowledge.render(CausoraClient("http://fake"))


def test_knowledge_base_lists_runbooks_and_filters(kb):
    at = AppTest.from_function(_kb_page, default_timeout=30).run()
    assert not at.exception
    assert any("2 of 3 documents" in c.value for c in at.caption)  # runbooks by default
    at.text_input[0].set_value("deadlock").run()
    assert any("1 of 3 documents" in c.value for c in at.caption)
    at.multiselect[0].set_value([]).run()
    at.text_input[0].set_value("INC-2043").run()
    assert any("1 of 3 documents" in c.value for c in at.caption)
    assert kb["explain"] == []


def test_knowledge_base_shows_api_errors(api, monkeypatch):
    monkeypatch.setattr(CausoraClient, "list_documents", lambda self: AskResult(
        failure=ApiFailure("db_unavailable", "database unavailable", 503)))
    at = AppTest.from_function(_kb_page, default_timeout=30).run()
    assert any("db_unavailable" in e.value for e in at.error) and not at.exception


# ---- clarification for vague questions ----------------------------------------------------

VAGUE = "why did the shipment fail?"
RENAME = {"incident_id": "INC-2045", "pipeline": "shipments_sync",
          "task": "transform_shipments", "date": "2026-10-06"}
DUPLICATE = {"incident_id": None, "pipeline": "shipments_sync", "task": "load_shipments_fact",
             "date": "2026-10-15"}


def clarification(labels=("shipments_sync / transform_shipments on 2026-10-06 (INC-2045)",
                          "shipments_sync / load_shipments_fact on 2026-10-15"),
                  descriptions=("INC-2045: Upstream column rename breaks transform_shipments",
                                "Logged error: UniqueViolation on shipments_fact_pkey")):
    body = refused("ambiguous_query", best=0.717)
    body["answer"] = "Several different failures match this question. Which one do you mean?"
    body["clarification"] = {"reason": "several_failures_match", "candidates": [
        {"id": f"c{i}", "label": label, "description": desc, "subject": subj}
        for i, (label, desc, subj) in enumerate(zip(labels, descriptions, (RENAME, DUPLICATE),
                                                    strict=True))]}
    return body


def answered_about(subject, text="It failed with KeyError: 'carrier_code'."):
    body = answered(VAGUE, path="extra/incidents/INC-2045.md", text=text)
    body["retrieval_info"] = {"conversation": {"status": "contextualized",
                                               "subject": dict(subject)}}
    return body


def choice_radio(at):
    return next(r for r in at.radio if r.key and r.key.startswith("choice::"))


def continue_button(at):
    return next(b for b in at.button if b.key and b.key.startswith("continue::"))


def ask_vague(api):
    _, script = api
    script.append(clarification())
    at = run_app()
    at.chat_input[0].set_value(VAGUE).run()
    assert not at.exception
    return at


def test_choices_render_as_a_radio_list_with_a_disabled_confirm_button(api):
    calls, _ = api
    at = ask_vague(api)
    radio = choice_radio(at)
    assert radio.label == "Which failure do you mean?" and radio.value is None
    assert radio.options == [
        r"shipments\_sync / transform\_shipments on 2026\-10\-06 \(INC\-2045\)",
        r"shipments\_sync / load\_shipments\_fact on 2026\-10\-15"]
    assert continue_button(at).disabled  # nothing picked yet
    texts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    assert any("Clarification needed" in t for t in texts) or \
        any("Clarification needed" in str(b.proto) for b in at.get("badge"))
    assert len(calls["ask"]) == 1 and calls["ask"][0]["subject"] is None


def test_picking_selects_without_sending_and_survives_reruns(api):
    calls, _ = api
    at = ask_vague(api)
    choice_radio(at).set_value(0).run()
    assert choice_radio(at).value == 0 and not continue_button(at).disabled
    choice_radio(at).set_value(1).run()  # another pick moves the selection
    assert choice_radio(at).value == 1
    at.run()
    at.run()  # harmless reruns keep it
    assert choice_radio(at).value == 1
    assert len(calls["ask"]) == 1  # selecting never calls the API


def test_continue_sends_the_original_question_with_the_picked_subject(api):
    calls, script = api
    at = ask_vague(api)
    script.append(answered_about(RENAME))
    choice_radio(at).set_value(0).run()
    continue_button(at).click().run()
    assert not at.exception
    sent = calls["ask"][1]
    assert sent["question"] == VAGUE
    assert sent["subject"] == {"incident_id": "INC-2045", "pipeline": "shipments_sync",
                               "task": "transform_shipments", "date": "2026-10-06"}
    assert sent["history"] == [{"question": VAGUE}]  # the clarified turn: question only
    assert roles(at) == ["user", "assistant", "user", "assistant"]
    assert at.session_state.messages[2]["selection"].startswith("shipments_sync /")
    assert any(c.value.startswith(r"Selected: shipments\_sync") for c in at.caption)


@pytest.mark.parametrize("follow_up", ["What is the solution for it?", "How do we prevent it?"])
def test_a_follow_up_after_a_pick_sends_the_picked_subject_in_history(api, follow_up):
    calls, script = api
    at = ask_vague(api)
    script.append(answered_about(RENAME))
    choice_radio(at).set_value(0).run()
    continue_button(at).click().run()
    at.chat_input[0].set_value(follow_up).run()
    history = calls["ask"][2]["history"]
    assert calls["ask"][2]["question"] == follow_up and calls["ask"][2]["subject"] is None
    assert history[-1]["subject"] == {"incident_id": "INC-2045", "pipeline": "shipments_sync",
                                      "task": "transform_shipments", "date": "2026-10-06"}


def test_a_typed_reply_matching_one_choice_is_sent_as_that_pick(api):
    calls, script = api
    at = ask_vague(api)
    script.append(answered_about(RENAME))
    at.chat_input[0].set_value("the column rename one").run()
    sent = calls["ask"][1]
    assert sent["question"] == VAGUE and sent["subject"]["incident_id"] == "INC-2045"
    user = at.session_state.messages[2]
    assert user["content"] == "the column rename one" and user["selection"]


def test_a_typed_reply_matching_several_choices_asks_again_without_calling_the_api(api):
    calls, _ = api
    at = ask_vague(api)
    at.chat_input[0].set_value("i mean about the shipment").run()
    assert len(calls["ask"]) == 1  # no request: the page narrows the choices itself
    local = at.session_state.messages[-1]
    assert local["local"] and [c.id for c in local["choices"]] == ["c0", "c1"]
    radio = choice_radio(at)
    assert len(radio.options) == 2 and radio.value is None
    assert any("Which of these failures" in i.value for i in at.info)


def test_a_typed_reply_matching_nothing_is_a_new_question(api):
    calls, _ = api
    at = ask_vague(api)
    at.chat_input[0].set_value("why did the orders_etl load fail on 2026-09-04?").run()
    assert calls["ask"][1]["question"] == "why did the orders_etl load fail on 2026-09-04?"
    assert calls["ask"][1]["subject"] is None


def test_choice_labels_and_descriptions_are_rendered_literally(api):
    _, script = api
    script.append(clarification(labels=("**bold** [link](http://x) <b>", "plain"),
                                descriptions=("`code` _x_", "")))
    at = run_app()
    at.chat_input[0].set_value(VAGUE).run()
    radio = choice_radio(at)
    assert radio.options[0] == r"\*\*bold\*\* \[link\]\(http://x\) \<b\>"
    assert radio.proto.captions[0] == r"\`code\` \_x\_"


def test_existing_refusals_show_no_choices(api):
    _, script = api
    script.append(refused("below_min_similarity"))
    at = run_app()
    at.chat_input[0].set_value(VAGUE).run()
    assert not [r for r in at.radio if r.key and r.key.startswith("choice::")]
    assert not at.exception


# ---- state across navigation, and the mascot ----------------------------------------------


def _two_page_script():
    """Stands in for st.navigation: one script, one page rendered per run (AppTest cannot
    switch function pages), so widgets of the page that is not shown are not rendered."""
    import streamlit as st

    from incident_copilot.ui import chat, knowledge
    from incident_copilot.ui.client import CausoraClient

    client = CausoraClient("http://fake")
    if st.session_state.get("_page", "chat") == "chat":
        chat.render(client)
    else:
        knowledge.render(client)


def test_chat_state_survives_a_visit_to_the_knowledge_base(api, monkeypatch):
    _, script = api
    monkeypatch.setattr(CausoraClient, "list_documents", lambda self: AskResult(data=[]))
    script.append(clarification())
    at = AppTest.from_function(_two_page_script, default_timeout=30).run()
    at.chat_input[0].set_value(VAGUE).run()
    at.toggle(key="opt_override_k").set_value(True).run()
    choice_radio(at).set_value(1).run()
    conversation = at.session_state["conversation_id"]

    at.session_state["_page"] = "kb"
    at.run()
    assert any(t.value == "Knowledge Base" for t in at.title) and not at.exception
    at.session_state["_page"] = "chat"
    at.run()

    assert roles(at) == ["user", "assistant"]
    assert at.session_state["conversation_id"] == conversation
    assert at.toggle(key="opt_override_k").value is True
    assert choice_radio(at).value == 1  # the pending choice is still picked
    assert not continue_button(at).disabled


def test_clear_conversation_starts_a_new_conversation(api):
    calls, _ = api
    at = run_app()
    at.chat_input[0].set_value(Q1).run()
    first = at.session_state["conversation_id"]
    next(b for b in at.button if b.label == "Clear conversation").click().run()
    assert at.session_state.messages == []
    assert at.session_state["conversation_id"] != first


def test_a_new_answer_speaks_for_five_seconds_then_idles(api, monkeypatch):
    from incident_copilot.ui import chat

    clock = [1000.0]
    monkeypatch.setattr(chat.time, "time", lambda: clock[0])
    at = run_app()
    at.chat_input[0].set_value(Q1).run()
    assert at.session_state.messages[-1]["answered_at"] == 1000.0
    styles = [h.proto.body for h in at.get("html") if ".st-key-amsg_" in h.proto.body]
    assert any(".st-key-amsg_" in s and " 5;" in s for s in styles)  # 5 cycles of 1 s

    clock[0] += 3.2  # a rerun mid-way continues, it does not start over
    at.run()
    styles = [h.proto.body for h in at.get("html") if ".st-key-amsg_" in h.proto.body]
    assert any(" 2;" in s for s in styles)

    clock[0] += 5  # after five seconds: idle, and a later rerun (or coming back) stays idle
    at.run()
    assert not [h for h in at.get("html") if ".st-key-amsg_" in h.proto.body]


def test_refusals_and_failures_do_not_speak(api):
    _, script = api
    script.append(refused("below_min_similarity"))
    at = run_app()
    at.chat_input[0].set_value(Q1).run()
    assert "answered_at" not in at.session_state.messages[-1]
    assert not [h for h in at.get("html") if ".st-key-amsg_" in h.proto.body]


def test_user_messages_use_the_frog_and_answers_the_cat(api):
    from incident_copilot.ui import chat

    at = run_app()
    at.chat_input[0].set_value(Q1).run()
    avatars = [m.avatar for m in at.chat_message]
    assert chat.USER_AVATAR.endswith("frog_avatar.png")
    assert len(avatars) == 2 and not at.exception


# ---- Knowledge Base: cards and the document viewer ------------------------------------------


def _open_card(at, title):
    """Click the card (its full-size transparent button, labelled "Open <title>")."""
    next(b for b in at.button if b.label == f"Open {title}").click().run()
    return at


def _explain(at):
    next(b for b in at.button if b.label == "Explain with AI").click().run()
    return at


def test_documents_are_cards_and_a_click_opens_the_viewer_without_the_model(kb):
    at = AppTest.from_function(_kb_page, default_timeout=30).run()
    labels = [b.label for b in at.button if b.label.startswith("Open ")]
    assert labels == ["Open Runbook: Upstream API rate limit (HTTP 429)",
                      "Open Runbook: Database deadlock during load"]  # runbooks by default
    assert kb["get_document"] == []  # nothing is fetched before a click
    _open_card(at, "Runbook: Upstream API rate limit (HTTP 429)")
    assert not at.exception
    assert at.session_state["kb_open"] == "runbooks/api-rate-limited.md"
    assert any("Runbook: Upstream API rate limit" in h.value for h in at.subheader)
    assert kb["get_document"] == ["runbooks/api-rate-limited.md"] and kb["explain"] == []
    at.run()  # a rerun keeps the same document open, from the cache, still no model call
    assert at.session_state["kb_open"] == "runbooks/api-rate-limited.md"
    assert len(kb["get_document"]) == 1 and kb["explain"] == []


def test_closing_the_viewer_keeps_the_filters(kb):
    at = AppTest.from_function(_kb_page, default_timeout=30).run()
    at.text_input(key="kb_query").set_value("rate").run()
    _open_card(at, "Runbook: Upstream API rate limit (HTTP 429)")
    _explain(at)
    # The state the dialog's X, Escape or a backdrop click leave (on_dismiss=_close_viewer,
    # tested below); AppTest cannot press the dialog's own X.
    del at.session_state["kb_open"]
    del at.session_state["kb_explain_open"]
    at.run()
    assert "kb_open" not in at.session_state and not at.exception
    assert not [h for h in at.subheader if "Runbook" in h.value]  # viewer closed
    assert at.text_input(key="kb_query").value == "rate"  # search kept
    assert any("1 of 3 documents" in c.value for c in at.caption)


def test_dismiss_callback_clears_the_open_document_and_its_panel():
    from incident_copilot.ui import knowledge

    state = {"kb_open": "a.md", "kb_explain_open": "a.md", "explanations": {"k": 1}}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(knowledge.st, "session_state", state)
        knowledge._close_viewer()
    assert state == {"explanations": {"k": 1}}  # explanations stay cached for the session


def test_explain_opens_the_panel_beside_the_document(kb):
    at = AppTest.from_function(_kb_page, default_timeout=30).run()
    _open_card(at, "Runbook: Upstream API rate limit (HTTP 429)")
    _explain(at)
    assert kb["explain"] == ["runbooks/api-rate-limited.md"]
    assert at.session_state["kb_explain_open"] == "runbooks/api-rate-limited.md"
    assert any(m.value == "#### AI explanation" for m in at.markdown)
    # The document is still shown next to it.
    assert [e.label for e in at.expander if e.label.startswith("Chunk")] == \
        ["Chunk 0 · Symptoms", "Chunk 1 · Resolution"]
    # The same header button now hides it (one toggle, never disabled, so focus stays).
    assert not [b for b in at.button if b.label == "Explain with AI"]
    toggle = next(b for b in at.button if b.label == "Hide explanation")
    assert not toggle.disabled
    toggle.click().run()
    assert "kb_explain_open" not in at.session_state
    assert any("Runbook: Upstream API rate limit" in h.value for h in at.subheader)  # still open
    _explain(at)  # reopening uses the session cache: no second call
    assert kb["explain"] == ["runbooks/api-rate-limited.md"]


def test_another_document_gets_its_own_explanation(kb, monkeypatch):
    deadlock = {**DETAIL, "document": DOCS[1]}
    monkeypatch.setattr(CausoraClient, "get_document", lambda self, path: (
        kb["get_document"].append(path) or AskResult(
            data=deadlock if "deadlock" in path else DETAIL)))
    at = AppTest.from_function(_kb_page, default_timeout=30).run()
    _open_card(at, "Runbook: Upstream API rate limit (HTTP 429)")
    _explain(at)
    # Close the viewer (as on_dismiss does), then click the other card.
    del at.session_state["kb_open"]
    del at.session_state["kb_explain_open"]
    at.run()
    _open_card(at, "Runbook: Database deadlock during load")
    assert any("Database deadlock" in h.value for h in at.subheader)
    assert not [m for m in at.markdown if m.value == "#### AI explanation"]  # panel closed
    _explain(at)
    assert kb["explain"] == ["runbooks/api-rate-limited.md", "runbooks/deadlock-detected.md"]
    assert set(at.session_state["explanations"]) == {"runbooks/api-rate-limited.md@h1",
                                                     "runbooks/deadlock-detected.md@h2"}


def test_opening_a_new_document_closes_the_previous_panel():
    from incident_copilot.ui import knowledge

    state = {"kb_open": "a.md", "kb_explain_open": "a.md"}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(knowledge.st, "session_state", state)
        knowledge._open_viewer("a.md")  # the same document: panel kept
        assert state["kb_explain_open"] == "a.md"
        knowledge._open_viewer("b.md")
    assert state == {"kb_open": "b.md"}


def test_a_failed_explanation_keeps_the_document_and_can_be_retried(kb, monkeypatch):
    outcomes = [AskResult(failure=ApiFailure("llm_upstream_error", "Gemini API error 503", 502)),
                None]

    def explain(self, path):
        kb["explain"].append(path)
        item = outcomes.pop(0)
        return item or AskResult(data={"path": path, "summary": "Now it works.", "points": [],
                                       "not_documented": [], "sections": []})

    monkeypatch.setattr(CausoraClient, "explain", explain)
    at = AppTest.from_function(_kb_page, default_timeout=30).run()
    _open_card(at, "Runbook: Upstream API rate limit (HTTP 429)")
    _explain(at)
    assert any("503" in e.value for e in at.error) and not at.exception
    assert [e.label for e in at.expander if e.label.startswith("Chunk")]  # document still there
    next(b for b in at.button if b.label == "Try again").click().run()
    assert len(kb["explain"]) == 2 and not at.error
    assert any("Now it works." in m.value.replace("\\", "") for m in at.markdown)


def test_a_document_that_cannot_be_loaded_shows_an_error_in_the_viewer(kb, monkeypatch):
    monkeypatch.setattr(CausoraClient, "get_document", lambda self, path: AskResult(
        failure=ApiFailure("document_not_found", "no indexed document has this path", 404)))
    at = AppTest.from_function(_kb_page, default_timeout=30).run()
    _open_card(at, "Runbook: Upstream API rate limit (HTTP 429)")
    assert any("document_not_found" in e.value for e in at.error) and not at.exception


# ---- chat: where next steps come from, what is unknown, near misses -------------------------


def test_steps_are_split_into_recovery_and_prevention_with_their_support(api):
    _, script = api
    body = answered(Q1)
    body["retrieved"].append({"evidence_id": "E2", "path": "runbooks/sensor-timeout.md",
                              "source_type": "runbook", "section": "Resolution",
                              "text": "Clear the sensor.", "used_in_answer": False,
                              "scores": {"vector_similarity": 0.55}, "origin": "error_link"})
    body |= {"next_steps": ["Cancel the backfill.", "Clear the sensor.", "Add a pool.",
                            "Restart."],
             "next_step_evidence": [["E1"], ["E2"], ["E1"], []],
             "next_step_kinds": ["recovery", "recovery", "prevention", "recovery"],
             "next_step_support": ["incident_report", "general_guidance", "incident_report",
                                   "unsupported"],
             "unknowns": ["Whether the provider lowered the quota is not ruled out."]}
    script.append(body)
    at = run_app()
    at.chat_input[0].set_value(Q1).run()
    text = " ".join(m.value for m in at.markdown).replace("\\", "")
    assert "**Next steps: recovery**" in text and "**Next steps: prevention**" in text
    assert text.index("Cancel the backfill.") < text.index("Next steps: prevention") \
        < text.index("Add a pool.")
    assert text.count(":green-badge[From the incident report]") == 2
    assert text.count(":orange-badge[General guidance]") == 1
    assert "Unsupported: no evidence was cited for this step" in text
    assert "**Not established by the evidence**" in text
    assert "Whether the provider lowered the quota is not ruled out." in text


def test_an_older_api_without_step_labels_shows_one_list(api):
    _, script = api
    script.append(answered(Q1) | {"next_steps": ["Do it."], "next_step_evidence": [["E1"]]})
    at = run_app()
    at.chat_input[0].set_value(Q1).run()
    text = " ".join(m.value for m in at.markdown)
    assert "**Next steps**" in text and "recovery" not in text and not at.exception


def test_a_near_miss_answer_says_which_failure_and_the_real_similarity(api):
    _, script = api
    body = answered("Why did the loyalty join fail?")
    body["gate"] = {"best_similarity": 0.693, "min_similarity": 0.72, "passed": False,
                    "accepted_by": {"rule": "sole_matching_failure",
                                    "candidate": "run:customer_360/join_loyalty/2026-09-22"}}
    body["retrieval_info"] = {"conversation": {
        "status": "matched_in_index", "subjects": ["customer_360 / join_loyalty on 2026-09-22"],
        "subject": {"pipeline": "customer_360", "task": "join_loyalty", "date": "2026-09-22"}}}
    script.append(body)
    at = run_app()
    at.chat_input[0].set_value("Why did the loyalty join fail?").run()
    caption = " ".join(c.value for c in at.caption).replace("\\", "")
    assert "only indexed failure this question matches" in caption
    assert "customer_360 / join_loyalty on 2026-09-22" in caption and "0.693" in caption
    # A follow-up keeps that failure as its subject.
    at.chat_input[0].set_value("How do we prevent it?").run()
    calls, _ = api
    assert calls["ask"][1]["history"][0]["subject"] == {
        "pipeline": "customer_360", "task": "join_loyalty", "date": "2026-09-22"}
