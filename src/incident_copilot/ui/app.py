"""Causora UI entry point. Run from the repository root:

    python -m streamlit run src/incident_copilot/ui/app.py --server.address 127.0.0.1

Two pages, linked in the sidebar under the Causora brand: Chat (ui/chat.py) and Knowledge
Base (ui/knowledge.py). Every answer, document and explanation comes from the API at
UI_API_URL; the UI never reads the database, the data directory or the Gemini key.

The brand type is Fredoka (rounded, bold), loaded from Google Fonts by the browser. Nothing
depends on it: without the font service the brand falls back to other rounded or bold system
fonts, and every other text keeps the theme's font.
"""

import streamlit as st

from incident_copilot.config import get_settings
from incident_copilot.ui import chat, knowledge
from incident_copilot.ui.client import CausoraClient, HealthResult

# Each page sets its own layout (below), so a run never switches it twice; calls are additive.
st.set_page_config(page_title="Causora", page_icon=":material/troubleshoot:")

_settings = get_settings()
client = CausoraClient(_settings.ui_api_url, _settings.ui_request_timeout_seconds)

# Colours, fonts and radii come from .streamlit/config.toml. This block only does what the
# theme cannot: the mascot avatar, the "thinking" bob of the avatar of the answer that is
# being waited for (a container keyed `causora_pending`, which exists only while a request
# runs, so the animation stops on every outcome without extra state), a new answer's
# speaking frames, user messages on the right, and the clarification choices (mauve
# #8E6A8B when picked: 4.3:1 on the page, above the 3:1 needed for controls). Under
# prefers-reduced-motion nothing moves. Also the brand (sidebar and chat heading), and the
# Knowledge Base cards and document viewer (ui/knowledge.py).
_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Fredoka:wght@600;700&display=swap");
:root {
  --causora-brand-font: "Fredoka", "Baloo 2", "Nunito", ui-rounded, "Arial Rounded MT Bold",
                        "Segoe UI Black", "Trebuchet MS", system-ui, sans-serif;
}

/* ---- brand ---- */
.causora-brand {
  font-family: var(--causora-brand-font); font-weight: 700; font-size: 2.35rem;
  line-height: 1.05; letter-spacing: 0.01em; color: #301B3F; margin: 0.15rem 0 0.1rem;
}
.causora-brand .dot { color: #8E6A8B; }
.causora-tagline { font-size: 0.8rem; color: #3C415C; margin: 0 0 0.6rem; }
.st-key-causora_hero h1 {
  font-family: var(--causora-brand-font) !important; font-weight: 700 !important;
  font-size: clamp(2.8rem, 8vw, 4.4rem) !important; line-height: 1.02 !important;
  letter-spacing: 0.01em; color: #301B3F; padding: 0.2rem 0 0.4rem !important;
  overflow-wrap: anywhere;
}
.st-key-causora_nav [data-testid="stPageLink"] a { padding-top: 0.35rem;
                                                   padding-bottom: 0.35rem; }
.st-key-causora_health [data-testid="stCaptionContainer"] { font-size: 0.78rem; }

/* ---- Knowledge Base cards: the transparent button covers the whole card ---- */
.st-key-kb_cards { display: grid !important; gap: 0.6rem !important;
                   grid-template-columns: repeat(auto-fill, minmax(17rem, 1fr)); }
.st-key-kb_cards > div { width: auto !important; height: 100%; }
[class*="st-key-kbcard_"] { position: relative; background: #FFFFFF; height: 100%;
                            transition: border-color 0.15s, box-shadow 0.15s; }
[class*="st-key-kbcard_"]:hover { border-color: #8E6A8B !important;
                                  box-shadow: 0 2px 10px rgba(48, 27, 63, 0.10); }
[class*="st-key-kbcard_"] [data-testid="stElementContainer"]:has([data-testid="stButton"]) {
  position: absolute !important; inset: 0; margin: 0; z-index: 1; height: auto !important;
}
[class*="st-key-kbcard_"] [data-testid="stButton"],
[class*="st-key-kbcard_"] [data-testid="stButton"] button {
  width: 100% !important; height: 100% !important;
}
[class*="st-key-kbcard_"] [data-testid="stButton"] button { opacity: 0; cursor: pointer; }
[class*="st-key-kbcard_"]:has(button:focus-visible) { outline: 2px solid #8E6A8B;
                                                      outline-offset: 2px; }

/* ---- document viewer (st.dialog, width "large"): ~90% of the viewport, own scrolling ----
   stDialog is the dimmed backdrop; its child div is the panel; the panes scroll, the panel
   does not, so the header and the dialog's X never scroll away. */
[data-testid="stDialog"] > div:has(.st-key-kb_viewer_head) {
  width: 90vw !important; max-width: 90vw !important; height: 88vh; max-height: 88vh;
  border-radius: 0.9rem; box-shadow: 0 18px 60px rgba(21, 21, 21, 0.28);
}
/* The dialog's own heading stays for screen readers only; the viewer's header row (Explain
   with AI | title) takes its place, beside the dialog's X (absolutely placed, top right). */
section[role="dialog"]:has(.st-key-kb_viewer_head) > h2 {
  position: absolute !important; width: 1px; height: 1px; padding: 0 !important;
  margin: -1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0;
}
section[role="dialog"]:has(.st-key-kb_viewer_head) > button[aria-label="Close"] {
  z-index: 3;
}
.st-key-kb_viewer_head { position: sticky; top: 0; z-index: 2; background: #FAF8F6;
                         padding: 0.2rem 2.6rem 0.6rem 0; border-bottom: 1px solid #D9D0CE; }
.st-key-kb_viewer_head > div:has(> .st-key-kb_viewer_title) { flex: 1 1 16rem !important;
                                                              min-width: 0; }
.st-key-kb_viewer_title { min-width: 0; }
/* Streamlit pulls the next element up under a heading (-1rem); here that hid the caption. */
.st-key-kb_viewer_title [data-testid="stHeading"] [data-testid="stMarkdownContainer"] {
  margin-bottom: 0 !important;
}
.st-key-kb_ai_pane h4 { font-size: 1.15rem !important; }
.st-key-kb_viewer_title h3 { padding: 0 0 0.15rem !important; margin: 0 !important;
                             line-height: 1.3 !important; overflow-wrap: anywhere; }
.st-key-kb_viewer_body { align-items: stretch; flex-wrap: nowrap !important; }
.st-key-kb_doc_pane, .st-key-kb_ai_pane {
  max-height: calc(88vh - 10.5rem); overflow-y: auto; overscroll-behavior: contain;
}
/* The flex items of a horizontal container are Streamlit's layout wrappers of the panes. */
.st-key-kb_viewer_body > div:has(> .st-key-kb_doc_pane) { flex: 1 1 0 !important;
                                                          min-width: 0; }
.st-key-kb_viewer_body > div:has(> .st-key-kb_ai_pane) { flex: 0 0 38% !important;
                                                         min-width: 0; }
.st-key-kb_doc_pane { min-width: 0; padding-right: 0.4rem; }
.st-key-kb_ai_pane { min-width: 0; background: #FFFFFF; }
.st-key-kb_doc_text table { display: block; overflow-x: auto; }
@media (max-width: 900px) {  /* the explanation goes below the document; both still scroll */
  [data-testid="stDialog"] > div:has(.st-key-kb_viewer_head) {
    width: 96vw !important; max-width: 96vw !important; height: 92vh; max-height: 92vh;
  }
  .st-key-kb_viewer_head { flex-wrap: wrap !important; }
  .st-key-kb_viewer_body { flex-wrap: wrap !important; }
  .st-key-kb_viewer_body > div:has(> .st-key-kb_doc_pane),
  .st-key-kb_viewer_body > div:has(> .st-key-kb_ai_pane) { flex: 1 1 100% !important; }
  .st-key-kb_doc_pane, .st-key-kb_ai_pane { max-height: calc(92vh - 13rem); }
  .st-key-kb_viewer_body:has(.st-key-kb_ai_pane) .st-key-kb_doc_pane,
  .st-key-kb_ai_pane { max-height: calc((92vh - 14rem) / 2); }
}
[data-testid="stChatMessageAvatarImage"] { background: transparent; border-radius: 0; }
.st-key-causora_pending [data-testid="stChatMessageAvatarImage"] {
  animation: causora-bob 2.4s ease-in-out infinite;
}
@keyframes causora-bob { 0%, 100% { transform: translateY(0); }
                         50% { transform: translateY(-3px); } }
/* A new answer speaks: its avatar is a sprite (open | closed mouth); chat.py adds the rule
   for that answer only while it speaks, so this just alternates the two halves. */
@keyframes causora-speak { 0% { object-position: 0% 0; } 50% { object-position: 100% 0; } }

/* The user's messages on the right (frog), Causora's on the left (cat). */
[class*="st-key-umsg_"] [data-testid="stChatMessage"] {
  flex-direction: row-reverse; margin-left: auto; width: fit-content; max-width: min(85%, 44rem);
}

/* Clarification choices. Each row (option + caption) is a bordered box; the label's ::after
   covers the whole row, so a click anywhere on it picks it; the picked row and its circle
   turn mauve; keyboard focus shows a mauve ring. */
[class*="st-key-clarify_"] [data-testid="stRadioGroup"] { width: 100%; gap: 0.4rem; }
[class*="st-key-clarify_"] [data-testid="stRadioGroup"] > div {
  position: relative; box-sizing: border-box; width: 100%; padding: 0.55rem 0.75rem;
  border: 1px solid #D9D0CE; border-radius: 0.5rem; background: #FFFFFF;
}
[class*="st-key-clarify_"] [data-testid="stRadioGroup"] > div:hover { border-color: #8E6A8B; }
[class*="st-key-clarify_"] [data-testid="stRadioOption"] { position: static; }
[class*="st-key-clarify_"] [data-testid="stRadioOption"]::after {
  content: ""; position: absolute; inset: 0; z-index: 1; cursor: pointer;
}
[class*="st-key-clarify_"] [data-testid="stRadioCaption"] { pointer-events: none; }
[class*="st-key-clarify_"] [data-testid="stRadioGroup"] > div:has(input:checked) {
  border-color: #8E6A8B; background: #F5EEF4;
}
[class*="st-key-clarify_"] [data-testid="stRadioGroup"] > div:has(input:checked)
  [data-testid="stRadioOption"] > div > div:first-child {
  background-color: #8E6A8B !important; border-color: #8E6A8B !important;
}
[class*="st-key-clarify_"] [data-testid="stRadioGroup"] > div:has(input:focus-visible) {
  outline: 2px solid #8E6A8B; outline-offset: 2px;
}

@media (prefers-reduced-motion: reduce) {
  .st-key-causora_pending [data-testid="stChatMessageAvatarImage"] { animation: none; }
  [class*="st-key-amsg_"] [data-testid="stChatMessageAvatarImage"] { animation: none !important; }
}
</style>
"""
st.html(_CSS)


@st.cache_data(ttl=30, show_spinner=False)
def _health(base_url: str) -> HealthResult:
    return CausoraClient(base_url).health()


def _brand(pages: list) -> None:
    """The sidebar top: the Causora brand, then the page links."""
    with st.sidebar:
        st.html('<p class="causora-brand">Causora<span class="dot">.</span></p>'
                '<p class="causora-tagline">Incident investigation for data pipelines</p>')
        with st.container(key="causora_nav", gap=None):
            for p in pages:
                st.page_link(p)


def _connection() -> None:
    """API and database status: always shown, but small. A problem still stands out."""
    with st.sidebar, st.container(key="causora_health", gap="small"):
        health = _health(client.base_url)
        if health.db_ok:
            st.caption(f":green[:material/check_circle:] API and database reachable · "
                       f"{health.documents} documents, {health.chunks} chunks")
        elif health.api_reachable:
            st.warning(f"API reachable, database not: {health.message}",
                       icon=":material/warning:")
        else:
            st.error(f"API not reachable: {health.message}", icon=":material/error:")
        st.caption(f"API: `{client.base_url}` · Answers come only from the indexed task "
                   "logs, incident reports, troubleshooting guides and table schemas, with "
                   "citations; when the evidence is too weak, Causora says so.")
        if st.button("Check again", icon=":material/refresh:", type="tertiary"):
            _health.clear()
            st.rerun()


def _chat_page() -> None:
    st.set_page_config(layout="centered")  # readable line length for answers
    chat.render(client)


def _knowledge_page() -> None:
    st.set_page_config(layout="wide")  # room for the document table
    knowledge.render(client)


PAGES = [
    st.Page(_chat_page, title="Chat", icon=":material/forum:", url_path="chat", default=True),
    st.Page(_knowledge_page, title="Knowledge Base", icon=":material/menu_book:",
            url_path="knowledge-base"),
]
# The built-in menu is hidden so that the brand can sit above the links (_brand).
page = st.navigation(PAGES, position="hidden")
_brand(PAGES)
_connection()
page.run()
