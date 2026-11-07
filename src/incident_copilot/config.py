"""Central configuration. Every setting comes from environment variables / .env.

Add new settings here as layers arrive; nothing else in the codebase should read
os.environ directly.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# chunks.embedding is vector(384) in db/init/001_schema.sql.
SCHEMA_EMBEDDING_DIM = 384


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://copilot:change-me-locally@localhost:5434/incident_copilot"
    log_level: str = "INFO"

    # Day 2
    # Root of the corpus; documents.path is stored relative to it.
    data_dir: str = "data"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    # Where fastembed keeps downloaded model files (a named volume in Docker).
    embedding_cache_dir: str = ".cache/fastembed"

    # Day 3
    gemini_api_key: str = ""
    llm_model: str = "gemini-3.8-flash"
    llm_timeout_seconds: float = 30.0
    # Retries for transient LLM failures (HTTP 408/429/500/502/503/504, network timeouts),
    # with jittered exponential backoff. Permanent errors (400, 401, 403, 404) never retry.
    llm_max_retries: int = 3
    # Upper bound for one generation, all attempts and backoff waits included. Each attempt
    # gets min(LLM_TIMEOUT_SECONDS, time left); no attempt starts with under 5 s left.
    llm_total_timeout_seconds: float = 60.0
    # Chunks retrieved and shown to the LLM per /ask (request `top_k` overrides it).
    ask_top_k: int = 8
    # A name from retrieval.RETRIEVERS ("vector", "vector+error_link", "vector+date",
    # "vector+error_link+date") or retrieval.EXPERIMENTAL_RETRIEVERS. The default (approved
    # 2026-10-29) is vector+error_link+date plus evidence selection (retrieval/evidence.py):
    # the matched runbook's Resolution section, a dated failed log's traceback, anchor-safe
    # links and, for dated-log questions, at most 2 chunks per document. On the 50 seed
    # questions it refused 0/44 answerable questions (vector+error_link+date: 12/44 and 15/44
    # in two runs, e.g. Q030 because the Resolution was never retrieved) and still refused 6/6
    # unanswerable ones. Set RETRIEVER=vector+error_link+date to go back.
    retriever: str = "vector+error_link+evidence+context+scoped+date"
    # Pre-LLM refusal gate on the best cosine similarity (to the question). Calibrated on the
    # 50 seed questions with bge-small-en-v1.5: highest blocked unanswerable 0.700, lowest
    # answerable 0.736; 0.72 is the midpoint. Re-run
    # `python -m incident_copilot.evaluation.retrieval_smoke` after changing model or data.
    min_similarity: float = 0.72
    # Clarification for vague questions (generation/clarify.py). A question refused only by
    # the gate above, that names no subject of its own, is answered with "which one do you
    # mean?" when 2..CLARIFY_MAX_CANDIDATES distinct failures (incidents or failed runs) have
    # evidence at or above CLARIFY_MIN_SIMILARITY. Never an answer: the user picks one and the
    # normal gate decides again. 0.67 keeps every gate-refused unanswerable seed and multi-doc
    # question refused (their best similarity is 0.700 at most, and those above 0.67 name an
    # unknown or explicit subject).
    clarify_min_similarity: float = 0.67
    clarify_max_candidates: int = 5

    # Streamlit UI (ui/app.py). It only calls this API; it never reads the Gemini key.
    ui_api_url: str = "http://127.0.0.1:8000"
    # Per-question wait for /ask; covers the LLM timeout plus its retries.
    ui_request_timeout_seconds: float = 120.0

    # Day 4-5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    abstain_threshold: float = 0.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
