"""Streamlit chat front end for the HTTP API (`app.py`).

The UI is a thin client: every question goes to POST /ask, so retrieval, the refusal gate,
the LLM call and citation validation all stay in the API. Nothing here imports the
retrieval or generation layers, and the Streamlit process never calls Gemini.

  client.py  HTTP calls to /health/db and /ask, with timeouts and API errors as values
  view.py    pure helpers that turn an /ask response into what the page shows (tested)
  app.py     the Streamlit page:  python -m streamlit run src/incident_copilot/ui/app.py
"""
