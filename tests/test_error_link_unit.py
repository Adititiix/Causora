"""ErrorLinkRetriever.find_links without a database: the vector search is a fake."""

from fakes import CountingEmbedder
from incident_copilot.retrieval import ErrorLinkRetriever, Filters
from incident_copilot.retrieval.base import Hit


def _hit(path, source_type, error=None, rank=1):
    return Hit(chunk_id=path, document_id=path, path=path, source_type=source_type,
               section=None, ts_start=None, ts_end=None, text="", chunk_index=0,
               content_hash=None, pipeline_name="orders_etl", rank=rank,
               scores={"vector_similarity": 0.8, "vector_distance": 0.2,
                       "search_similarity": 0.9},
               metadata={"error": error} if error else {})


class FakeSearch:
    def __init__(self, results):
        self.results, self.calls = results, []

    def search(self, vec, k, filters, score_vec=None):
        self.calls.append(filters)
        return list(self.results)


def test_logs_with_the_same_error_line_are_embedded_and_searched_once():
    e = CountingEmbedder()
    r = ErrorLinkRetriever(e)
    r.vector = FakeSearch([_hit("runbooks/http_429.md", "runbook")])
    logs = [_hit("logs/a.log", "log", "HTTPError: 429", 1),
            _hit("logs/b.log", "log", "HTTPError: 429", 2)]
    links = r.find_links(logs, e.embed_query("q"), Filters())
    assert e.query_calls == 2  # the question, then one error line
    assert len(r.vector.calls) == 1
    # Same result as before: the runbook is linked once, from the first log.
    assert list(links) == ["logs/a.log"]
    assert links["logs/a.log"].linked_from == "logs/a.log"


def test_different_error_lines_are_each_searched():
    e = CountingEmbedder()
    r = ErrorLinkRetriever(e)
    r.vector = FakeSearch([_hit("runbooks/x.md", "runbook")])
    logs = [_hit("logs/a.log", "log", "KeyError: 'x'", 1),
            _hit("logs/b.log", "log", "HTTPError: 429", 2)]
    r.find_links(logs, e.embed_query("q"), Filters())
    assert e.query_calls == 3
    assert len(r.vector.calls) == 2
