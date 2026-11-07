import subprocess
import sys
from collections import Counter

import pytest

from incident_copilot.ingestion import UnsafePathError, ingest_paths
from incident_copilot.ingestion.sources import classify, discover, is_under, normalize_root


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("runbooks/oom-killed.md", "runbook"),
        ("logs/orders_etl/2026-09-04__transform_orders.log", "log"),
        ("schemas/orders_raw.md", "schema"),
        ("incidents/INC-2031.md", "incident"),
        ("extra/runbooks/my-runbook.md", "runbook"),
        ("extra/logs/orders_etl/2026-10-01__x.log", "log"),
        ("extra/incidents/INC-9000.md", "incident"),
        ("manifest.json", None),
        ("runbooks/.gitkeep", None),
        ("extra/logs/.gitkeep", None),
        ("logs/orders_etl/notes.md", None),
        ("extra/other/x.md", None),
    ],
)
def test_classify(path, expected):
    assert classify(path) == expected


@pytest.mark.parametrize(
    "bad",
    ["../secrets", "runbooks/../../etc/passwd", "/etc/passwd", "C:\\Windows\\win.ini",
     "C:relative", "\\\\server\\share\\x", "\\rooted", "logs/..", ".."],
)
def test_unsafe_roots_are_rejected(tmp_path, bad):
    with pytest.raises(UnsafePathError):
        normalize_root(tmp_path, bad)


def test_safe_roots_are_normalized(tmp_path):
    assert normalize_root(tmp_path, ".") == "."
    assert normalize_root(tmp_path, "") == "."
    assert normalize_root(tmp_path, "runbooks") == "runbooks"
    assert normalize_root(tmp_path, "./logs/orders_etl/") == "logs/orders_etl"
    assert normalize_root(tmp_path, "logs\\orders_etl") == "logs/orders_etl"


def test_ingest_paths_rejects_traversal_before_touching_the_database(tmp_path):
    with pytest.raises(UnsafePathError):
        ingest_paths(["../outside"], embedder=None, data_dir=tmp_path)


def test_is_under():
    assert is_under("logs/a/x.log", ".")
    assert is_under("logs/a/x.log", "logs")
    assert is_under("logs/a/x.log", "logs/a/x.log")
    assert not is_under("logs2/x.log", "logs")


def test_discover_finds_the_whole_corpus_and_ignores_other_files(dataset):
    files = discover(dataset["root"], ["."])
    assert Counter(f.source_type for f in files) == {
        "runbook": 18, "log": 60, "schema": 12, "incident": 14,
    }
    assert all("manifest" not in f.path for f in files)
    assert [f.path for f in files] == sorted(f.path for f in files)


def test_discover_treats_extra_like_the_main_folders(dataset, tmp_path):
    (tmp_path / "extra" / "runbooks").mkdir(parents=True)
    (tmp_path / "extra" / "runbooks" / "mine.md").write_text("# Runbook: mine\n", "utf-8")
    (tmp_path / "extra" / "logs" / "p1").mkdir(parents=True)
    (tmp_path / "extra" / "logs" / "p1" / "2026-01-01__t.log").write_text("x", "utf-8")
    files = {f.path: f.source_type for f in discover(tmp_path, ["."])}
    assert files == {"extra/runbooks/mine.md": "runbook",
                     "extra/logs/p1/2026-01-01__t.log": "log"}
    assert [f.path for f in discover(tmp_path, ["extra/logs"])] == [
        "extra/logs/p1/2026-01-01__t.log"
    ]


def test_ingest_core_does_not_import_fastapi_or_cli():
    code = (
        "import sys, incident_copilot.ingestion.ingest; "
        "bad = [m for m in sys.modules if m.split('.')[0] in ('fastapi', 'starlette', 'uvicorn')"
        " or m in ('incident_copilot.ingestion.cli', 'incident_copilot.api.main')]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
