"""The incremental notebook runner: saves after each cell, stops on errors and low memory."""

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")
pytest.importorskip("psutil")

from incident_copilot.evaluation import run_notebook  # noqa: E402


def make(path, *sources):
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell("# t")] + [nbformat.v4.new_code_cell(s)
                                                       for s in sources]
    nbformat.write(nb, path)


def outputs(path):
    nb = nbformat.read(path, as_version=4)
    return [(c.execution_count, "".join(o.get("text", "") for o in c.outputs))
            for c in nb.cells if c.cell_type == "code"]


def test_runs_every_cell_and_saves_outputs(tmp_path):
    p = tmp_path / "ok.ipynb"
    make(p, "x = 20", "print(x + 22)")
    assert run_notebook.run(p, timeout=120, min_free_mb=0, log=lambda m: None) == 0
    assert outputs(p) == [(1, ""), (2, "42\n")]


def test_a_failing_cell_keeps_the_outputs_before_it(tmp_path):
    p = tmp_path / "fail.ipynb"
    make(p, "print('kept')", "raise ValueError('boom')", "print('never')")
    msgs = []
    assert run_notebook.run(p, timeout=120, min_free_mb=0, log=msgs.append) == 1
    got = outputs(p)
    assert got[0] == (1, "kept\n") and got[2] == (None, "")
    assert "boom" in msgs[-1]


def test_low_memory_stops_before_running_and_does_not_retry(tmp_path, monkeypatch):
    p = tmp_path / "mem.ipynb"
    make(p, "print('a')", "print('b')")
    free = iter([10_000, 100])  # enough for cell 1, then too little
    monkeypatch.setattr(run_notebook, "_available_mb", lambda: next(free))
    msgs = []
    assert run_notebook.run(p, timeout=120, min_free_mb=1500, log=msgs.append) == 3
    assert outputs(p) == [(1, "a\n"), (None, "")]
    assert msgs[-1].startswith("STOPPED before code cell 2/2")
