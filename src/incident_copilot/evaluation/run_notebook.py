"""Execute a notebook cell by cell, saving it after every cell, with a memory guard.

    python -m incident_copilot.evaluation.run_notebook notebooks/incident_copilot_evaluation.ipynb

Unlike `jupyter nbconvert --execute --inplace`, which writes outputs only at the very end, an
interrupted run keeps the outputs of every finished cell. Before each cell the available RAM
is checked; below --min-free-mb the run stops (exit code 3) instead of risking the machine.
It never retries. Exit codes: 0 done, 1 a cell raised, 3 stopped for low memory.

The kernel is started with this interpreter's directory first on PATH, so the `python3`
kernel runs this virtualenv even when a user-level kernelspec just says `python`.
Needs requirements-notebook.txt (nbclient, ipykernel, psutil comes with them).
"""

import argparse
import os
import sys
import time
from pathlib import Path


def _available_mb() -> float:
    import psutil

    return psutil.virtual_memory().available / 2**20


def run(path: Path, *, timeout: int = 1800, min_free_mb: float = 1500,
        kernel_name: str = "python3", log=print) -> int:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:  # never mix stale outputs with this run's
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    nbformat.write(nb, path)
    client = NotebookClient(nb, timeout=timeout, kernel_name=kernel_name,
                            resources={"metadata": {"path": str(path.parent)}})
    code_cells = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    started = time.perf_counter()
    with client.setup_kernel():
        for n, index in enumerate(code_cells, start=1):
            free = _available_mb()
            if free < min_free_mb:
                nbformat.write(nb, path)
                log(f"STOPPED before code cell {n}/{len(code_cells)}: {free:.0f} MB available "
                    f"< {min_free_mb:.0f} MB. Outputs of finished cells are saved. Not retrying.")
                return 3
            t0 = time.perf_counter()
            try:
                client.execute_cell(nb.cells[index], index)
            except CellExecutionError as exc:
                nbformat.write(nb, path)
                first = str(exc).strip().splitlines()[-1] if str(exc).strip() else ""
                log(f"FAILED in code cell {n}/{len(code_cells)} (notebook cell {index}): "
                    f"{first[:300]}")
                return 1
            nbformat.write(nb, path)
            log(f"cell {n:>2}/{len(code_cells)} done in {time.perf_counter() - t0:6.1f} s "
                f"(total {time.perf_counter() - started:6.0f} s, {free:,.0f} MB free before)")
    log(f"DONE: {len(code_cells)} code cells in {time.perf_counter() - started:.0f} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m incident_copilot.evaluation.run_notebook")
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per cell")
    parser.add_argument("--min-free-mb", type=float, default=1500)
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args(argv)
    return run(args.notebook, timeout=args.timeout, min_free_mb=args.min_free_mb,
               kernel_name=args.kernel, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    raise SystemExit(main())
