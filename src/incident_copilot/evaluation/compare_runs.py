"""Compare two evaluation runs (folders written by the notebook) question by question.

    python -m incident_copilot.evaluation.compare_runs eval/results/<baseline> eval/results/<new>
        --pair "OLD_LABEL=NEW_LABEL"   (labels as in per_question_retrieval.csv)

Default pairs: every configuration present in both runs under the same label (did unchanged
code produce unchanged numbers?), plus the baseline's production default against the new
run's production default. For each pair it prints overall and per-category metrics with
deltas, every question whose result changed, and refusal-gate decisions that changed.
Writes compare_<baseline-run-id>.md and .csv into the NEW run's folder; the baseline folder
is only read.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = ["hit@1", "hit@3", "hit@5", "hit@8", "recall@5", "all@3", "all@5", "all@8", "rr"]
CHANGE_METRICS = ["hit@1", "hit@5", "all@5", "all@8", "rr"]


def load_run(folder: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame | None]:
    per = pd.read_csv(folder / "per_question_retrieval.csv")
    for col in ("expected_paths", "retrieved_paths"):
        per[col] = per[col].map(json.loads)
    meta = json.loads((folder / "run_metadata.json").read_text("utf-8"))
    gate_path = folder / "abstention_per_question.csv"
    gate = pd.read_csv(gate_path) if gate_path.exists() else None
    return per, meta, gate


def production_label(meta: dict) -> str:
    return (f"{meta['default_retriever']} | w{meta['log_error_context_entries']} "
            "(production index)")


def summarize(per: pd.DataFrame, label: str) -> pd.DataFrame:
    a = per[(per.config == label) & ~per.should_abstain]
    rows = [("ALL", a)] + [(c, g) for c, g in a.groupby("category")]
    out = pd.DataFrame({name: {"n": len(g), **g[METRICS].mean().to_dict()} for name, g in rows}).T
    return out.rename(columns={"rr": "MRR"})


def compare_pair(base: pd.DataFrame, new: pd.DataFrame, base_label: str,
                 new_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    b, n = summarize(base, base_label), summarize(new, new_label)
    table = pd.concat({"baseline": b, "new": n, "delta": n[b.columns] - b}, axis=1)
    bq = base[base.config == base_label].set_index("id")
    nq = new[new.config == new_label].set_index("id")
    changes = []
    for qid in bq.index.intersection(nq.index):
        if bq.loc[qid, "should_abstain"]:
            continue
        diffs = {m: (bq.loc[qid, m], nq.loc[qid, m]) for m in CHANGE_METRICS
                 if abs(float(nq.loc[qid, m]) - float(bq.loc[qid, m])) > 1e-9}
        if diffs:
            worse = any(after < before for before, after in diffs.values())
            better = any(after > before for before, after in diffs.values())
            changes.append({"id": qid, "category": bq.loc[qid, "category"],
                            "effect": "mixed" if worse and better else
                            "regressed" if worse else "improved",
                            **{m: f"{x:.3g} -> {y:.3g}" for m, (x, y) in diffs.items()}})
    return table, pd.DataFrame(changes)


def compare_gate(base_gate: pd.DataFrame | None, new_gate: pd.DataFrame | None) -> pd.DataFrame:
    if base_gate is None or new_gate is None:
        return pd.DataFrame()
    m = base_gate.merge(new_gate, on="id", suffixes=("_baseline", "_new"))
    return m[m.refused_baseline != m.refused_new][
        ["id", "category_baseline", "should_abstain_baseline", "best_similarity_baseline",
         "best_similarity_new", "refused_baseline", "refused_new"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m incident_copilot.evaluation.compare_runs")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--pair", action="append", default=[],
                        help="'BASELINE_LABEL=NEW_LABEL' (repeatable)")
    args = parser.parse_args(argv)

    base, base_meta, base_gate = load_run(args.baseline)
    new, new_meta, new_gate = load_run(args.new)
    pairs = [tuple(p.split("=", 1)) for p in args.pair]
    if not pairs:
        common = sorted(set(base.config) & set(new.config))
        pairs = [(c, c) for c in common]
        defaults = (production_label(base_meta), production_label(new_meta))
        if defaults not in pairs:
            pairs.append(defaults)

    pd.set_option("display.width", 200)
    md = [f"# Run comparison: {base_meta['run_id']} (baseline) -> {new_meta['run_id']} (new)\n"]
    csv_rows = []
    for base_label, new_label in pairs:
        if base_label not in set(base.config) or new_label not in set(new.config):
            print(f"skip: {base_label!r} -> {new_label!r} (not in both runs)")
            continue
        table, changes = compare_pair(base, new, base_label, new_label)
        title = f"{base_label}  ->  {new_label}"
        print("\n" + "=" * len(title) + f"\n{title}\n" + "=" * len(title))
        print(table.round(3).to_string())
        n_reg = int((changes.effect.isin(["regressed", "mixed"])).sum()) if len(changes) else 0
        n_imp = int((changes.effect == "improved").sum()) if len(changes) else 0
        print(f"\nper-question changes: {n_imp} improved, {n_reg} regressed or mixed")
        if len(changes):
            print(changes.fillna("").to_string(index=False))
        md += [f"\n## {title}\n", "```", table.round(3).to_string(), "```",
               f"\nPer-question changes: {n_imp} improved, {n_reg} regressed or mixed.\n"]
        if len(changes):
            md += ["```", changes.fillna("").to_string(index=False), "```"]
        flat = table.copy()
        flat.columns = [f"{a}:{b}" for a, b in flat.columns]
        csv_rows.append(flat.assign(baseline_config=base_label, new_config=new_label))

    gate = compare_gate(base_gate, new_gate)
    gate_line = (f"refusal decisions changed for {len(gate)} questions (baseline threshold "
                 f"{base_meta['min_similarity']}, new {new_meta['min_similarity']})")
    print("\n" + gate_line)
    if len(gate):
        print(gate.to_string(index=False))
    md += [f"\n## Refusal gate\n\n{gate_line}.\n"]
    if len(gate):
        md += ["```", gate.to_string(index=False), "```"]

    stem = f"compare_{base_meta['run_id']}"
    (args.new / f"{stem}.md").write_text("\n".join(md) + "\n", "utf-8")
    if csv_rows:
        pd.concat(csv_rows).to_csv(args.new / f"{stem}.csv")
    print(f"\nwrote {args.new / (stem + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
