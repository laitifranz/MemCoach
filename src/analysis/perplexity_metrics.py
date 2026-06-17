"""Compute perplexity metrics from perplexity evaluation dataset.jsonl files.

Usage:
    uv run src/analysis/perplexity_metrics.py
    uv run src/analysis/perplexity_metrics.py --run-scope latest
    uv run src/analysis/perplexity_metrics.py --run-scope all --sort-by ppl --descending
    uv run src/analysis/perplexity_metrics.py --root experiments/evaluation/perplexity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

METHOD_DISPLAY_NAMES = {
    "baseline_flux": "Edit model",
    "teacher_oracle": "Teacher oracle",
    "zero_shot": "Zero-shot",
    "memcoach": "MemCoach (ours)",
}

METHOD_DISPLAY_ORDER = [
    "baseline_flux",
    "teacher_oracle",
    "zero_shot",
    "memcoach",
]

METHOD_SORT_RANK = {method: idx for idx, method in enumerate(METHOD_DISPLAY_ORDER)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze perplexity dataset.jsonl files and report PPL stats per method."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/evaluation/perplexity"),
        help="Root folder containing perplexity experiment runs.",
    )
    parser.add_argument(
        "--run-scope",
        choices=["latest", "all"],
        default="latest",
        help="Use only latest run per method or aggregate all runs.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["method", "ppl", "total_examples"],
        default="method",
        help="Sort output table by this column.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def discover_datasets(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for dataset_path in sorted(root.rglob("dataset.jsonl")):
        rel = dataset_path.relative_to(root)
        parts = rel.parts

        if len(parts) < 2:
            continue

        method = parts[0]
        run_id = parts[1] if len(parts) >= 3 else "default"

        with dataset_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                perplexity = _safe_float(item.get("perplexity"))
                if perplexity is None:
                    continue

                rows.append(
                    {
                        "method": method,
                        "run_id": run_id,
                        "dataset_path": str(dataset_path),
                        "line_num": line_num,
                        "perplexity": perplexity,
                    }
                )

    return pd.DataFrame(rows)


def select_run_scope(df: pd.DataFrame, run_scope: str) -> pd.DataFrame:
    if df.empty or run_scope == "all":
        return df

    latest_run_per_method = df.groupby("method")["run_id"].max()
    selected = (
        df.merge(
            latest_run_per_method.rename("selected_run_id"),
            left_on="method",
            right_index=True,
            how="inner",
        )
        .query("run_id == selected_run_id")
        .drop(columns=["selected_run_id"])
    )
    return selected


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "MethodId",
                "Method",
                "Runs",
                "TotalExamples",
                "MeanPerplexity",
                "StdPerplexity",
                "MinPerplexity",
                "MaxPerplexity",
            ]
        )

    grouped = df.groupby("method", as_index=False).agg(
        Runs=("run_id", "nunique"),
        TotalExamples=("perplexity", "size"),
        MeanPerplexity=("perplexity", "mean"),
        StdPerplexity=("perplexity", "std"),
        MinPerplexity=("perplexity", "min"),
        MaxPerplexity=("perplexity", "max"),
    )

    grouped["MethodId"] = grouped["method"]
    grouped["Method"] = (
        grouped["method"].map(METHOD_DISPLAY_NAMES).fillna(grouped["method"])
    )
    grouped = grouped.drop(columns=["method"])
    return grouped[
        [
            "MethodId",
            "Method",
            "Runs",
            "TotalExamples",
            "MeanPerplexity",
            "StdPerplexity",
            "MinPerplexity",
            "MaxPerplexity",
        ]
    ]


def sort_summary(df: pd.DataFrame, sort_by: str, descending: bool) -> pd.DataFrame:
    if df.empty:
        return df

    if sort_by == "method":
        sort_rank_default = len(METHOD_SORT_RANK)
        return (
            df.assign(
                _method_sort=df["MethodId"]
                .map(METHOD_SORT_RANK)
                .fillna(sort_rank_default)
            )
            .sort_values(by="_method_sort", ascending=not descending)
            .drop(columns=["_method_sort"])
            .reset_index(drop=True)
        )

    column_map = {
        "ppl": "MeanPerplexity",
        "total_examples": "TotalExamples",
    }
    sort_col = column_map[sort_by]
    return df.sort_values(by=sort_col, ascending=not descending).reset_index(drop=True)


def print_summary(df: pd.DataFrame, run_scope: str, root: Path) -> None:
    print(f"Root: {root}")
    print(f"Run scope: {run_scope}")
    print()

    if df.empty:
        print("No valid dataset rows found.")
        return

    display_df = df.copy()
    if "MethodId" in display_df.columns:
        display_df = display_df.drop(columns=["MethodId"])

    for col in ["MeanPerplexity", "StdPerplexity", "MinPerplexity", "MaxPerplexity"]:
        display_df[col] = display_df[col].map(
            lambda v: "n/a" if pd.isna(v) else f"{v:.2f}"
        )

    print(display_df.to_string(index=False))


def main() -> None:
    args = parse_args()

    raw_df = discover_datasets(args.root)
    scoped_df = select_run_scope(raw_df, args.run_scope)
    summary_df = summarize(scoped_df)
    summary_df = sort_summary(summary_df, args.sort_by, args.descending)
    print_summary(summary_df, args.run_scope, args.root)


if __name__ == "__main__":
    main()
