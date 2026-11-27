"""Compare result CSV files produced by the experiment entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        row = {"experiment": path.stem, "rows": len(frame)}
        for column in ("val_acc", "accuracy", "val_loss", "loss"):
            if column in frame:
                row[f"final_{column}"] = frame[column].iloc[-1]
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize experiment CSV outputs")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.results_dir)
    print(summary.to_string(index=False) if not summary.empty else "No result CSV files found.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
