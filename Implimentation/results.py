#!/usr/bin/env python3
"""
results_by_dataset.py
---------------------
Creates per‑dataset‑and‑lambda plots from CSV traces produced by the C
experiments.  Every dataset + (λ₁, λ₂) tuple gets its own folder that
contains

• objective_vs_iter.png – objective f(x) over iterations, all algos
• objective_vs_time.png – objective f(x) over wall‑clock time, all algos
• iter_time_boxplot.png – per‑iteration time distribution, one box per algo
• totals.txt – one‑line summary with total runtime per algorithm

Usage
-----
$ python results_by_dataset.py --results_dir results --out_dir plots

Optional switches:
  --results_dir   Where the *.csv files live  [default: results]
  --out_dir       Where to create sub‑folders [default: plots]
  --max_iter      Truncate lines after this   [default: keep all]
"""

from __future__ import annotations

import argparse, os, glob, re, textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Regex for a file name like: ista_dataset_L1_L2.csv
#  – algo    : anything up to first underscore
#  – dataset : anything up to next underscore (may itself contain dashes etc.)
#  – l1      : next field
#  – l2      : last field (before .csv)
# Example: lbfgs_fista_housing_0_0.1.csv  → algo=lbfgs_fista, dataset=housing, l1=0, l2=0.1
_NAME_RE = re.compile(r"^(?P<algo>[^_]+)_(?P<dataset>[^_]+)_(?P<l1>[^_]+)_(?P<l2>[^.]+)\.csv$")


@dataclass
class TraceMeta:
    algo: str
    dataset: str
    l1: str
    l2: str
    path: Path


# -----------------------------------------------------------------------------

def _parse_csv_name(path: Path) -> TraceMeta | None:
    m = _NAME_RE.match(path.name)
    if m:
        return TraceMeta(**m.groupdict(), path=path)
    return None


def _grouper(traces: list[TraceMeta]):
    """Yield (dataset‑λ key, list[TraceMeta])."""
    groups: dict[str, list[TraceMeta]] = {}
    for t in traces:
        key = f"{t.dataset}_l1={t.l1}_l2={t.l2}"
        groups.setdefault(key, []).append(t)
    return groups.items()


# -----------------------------------------------------------------------------

def load_traces(meta_list: list[TraceMeta]) -> pd.DataFrame:
    frames = []
    for meta in meta_list:
        df = pd.read_csv(meta.path)
        df['algo'] = meta.algo
        frames.append(df)
    if not frames:
        raise RuntimeError("No CSVs to load for group")
    return pd.concat(frames, ignore_index=True)


def line_plot(df: pd.DataFrame, y: str, out_file: Path, *, x: str = "Iteration", ylab: str | None = None, max_iter: int | None = None, logy: bool = True):
    if max_iter is not None and x == "Iteration":
        df = df[df[x] <= max_iter]

    plt.figure()
    for algo, grp in df.groupby("algo"):
        plt.plot(grp[x], grp[y], label=algo, linewidth=1.3)
    if logy:
        plt.yscale("log")
    plt.xlabel(x.lower())
    plt.ylabel(ylab or y)
    plt.title(f"{y} vs {x.lower()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def box_plot_times(df: pd.DataFrame, out_file: Path):
    """Box‑plot of *per‑iteration* times, plus red diamonds at maxima and
    text for averages."""
    plt.figure()
    data, labels = [], []
    max_vals, mean_vals = [], []
    for algo, grp in df.groupby("algo"):
        it_times = np.diff(grp["time"].to_numpy(), prepend=0.0)
        data.append(it_times)
        labels.append(algo)
        max_vals.append(it_times.max())
        mean_vals.append(it_times.mean())

    bplots = plt.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    # mark max and annotate mean
    for i, (maxv, meanv) in enumerate(zip(max_vals, mean_vals), start=1):
        plt.plot(i, maxv, marker="D", color="red")
        plt.text(i, meanv, f"{meanv:.4f}s", ha="center", va="bottom", fontsize=8)

    plt.ylabel("time per iteration (s)")
    plt.title("Per‑iteration wall‑clock time")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def write_totals(df: pd.DataFrame, out_file: Path):
    totals = df.groupby("algo")["time"].max().sort_values()
    with open(out_file, "w", encoding="utf-8") as fh:
        for algo, t in totals.items():
            fh.write(f"{algo:12s}: {t:.6f} s\n")


# -----------------------------------------------------------------------------

def process_group(meta_list: list[TraceMeta], out_dir: Path, max_iter: int | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_traces(meta_list)

    # objective vs iteration
    line_plot(df, "objective", out_dir / "objective_vs_iter.png", ylab="objective f(x)", max_iter=max_iter)
    # objective vs time
    line_plot(df, "objective", out_dir / "objective_vs_time.png", ylab="objective f(x)", x="time", logy=True)
    # per‑iteration times
    box_plot_times(df, out_dir / "iter_time_boxplot.png")
    # totals
    write_totals(df, out_dir / "totals.txt")


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results", help="directory containing CSV traces")
    ap.add_argument("--out_dir", default="plots", help="where to put plot folders")
    ap.add_argument("--max_iter", type=int, default=None, help="truncate curves after this many iterations")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    traces = [_parse_csv_name(Path(p)) for p in glob.glob(str(results_dir / "*.csv"))]
    traces = [t for t in traces if t is not None]
    if not traces:
        raise FileNotFoundError(f"No matching CSV files found in {results_dir}")

    for key, metas in _grouper(traces):
        print(f"Processing {key} … ({len(metas)} files)")
        process_group(metas, out_root / key, args.max_iter)

    print("✓ All plots written to", out_root)


if __name__ == "__main__":
    from dataclasses import dataclass

    main()