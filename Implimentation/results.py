#!/usr/bin/env python3
"""
combined_plots.py
-----------------
• evolution of grad-norm vs iteration             (all algos on one plot)
• evolution of objective vs iteration             (all algos on one plot)
• per-iteration times: box-plot + max marker      (one box per algo)

Command-line switches
---------------------
--results_dir   directory containing *.csv         [default: results]
--out_dir       where to put the .png files        [default: plots]
--max_iter      cut lines at this iteration        [default: no cut]
"""

import argparse, os, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt

# ----------------------------------------------------------------------
def load_traces(results_dir: str) -> pd.DataFrame:
    frames = []
    for path in glob.glob(os.path.join(results_dir, "*.csv")):
        algo = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        df["algo"] = algo
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSV in {results_dir}")
    return pd.concat(frames, ignore_index=True)

# ----------------------------------------------------------------------
def line_plot(df, y, out_file, max_iter=None, ylab=None):
    if max_iter is not None:
        df = df[df["Iteration"] <= max_iter]

    plt.figure()
    for algo, grp in df.groupby("algo"):
        plt.plot(grp["Iteration"], grp[y], label=algo, linewidth=1.3)
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel(ylab or y)
    plt.title(f"{y} vs iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

# ----------------------------------------------------------------------
def box_plot_times(df, out_file):
    """Box-plot of *per-iteration* times, plus a red diamond at the max."""
    out = []
    for algo, grp in df.groupby("algo"):
        it_times = np.diff(grp["time"].to_numpy(), prepend=0.0)
        out.append((algo, it_times))

    plt.figure()
    data = [it_times for _, it_times in out]
    labels = [algo for algo, _ in out]
    bplots = plt.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    # mark the max of each algo
    for i, (_, it_times) in enumerate(out, start=1):
        plt.plot(i, it_times.max(), marker="D", color="red")
        # annotate the average just above the box
        avg = it_times.mean()
        plt.text(i, avg, f"{avg:.4f}s", ha="center", va="bottom", fontsize=8)

    plt.ylabel("time per iteration (s)")
    plt.title("Per-iteration wall-clock time")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir",     default="plots")
    ap.add_argument("--max_iter",    type=int, default=300,
                    help="truncate curves after this many iterations")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_traces(args.results_dir)

    line_plot(df, "gradnorm",
              os.path.join(args.out_dir, "gradnorm_vs_iter.png"),
              max_iter=args.max_iter, ylab="‖∇f(x)‖₂")

    line_plot(df, "objective",
              os.path.join(args.out_dir, "objective_vs_iter.png"),
              max_iter=args.max_iter, ylab="objective f(x)")

    box_plot_times(df,
                   os.path.join(args.out_dir, "iter_time_boxplot.png"))

    print("✓ graphs saved to", args.out_dir)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()