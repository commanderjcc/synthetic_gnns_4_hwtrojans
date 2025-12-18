#!/usr/bin/env python3
"""Plot facet grids from combined results CSV.

This script expects `combined_results.csv` to include a `source_file` column
containing the original per-run CSV filename (so we can extract the hidden size).
If `source_file` is missing, re-run the concat step with `--add-source`.

Produces:
  - plots/combined_graph_roc_auc.png
  - plots/combined_node_roc_auc.png

Usage:
  python scripts/plot_combined.py --csv results/combined_results.csv --out plots
"""
import argparse
import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def extract_hidden_from_source(fname: str) -> int | None:
    # Try patterns like _h96_ or _h96- etc. Fallback to any `h<number>` occurrence.
    m = re.search(r"_h(\d+)_", fname)
    if m:
        return int(m.group(1))
    m = re.search(r"_h(\d+)(?:\.|-|$)", fname)
    if m:
        return int(m.group(1))
    m = re.search(r"h(\d+)", fname)
    if m:
        return int(m.group(1))
    return None


def make_facets(df: pd.DataFrame, metric: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # Ensure hidden is categorical (sorted)
    df["hidden"] = df["hidden"].astype(int)
    df["hidden_cat"] = df["hidden"].astype(str)
    df["ratio_str"] = df["trojan_ratio"].astype(str)

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df,
        x="hidden_cat",
        y=metric,
        row="model",
        col="ratio_str",
        kind="bar",
        sharey=False,
        height=3.0,
        aspect=1.2,
        ci="sd",
    )
    g.set_axis_labels("hidden size", metric)
    g.fig.suptitle(f"{metric} by model, ratio, and hidden size", y=1.02)
    out_file = os.path.join(out_dir, f"combined_{metric}.png")
    plt.tight_layout()
    g.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")


def make_combined_facets(df: pd.DataFrame, metrics: list, out_dir: str, prefix: str = "graph"):
    os.makedirs(out_dir, exist_ok=True)
    df["hidden"] = df["hidden"].astype(int)
    df["hidden_cat"] = df["hidden"].astype(str)
    df["ratio_str"] = df["trojan_ratio"].astype(str)

    melt = df.melt(id_vars=["model", "ratio_str", "hidden_cat"], value_vars=metrics, var_name="metric", value_name="value")

    sns.set_theme(style="whitegrid")
    # point plot with hue=metric to compare multiple metrics on same axes
    g = sns.catplot(
        data=melt,
        x="hidden_cat",
        y="value",
        row="model",
        col="ratio_str",
        hue="metric",
        kind="point",
        sharey=False,
        height=3.0,
        aspect=1.2,
        dodge=True,
        ci="sd",
    )
    g.set_axis_labels("hidden size", ",".join(metrics))
    g.fig.suptitle(f"{prefix} metrics by model, ratio, and hidden size", y=1.02)
    out_file = os.path.join(out_dir, f"combined_{prefix}_metrics.png")
    plt.tight_layout()
    g.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/combined_results.csv")
    parser.add_argument("--out", type=str, default="plots")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "source_file" not in df.columns:
        print("Error: combined CSV missing 'source_file' column. Re-run concat with --add-source.")
        return

    # extract hidden from source_file
    df["hidden"] = df["source_file"].apply(lambda s: extract_hidden_from_source(str(s)) )
    if df["hidden"].isnull().any():
        missing = df[df["hidden"].isnull()]["source_file"].unique()[:10]
        print(f"Warning: could not extract hidden size from some source filenames (examples): {missing}")
        # drop rows without hidden
        df = df.dropna(subset=["hidden"]).copy()

    # make facet plots for graph_roc_auc and node_roc_auc
    make_facets(df, "graph_roc_auc", args.out)
    make_facets(df, "node_roc_auc", args.out)

    # Combined metric plots (overlay multiple metrics for easier comparison)
    graph_metrics = ["graph_ap", "graph_acc", "graph_f1", "graph_recall", "graph_precision", "graph_roc_auc"]
    node_metrics = ["node_ap", "node_acc", "node_f1", "node_recall", "node_precision", "node_roc_auc"]
    # filter metrics that actually exist in the dataframe
    graph_metrics = [m for m in graph_metrics if m in df.columns]
    node_metrics = [m for m in node_metrics if m in df.columns]
    if graph_metrics:
        make_combined_facets(df, graph_metrics, args.out, prefix="graph")
    if node_metrics:
        make_combined_facets(df, node_metrics, args.out, prefix="node")


if __name__ == "__main__":
    main()
