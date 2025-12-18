#!/usr/bin/env python3
"""
Run a matrix of experiments over graph sizes and trojan ratios, training GCN/GAT/H2GNN
and saving a CSV report. Use ``--min-nodes`` and ``--max-nodes`` to set graph sizes.

Example (small smoke test):
  uv run python matrix_runner.py --num-graphs 200 --val-graphs 40 --test-graphs 40 --min-nodes 180 --max-nodes 220 --trojan-ratios 0.3,0.6 --epochs 1 --batch-size 8 --device cpu

For full runs set `--num-graphs 50000 --device cuda` (make sure PyTorch CUDA build matches GPU).
"""
import argparse
import csv
import os
import time
from itertools import product
from typing import List

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import random as pyrandom
import numpy as np

from data_gen import GenerationConfig, SyntheticTrojanDataset
from data_gen import PregeneratedDiskDataset
import threading
import queue
from models import GCNClassifier, GATClassifier, GraphSAGEClassifier, H2GNNClassifier
import train_and_eval as te


class PrefetchLoader:
    """Wrap a DataLoader and prefetch batches moved to `device` in a background thread.

    This expects the underlying DataLoader to yield objects that implement
    `.to(device)` (e.g., `torch_geometric.data.Batch`). The prefetch queue
    size is controlled by `prefetch_batches`.
    """

    def __init__(self, loader, device: torch.device, prefetch_batches: int = 2):
        self.loader = loader
        self.device = device
        self.prefetch = max(1, int(prefetch_batches))
        self._queue = queue.Queue(maxsize=self.prefetch)
        self._sentinel = object()
        self._thread = None
        self._exc = None

    def __iter__(self):
        self._it = iter(self.loader)
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()
        return self

    def _run(self):
        try:
            for batch in self._it:
                try:
                    batch = batch.to(self.device)
                except Exception:
                    # best-effort move; if it fails, put the raw batch
                    pass
                self._queue.put(batch)
        except Exception as e:
            self._exc = e
        finally:
            # signal end
            self._queue.put(self._sentinel)

    def __next__(self):
        item = self._queue.get()
        if item is self._sentinel:
            raise StopIteration
        if self._exc:
            # surface worker thread exception in main thread
            raise self._exc
        return item

    def __len__(self):
        try:
            return len(self.loader)
        except Exception:
            raise TypeError("underlying loader has no length")


def run_experiment(min_nodes: int, max_nodes: int, trojan_ratio: float, num_graphs: int, val_graphs: int, test_graphs: int,
                   batch_size: int, epochs: int, device: torch.device, dataset_mode: str, mix_prob: float,
                   max_struct_features_nodes: int, betweenness_k: int, seed: int = 42, pregen_dir: str | None = None,
                   model_name: str | None = None, hidden_size: int | None = None,
                   out_csv: str | None = None, max_steps: int | None = None, val_every_steps: int = 100):
    cfg = GenerationConfig(
        num_graphs=num_graphs,
        trojan_ratio=trojan_ratio,
        node_range=(min_nodes, max_nodes),
        trojan_mix_prob=mix_prob,
        seed=seed,
        max_struct_features_nodes=max_struct_features_nodes,
        betweenness_sample_k=betweenness_k,
    )

    # Create datasets (ondemand recommended for large num_graphs). New `disk` mode
    # loads pre-generated files from disk using workers and prefetching.
    if dataset_mode == "ondemand":
        train_ds = SyntheticTrojanDataset(cfg, length=num_graphs, seed_offset=0)
        val_ds = SyntheticTrojanDataset(cfg, length=val_graphs, seed_offset=1_000_000)
        test_ds = SyntheticTrojanDataset(cfg, length=test_graphs, seed_offset=2_000_000)
    elif dataset_mode == "pregen":
        # pregenerate via generate_dataset (keeps memory for smaller runs)
        from data_gen import generate_dataset

        graphs = generate_dataset(cfg)
        # simple split
        train_ds = graphs[: int(len(graphs) * 0.7)]
        val_ds = graphs[int(len(graphs) * 0.7) : int(len(graphs) * 0.8)]
        test_ds = graphs[int(len(graphs) * 0.8) :]
    elif dataset_mode == "disk":
        # Expect pre-generated .pt files under pregen_dir; dataset will lazily torch.load
        if pregen_dir is None:
            pregen_dir = os.environ.get("PREGENDIR", "pregen_data")
        train_ds = PregeneratedDiskDataset(os.path.join(pregen_dir, "train"))
        val_ds = PregeneratedDiskDataset(os.path.join(pregen_dir, "val"))
        test_ds = PregeneratedDiskDataset(os.path.join(pregen_dir, "test"))
    else:
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}")

    # Tune loader workers / pin_memory depending on mode and device
    num_workers = 0
    pin_memory = False
    if dataset_mode == "disk":
        num_workers = min(4, (os.cpu_count() or 4))
        pin_memory = True if device.type != "cpu" else False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # If disk mode + CUDA, wrap loaders with a prefetching loader that moves batches to device
    if dataset_mode == "disk" and device.type != "cpu":
        train_loader = PrefetchLoader(train_loader, device=device, prefetch_batches=2)
        val_loader = PrefetchLoader(val_loader, device=device, prefetch_batches=1)
        test_loader = PrefetchLoader(test_loader, device=device, prefetch_batches=1)

    # infer input dim from a sample
    sample = train_ds[0]
    in_dim = sample.x.size(-1)

    results = []

    # Build list of models to run (filter if a single model_name was provided)
    available = [
        ("H2GNN", H2GNNClassifier),
        ("GAT", GATClassifier),
        ("GCN", GCNClassifier),
        ("GraphSAGE", GraphSAGEClassifier),
    ]
    if model_name:
        desired = [m for m in available if m[0].lower() == model_name.lower()]
        if not desired:
            raise ValueError(f"Unknown model: {model_name}")
        run_models = desired
    else:
        run_models = available

    for model_name, ModelCls in tqdm(run_models, desc="models", leave=False):
        time.sleep(3) # brief pause between models to reduce GPU memory contention
        print(f"\nRunning {model_name} | nodes [{min_nodes},{max_nodes}] | trojan_ratio={trojan_ratio}")
        # Pass hidden size if provided
        if hidden_size is not None:
            model = ModelCls(in_channels=in_dim, hidden=hidden_size).to(device)
        else:
            model = ModelCls(in_channels=in_dim).to(device)
        start = time.time()
        hist = te.train(
            model,
            train_loader,
            val_loader,
            device,
            epochs=epochs,
            max_steps=max_steps,
            val_every_steps=val_every_steps,
            trojan_graph_ratio=trojan_ratio,
            node_trojan_frac_estimate=0.1,
        )
        elapsed = time.time() - start
        metrics = te.evaluate(model, test_loader, device)

        row = {
            "model": model_name,
            "min_nodes": min_nodes,
            "max_nodes": max_nodes,
            "trojan_ratio": trojan_ratio,
            "num_graphs": num_graphs,
            "batch_size": batch_size,
            "epochs": epochs,
            "train_time_s": round(elapsed, 2),
            "graph_bce": metrics["graph"]["bce"],
            "graph_roc_auc": metrics["graph"]["roc_auc"],
            "graph_ap": metrics["graph"]["ap"],
            "graph_acc": metrics["graph"]["acc"],
            "graph_f1": metrics["graph"]["f1"],
            "graph_recall": metrics["graph"]["recall"],
            "graph_precision": metrics["graph"]["precision"],
            "node_bce": metrics["node"]["bce"],
            "node_roc_auc": metrics["node"]["roc_auc"],
            "node_ap": metrics["node"]["ap"],
            "node_acc": metrics["node"]["acc"],
            "node_f1": metrics["node"]["f1"],
            "node_recall": metrics["node"]["recall"],
            "node_precision": metrics["node"]["precision"],
        }
        results.append(row)

        # cleanup to help free GPU memory between model runs
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # save per-model loss / val metric plots (train steps vs val steps)
        try:
            import matplotlib.pyplot as plt

            plot_dir = os.path.join("plots", f"{model_name}")
            os.makedirs(plot_dir, exist_ok=True)

            # training loss per step
            plt.figure()
            plt.plot(range(1, len(hist["train_loss"]) + 1), hist["train_loss"], marker=".", label="train_loss")
            plt.plot(range(1, len(hist["train_loss"]) + 1), hist["train_loss_graph"], marker=".", label="train_loss_graph")
            plt.plot(range(1, len(hist["train_loss"]) + 1), hist["train_loss_node"], marker=".", label="train_loss_node")
            plt.legend()
            plt.xlabel("step")
            plt.ylabel("train_loss")
            plt.title(f"{model_name} train loss (per step)")
            unique_tag = f"r{trojan_ratio}_nodes{min_nodes}-{max_nodes}_bs{batch_size}_ep{epochs}_{int(time.time())}_{os.getpid()}"
            pfile = os.path.join(plot_dir, f"{model_name}_{unique_tag}_train_loss_steps.png")
            plt.tight_layout()
            plt.savefig(pfile)
            plt.close()

            # val metric over steps (scatter)
            plt.figure()
            xs = hist.get("val_steps", [])
            if xs:
                plt.plot(xs, hist.get("val_graph_bce", []), marker="o", label="val_graph_bce")
                plt.plot(xs, hist.get("val_graph_roc_auc", []), marker="o", label="val_graph_roc_auc")
                plt.plot(xs, hist.get("val_graph_ap", []), marker="o", label="val_graph_ap")
                plt.plot(xs, hist.get("val_graph_f1", []), marker="o", label="val_graph_f1")
                plt.plot(xs, hist.get("val_graph_recall", []), marker="o", label="val_graph_recall")
                plt.plot(xs, hist.get("val_graph_precision", []), marker="o", label="val_graph_precision")
            plt.xlabel("step")
            plt.title(f"{model_name} validation graph-level metrics (per val step)")
            plt.legend()
            pfile2 = os.path.join(plot_dir, f"{model_name}_{unique_tag}_val_graph.png")
            plt.tight_layout()
            plt.savefig(pfile2)
            plt.close()

            plt.figure()
            xs = hist.get("val_steps", [])
            if xs:
                plt.plot(xs, hist.get("val_node_bce", []), marker="o", label="val_node_bce")
                plt.plot(xs, hist.get("val_node_roc_auc", []), marker="o", label="val_node_roc_auc")
                plt.plot(xs, hist.get("val_node_ap", []), marker="o", label="val_node_ap")
                plt.plot(xs, hist.get("val_node_f1", []), marker="o", label="val_node_f1")
                plt.plot(xs, hist.get("val_node_recall", []), marker="o", label="val_node_recall")
                plt.plot(xs, hist.get("val_node_precision", []), marker="o", label="val_node_precision")
            plt.xlabel("step")
            plt.title(f"{model_name} validation node-level metrics (per val step)")
            plt.legend()
            pfile2 = os.path.join(plot_dir, f"{model_name}_{unique_tag}_val_node.png")
            plt.tight_layout()
            plt.savefig(pfile2)
            plt.close()
            print(f"Saved model plots to {plot_dir}")
        except Exception as e:
            print(f"Could not save per-model plots for {model_name}: {e}")
        # finally:
        #     breakpoint()

        # incremental CSV write if requested
        if out_csv:
            header = not os.path.exists(out_csv)
            with open(out_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if header:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()

    return results


def set_global_seed(seed: int) -> None:
    """Set seeds for python, numpy and torch to improve reproducibility."""
    pyrandom.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # encourage deterministic behavior where possible
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def parse_ratio_list(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(description="Matrix runner for trojan graph experiments")
    parser.add_argument("--min-nodes", type=int, required=True)
    parser.add_argument("--max-nodes", type=int, required=True)
    parser.add_argument("--trojan-ratios", type=str, default="0.6", help="Comma-separated trojan ratios to sweep")
    parser.add_argument("--num-graphs", type=int, default=50000, help="Training set size per experiment")
    parser.add_argument("--val-graphs", type=int, default=2000)
    parser.add_argument("--test-graphs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-csv", type=str, default="matrix_results.csv")
    parser.add_argument("--dataset-mode", type=str, default="ondemand", choices=["ondemand", "pregen", "disk"])
    parser.add_argument("--pregen-dir", type=str, default="pregen_data", help="Directory containing pre-generated train/val/test subfolders with .pt files")
    parser.add_argument("--model", type=str, default="all", help="Model to run (GCN,GAT,H2GNN,GraphSAGE) or 'all'")
    parser.add_argument("--hidden-size", type=int, default=None, help="Hidden size to pass to the model constructor (overrides model default)")
    parser.add_argument("--trojan-mix-prob", type=float, default=0.08)
    parser.add_argument("--max-struct-features-nodes", type=int, default=4000)
    parser.add_argument("--betweenness-k", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=None, help="Total training steps (overrides epochs when set)")
    parser.add_argument("--val-every-steps", type=int, default=100, help="Run validation every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset/model generation")
    args = parser.parse_args()

    # Set global RNG seeds for reproducibility
    set_global_seed(args.seed)

    device = te._choose_device(args.device)
    ratios = parse_ratio_list(args.trojan_ratios)

    # Run experiments for provided ratios (single size passed via args)
    all_rows = []
    out_path = os.path.abspath(args.out_csv)
    # remove existing out file so incremental writes start fresh
    if os.path.exists(out_path):
        os.remove(out_path)

    for ratio in ratios:
        rows = run_experiment(
            args.min_nodes,
            args.max_nodes,
            ratio,
            args.num_graphs,
            args.val_graphs,
            args.test_graphs,
            args.batch_size,
            args.epochs,
            device,
            args.dataset_mode,
            args.trojan_mix_prob,
            args.max_struct_features_nodes,
            args.betweenness_k,
            args.seed,
            args.pregen_dir,
            model_name=(None if args.model.lower() in ["all", "*"] else args.model),
            hidden_size=args.hidden_size,
            out_csv=out_path,
            max_steps=args.max_steps,
            val_every_steps=args.val_every_steps,
        )
        all_rows.extend(rows)

    print(f"Wrote results to {out_path}")

    try:
        _make_plots_from_csv(out_path)
    except Exception as e:
        print(f"Plotting failed: {e}")


def _make_plots_from_csv(csv_path: str, out_dir: str = "plots"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    # Aggregate by trojan_ratio and model
    group = df.groupby(["trojan_ratio", "model"]).mean().reset_index()

    # Include the new metrics (f1, recall, precision) for both graph and node levels
    graph_metrics = ["graph_roc_auc", "graph_ap", "graph_acc", "graph_f1", "graph_recall", "graph_precision"]
    node_metrics = ["node_roc_auc", "node_ap", "node_acc", "node_f1", "node_recall", "node_precision"]

    # Melt to long-form for easy faceting with seaborn
    gdf = group[["trojan_ratio", "model"] + graph_metrics].melt(id_vars=["trojan_ratio", "model"], value_vars=graph_metrics, var_name="metric", value_name="value")
    ndf = group[["trojan_ratio", "model"] + node_metrics].melt(id_vars=["trojan_ratio", "model"], value_vars=node_metrics, var_name="metric", value_name="value")

    # Ensure trojan_ratio is treated as a categorical x-axis (keeps consistent ordering)
    gdf["trojan_ratio"] = gdf["trojan_ratio"].astype(str)
    ndf["trojan_ratio"] = ndf["trojan_ratio"].astype(str)

    sns.set_theme(style="whitegrid")

    # Graph-level metrics: facet by metric (columns), hue by model
    g = sns.catplot(
        data=gdf,
        x="trojan_ratio",
        y="value",
        hue="model",
        col="metric",
        kind="bar",
        col_wrap=3,
        sharey=False,
        height=4,
        aspect=1,
    )
    g.figure.suptitle("Graph-level metrics by Trojan ratio and model", y=1.02)
    out_file = os.path.join(out_dir, "graph_metrics_facet.png")
    plt.tight_layout()
    g.savefig(out_file)
    plt.close()
    print(f"Saved plot: {out_file}")

    # Node-level metrics: facet by metric (columns), hue by model
    g2 = sns.catplot(
        data=ndf,
        x="trojan_ratio",
        y="value",
        hue="model",
        col="metric",
        kind="bar",
        col_wrap=3,
        sharey=False,
        height=4,
        aspect=1,
    )
    g2.figure.suptitle("Node-level metrics by Trojan ratio and model", y=1.02)
    out_file2 = os.path.join(out_dir, "node_metrics_facet.png")
    plt.tight_layout()
    g2.savefig(out_file2)
    plt.close()
    print(f"Saved plot: {out_file2}")


if __name__ == "__main__":
    main()
