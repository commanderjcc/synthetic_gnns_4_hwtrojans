import argparse
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

from data_gen import GenerationConfig, GraphWithMeta, SyntheticTrojanDataset, generate_dataset
from models import GATClassifier, GCNClassifier, H2GNNClassifier


def _choose_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if torch.cuda.is_available() and torch.version.cuda is not None:
            return torch.device("cuda")
        print("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    if torch.cuda.is_available() and torch.version.cuda is not None:
        return torch.device("cuda")
    return torch.device("cpu")


def _split_dataset(graphs: List[GraphWithMeta], seed: int, val_len: Optional[int] = None, test_len: Optional[int] = None) -> Tuple[List, List, List]:
    data_list = [g.data for g in graphs]
    total = len(data_list)
    if val_len is None or test_len is None:
        train_len = int(total * 0.7)
        val_len = max(1, int(total * 0.1))
        test_len = total - train_len - val_len
    else:
        train_len = total - val_len - test_len
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(data_list, [train_len, val_len, test_len], generator=generator)
    return list(train_ds), list(val_ds), list(test_ds)


def _build_type_mapping(graphs: List[GraphWithMeta]) -> Dict[str, int]:
    types = sorted({g.trojan_type for g in graphs})
    return {t: idx for idx, t in enumerate(types)}


def _assign_type_ids(graphs: List[GraphWithMeta], type_to_id: Dict[str, int]) -> None:
    for g in graphs:
        g.data.trojan_type_id = torch.tensor([type_to_id.get(g.trojan_type, 0)], dtype=torch.long)


def _prepare_loaders(train_ds, val_ds, test_ds, batch_size: int = 8):
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def _compute_metrics(labels: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    out = {"bce": float(-np.mean(labels * np.log(probs + 1e-8) + (1 - labels) * np.log(1 - probs + 1e-8)))}
    if len(np.unique(labels)) > 1:
        out["roc_auc"] = float(roc_auc_score(labels, probs))
        out["ap"] = float(average_precision_score(labels, probs))
        out["f1"] = float(f1_score(labels, probs > 0.5))
        out["recall"] = float(recall_score(labels, probs > 0.5))
        out["precision"] = float(precision_score(labels, probs > 0.5))
    else:
        out["roc_auc"] = float("nan")
        out["ap"] = float("nan")
        out["f1"] = float("nan")
        out["recall"] = float("nan")
        out["precision"] = float("nan")
    out["acc"] = float(np.mean((probs > 0.5) == labels))
    return out


def evaluate(model, loader, device) -> Dict[str, Dict[str, float]]:
    model.eval()
    node_labels, node_logits = [], []
    graph_labels, graph_logits = [], []
    type_ids = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False, unit="batch"):
            batch = batch.to(device)
            g_log, n_log = model(batch)
            node_labels.append(batch.y_node.detach().cpu().numpy())
            node_logits.append(n_log.detach().cpu().numpy())
            graph_labels.append(batch.y_graph.detach().cpu().numpy())
            graph_logits.append(g_log.detach().cpu().numpy())
            if hasattr(batch, "trojan_type_id"):
                type_ids.append(batch.trojan_type_id.detach().cpu().numpy())
    node_labels = np.concatenate(node_labels)
    node_logits = np.concatenate(node_logits)
    graph_labels = np.concatenate(graph_labels)
    graph_logits = np.concatenate(graph_logits)
    type_ids = np.concatenate(type_ids) if type_ids else np.zeros_like(graph_labels)
    metrics = {
        "graph": _compute_metrics(graph_labels, graph_logits),
        "node": _compute_metrics(node_labels, node_logits),
        "type_ids": type_ids,
        "graph_probs": 1.0 / (1.0 + np.exp(-graph_logits)),
        "graph_labels": graph_labels,
    }
    return metrics


def _summarize_by_type(graph_probs: np.ndarray, graph_labels: np.ndarray, type_ids: np.ndarray, id_to_type: Dict[int, str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for type_id in np.unique(type_ids):
        mask = type_ids == type_id
        labels = graph_labels[mask]
        probs = graph_probs[mask]
        summary[id_to_type.get(int(type_id), f"type_{int(type_id)}")] = {
            "count": int(mask.sum()),
            "positives": int(labels.sum()),
            "avg_prob": float(probs.mean()),
            "acc": float(np.mean((probs > 0.5) == labels)),
        }
    return summary


def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 40,
    lr: float = 1e-3,
    node_w: float = 0.6,
    graph_w: float = 0.4,
    max_steps: int | None = None,
    val_every_steps: int = 100,
    trojan_graph_ratio: float | None = None,
    node_trojan_frac_estimate: float = 0.1,
):
    """Train the model for a given number of steps (preferred) or epochs.

    If `max_steps` is None, will run for `epochs * len(train_loader)` steps.
    Validation is run every `val_every_steps` steps (and at the end).
    Returns a history dict with per-step train loss and validation snapshots.
    """
    from itertools import cycle

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # Use class-pos weights for BCE losses to account for class imbalance.
    # If an estimate of positive fraction is provided, set pos_weight = (1 - p) / p
    pos_w_graph = None
    pos_w_node = None
    if trojan_graph_ratio is not None:
        g_p = float(trojan_graph_ratio)
        # avoid division by zero
        g_p = max(g_p, 1e-6)
        pos_w_graph = torch.tensor((1.0 - g_p) / g_p, device=device, dtype=torch.float32)
    if node_trojan_frac_estimate is not None:
        n_p = float(node_trojan_frac_estimate)
        n_p = max(n_p, 1e-6)
        pos_w_node = torch.tensor((1.0 - n_p) / n_p, device=device, dtype=torch.float32)

    bce_graph = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w_graph)
    bce_node = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w_node)

    # If a trojan_graph_ratio is provided, use it to set the graph/node loss weights.
    # The node estimate (`node_trojan_frac_estimate`) is an estimate of the fraction of
    # nodes in a graph that belong to the trojan (default ~0.1 = 10%). We normalize
    # the two weights so they sum to 1 to keep loss scale stable.
    if trojan_graph_ratio is not None:
        graph_w = float(trojan_graph_ratio)
        node_w = float(node_trojan_frac_estimate)
        s = graph_w + node_w
        if s > 0:
            graph_w = graph_w / s
            node_w = node_w / s

    # determine total steps
    loader_size = len(train_loader)
    if max_steps is None:
        max_steps = epochs * loader_size

    hist = {
        "train_loss": [],
        "train_loss_graph": [],
        "train_loss_node": [],
        "val_steps": [],
        "val_graph_bce": [],
        "val_graph_roc_auc": [],
        "val_graph_ap": [],
        "val_graph_f1": [],
        "val_graph_recall": [],
        "val_graph_precision": [],
        "val_node_bce": [],
        "val_node_roc_auc": [],
        "val_node_ap": [],
        "val_node_f1": [],
        "val_node_recall": [],
        "val_node_precision": [],
    }

    model.train()
    train_iter = cycle(iter(train_loader))
    pbar = tqdm(range(1, max_steps + 1), desc=f"Train steps", unit="step")
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = batch.to(device)
        opt.zero_grad()
        g_log, n_log = model(batch)
        node_loss = bce_node(n_log, batch.y_node.float())
        graph_loss = bce_graph(g_log, batch.y_graph.float())
        loss = node_w * node_loss + graph_w * graph_loss
        loss.backward()
        opt.step()

        hist["train_loss"].append(float(loss.item()))
        hist["train_loss_graph"].append(float(graph_loss.item()))
        hist["train_loss_node"].append(float(node_loss.item()))
        pbar.set_postfix(train_loss=float(loss.item()))

        if val_every_steps > 0 and (step % val_every_steps == 0 or step == max_steps):
            val_metrics = evaluate(model, val_loader, device)
            hist["val_steps"].append(step)
            hist["val_graph_bce"].append(val_metrics["graph"]["bce"])
            hist["val_graph_roc_auc"].append(val_metrics["graph"]["roc_auc"])
            hist["val_graph_ap"].append(val_metrics["graph"]["ap"])
            hist["val_graph_f1"].append(val_metrics["graph"]["f1"])
            hist["val_graph_recall"].append(val_metrics["graph"]["recall"])
            hist["val_graph_precision"].append(val_metrics["graph"]["precision"])
            hist["val_node_bce"].append(val_metrics["node"]["bce"])
            hist["val_node_roc_auc"].append(val_metrics["node"]["roc_auc"])
            hist["val_node_ap"].append(val_metrics["node"]["ap"])
            hist["val_node_f1"].append(val_metrics["node"]["f1"])
            hist["val_node_recall"].append(val_metrics["node"]["recall"])
            hist["val_node_precision"].append(val_metrics["node"]["precision"])
            if step % (val_every_steps * 10) == 0 or step == max_steps:
                print(f"Step {step}/{max_steps} | val graph auc {val_metrics['graph']['roc_auc']:.3f} | val node auc {val_metrics['node']['roc_auc']:.3f}")

    return hist


def _build_config(args) -> GenerationConfig:
    return GenerationConfig(
        num_graphs=args.num_graphs,
        trojan_ratio=args.trojan_ratio,
        node_range=(args.min_nodes, args.max_nodes),
        seed=args.seed,
        max_struct_features_nodes=args.max_struct_features_nodes,
        betweenness_sample_k=args.betweenness_k,
        trojan_mix_prob=args.trojan_mix_prob,
    )


def _make_loaders(args):
    cfg = _build_config(args)
    id_to_type: Dict[int, str] = {0: "clean"}

    if args.dataset_mode == "ondemand":
        train_ds = SyntheticTrojanDataset(cfg, length=args.num_graphs, seed_offset=0)
        val_ds = SyntheticTrojanDataset(cfg, length=args.val_graphs, seed_offset=1_000_000)
        test_ds = SyntheticTrojanDataset(cfg, length=args.test_graphs, seed_offset=2_000_000)
        id_to_type.update({v: k for k, v in train_ds.type_to_id.items()})
    else:
        graphs = generate_dataset(cfg)
        type_to_id = _build_type_mapping(graphs)
        id_to_type = {v: k for k, v in type_to_id.items()}
        _assign_type_ids(graphs, type_to_id)
        val_len = args.val_graphs if args.val_graphs else None
        test_len = args.test_graphs if args.test_graphs else None
        train_ds, val_ds, test_ds = _split_dataset(graphs, seed=args.seed, val_len=val_len, test_len=test_len)

    loaders = _prepare_loaders(train_ds, val_ds, test_ds, batch_size=args.batch_size)
    return loaders, id_to_type, train_ds