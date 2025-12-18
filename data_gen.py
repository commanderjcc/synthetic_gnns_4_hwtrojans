import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
from tqdm import trange

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import os
from typing import Iterable

# Define gate vocabulary used for node features
GATE_TYPES = [
    "and",
    "or",
    "xor",
    "mux",
    "ff",
    "const",
    "nand",
    "nor",
]


@dataclass
class GenerationConfig:
    num_graphs: int = 200
    trojan_ratio: float = 0.5  # probability a graph contains a trojan
    er_p_range: Tuple[float, float] = (0.02, 0.08)
    ba_m_range: Tuple[int, int] = (2, 4)
    node_range: Tuple[int, int] = (80, 320)
    trojan_mix_prob: float = 0.08  # probability of injecting two patterns
    seed: int = 17
    max_struct_features_nodes: int = 4000  # skip expensive metrics when graphs exceed this size
    betweenness_sample_k: int = 32  # number of nodes to sample for betweenness when enabled


@dataclass
class GraphWithMeta:
    data: Data
    trojan_type: str


def _sample_gate_one_hot(rng: random.Random) -> np.ndarray:
    idx = rng.randrange(len(GATE_TYPES))
    one_hot = np.zeros(len(GATE_TYPES), dtype=np.float32)
    one_hot[idx] = 1.0
    return one_hot


def _sample_clamped_normal(rng: random.Random, mean: float, std: float) -> float:
    """Sample from a normal distribution using the provided RNG and clamp to [0, 1]."""
    val = rng.gauss(mean, std)
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return float(val)


# Defaults for non-trojan and trojan distributions (means and stddevs)
NON_TROJAN_MEAN = 0.6
NON_TROJAN_STD = 0.15
TROJAN_MEAN = 0.4
TROJAN_STD = 0.1


def _attach_base_features(G: nx.Graph, rng: random.Random) -> None:
    for node in G.nodes():
        G.nodes[node]["gate_vec"] = _sample_gate_one_hot(rng)
        # Sample base (non-trojan) trigger and rare-signal values from
        # a clamped normal distribution so they differ slightly from trojan nodes.
        G.nodes[node]["trigger_prob"] = _sample_clamped_normal(rng, NON_TROJAN_MEAN, NON_TROJAN_STD)
        G.nodes[node]["rare_signal"] = _sample_clamped_normal(rng, NON_TROJAN_MEAN, NON_TROJAN_STD)
        G.nodes[node]["trojan"] = 0


def _pick_base_graph(cfg: GenerationConfig, rng: random.Random) -> nx.Graph:
    num_nodes = rng.randint(cfg.node_range[0], cfg.node_range[1])
    if rng.random() < 0.5:
        p = rng.uniform(*cfg.er_p_range)
        G = nx.erdos_renyi_graph(num_nodes, p)
    else:
        m = rng.randint(cfg.ba_m_range[0], cfg.ba_m_range[1])
        G = nx.barabasi_albert_graph(num_nodes, m)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    _attach_base_features(G, rng)
    return G


def _add_trigger_chain(G: nx.Graph, rng: random.Random) -> str:
    """A linear chain of nodes leading to a payload node."""
    length = rng.randint(4, 10)
    anchor = rng.choice(list(G.nodes()))
    start_id = G.number_of_nodes()
    prev = anchor
    for i in range(length):
        node_id = start_id + i
        G.add_node(node_id)
        G.nodes[node_id]["gate_vec"] = _sample_gate_one_hot(rng)
        # Trojan nodes get a slightly lower mean trigger_prob than base nodes
        G.nodes[node_id]["trigger_prob"] = _sample_clamped_normal(rng, TROJAN_MEAN, TROJAN_STD)
        # Preserve a strong rare_signal for the first trigger node, sample others
        G.nodes[node_id]["rare_signal"] = 1.0 if i == 0 else _sample_clamped_normal(rng, TROJAN_MEAN, TROJAN_STD)
        G.nodes[node_id]["trojan"] = 1
        G.add_edge(prev, node_id)
        prev = node_id
    payload_id = G.number_of_nodes()
    G.add_node(payload_id)
    G.nodes[payload_id]["gate_vec"] = _sample_gate_one_hot(rng)
    G.nodes[payload_id]["trigger_prob"] = _sample_clamped_normal(rng, TROJAN_MEAN, TROJAN_STD)
    G.nodes[payload_id]["rare_signal"] = _sample_clamped_normal(rng, TROJAN_MEAN, TROJAN_STD)
    G.nodes[payload_id]["trojan"] = 1
    G.add_edge(prev, payload_id)
    return "trigger_chain"


def _add_high_fanin(G: nx.Graph, rng: random.Random) -> str:
    """a combinatorial structure with many inputs converging on a single node"""
    target = rng.choice(list(G.nodes()))
    fan_in = rng.randint(5, 12)
    start_id = G.number_of_nodes()
    for i in range(fan_in):
        node_id = start_id + i
        G.add_node(node_id)
        G.nodes[node_id]["gate_vec"] = _sample_gate_one_hot(rng)
        G.nodes[node_id]["trigger_prob"] = _sample_clamped_normal(rng, TROJAN_MEAN, TROJAN_STD)
        G.nodes[node_id]["rare_signal"] = _sample_clamped_normal(rng, NON_TROJAN_MEAN, NON_TROJAN_STD)
        G.nodes[node_id]["trojan"] = 1
        G.add_edge(node_id, target)
        if rng.random() < 0.25 and i > 0:
            peer = rng.randint(start_id, node_id - 1)
            G.add_edge(node_id, peer)
    # Mark target as trojan and bump its trigger probability to a trojan-sampled value
    G.nodes[target]["trojan"] = 1
    G.nodes[target]["trigger_prob"] = _sample_clamped_normal(rng, TROJAN_MEAN, TROJAN_STD)
    return "high_fanin"


def _add_watermark_motif(G: nx.Graph, rng: random.Random) -> str:
    """a denser section of the graph"""
    motif_size = rng.randint(6, 12)
    start_id = G.number_of_nodes()
    nodes = []
    for i in range(motif_size):
        node_id = start_id + i
        nodes.append(node_id)
        G.add_node(node_id)
        G.nodes[node_id]["gate_vec"] = _sample_gate_one_hot(rng)
        G.nodes[node_id]["trigger_prob"] = _sample_clamped_normal(rng, NON_TROJAN_MEAN, NON_TROJAN_STD)
        G.nodes[node_id]["rare_signal"] = _sample_clamped_normal(rng, NON_TROJAN_MEAN, NON_TROJAN_STD)
        G.nodes[node_id]["trojan"] = 1
    # Build a dense directed-like motif by adding a ring and random chords
    for i in range(motif_size):
        G.add_edge(nodes[i], nodes[(i + 1) % motif_size])
    chord_count = rng.randint(2, max(2, motif_size // 2))
    for _ in range(chord_count):
        u, v = rng.sample(nodes, 2)
        G.add_edge(u, v)
    anchor = rng.choice(list(G.nodes() - set(nodes)))
    bridge_nodes = rng.sample(nodes, k=rng.randint(1, min(2, len(nodes))))
    for bn in bridge_nodes:
        G.add_edge(anchor, bn)
    return "watermark_motif"


def _add_feedback_loop(G: nx.Graph, rng: random.Random) -> str:
    """a loop where flip-flop like gates are more common"""
    loop_size = rng.randint(3, 8)
    start_id = G.number_of_nodes()
    nodes = []
    for i in range(loop_size):
        node_id = start_id + i
        nodes.append(node_id)
        G.add_node(node_id)
        gate_vec = _sample_gate_one_hot(rng)
        # Encourage including flip-flop like gates in the loop
        if rng.random() < 0.5:
            gate_vec = np.zeros(len(GATE_TYPES), dtype=np.float32)
            gate_vec[GATE_TYPES.index("ff")] = 1.0
        G.nodes[node_id]["gate_vec"] = gate_vec
        G.nodes[node_id]["trigger_prob"] = _sample_clamped_normal(rng, NON_TROJAN_MEAN, NON_TROJAN_STD)
        G.nodes[node_id]["rare_signal"] = _sample_clamped_normal(rng, NON_TROJAN_MEAN, NON_TROJAN_STD)
        G.nodes[node_id]["trojan"] = 1
    for i in range(loop_size):
        G.add_edge(nodes[i], nodes[(i + 1) % loop_size])
    if rng.random() < 0.5:
        u, v = rng.sample(nodes, 2)
        G.add_edge(u, v)
    tap = rng.choice(list(G.nodes() - set(nodes)))
    G.add_edge(nodes[-1], tap)
    return "feedback_loop"


PATTERN_FUNCS: List[Callable[[nx.Graph, random.Random], str]] = [
    _add_trigger_chain,
    _add_high_fanin,
    _add_watermark_motif,
    _add_feedback_loop,
]


def _compute_structural_features(G: nx.Graph, cfg: GenerationConfig) -> Tuple[np.ndarray, np.ndarray]:
    degs = dict(G.degree())
    max_deg = max(degs.values()) if degs else 1
    node_ids = sorted(G.nodes())
    n = len(node_ids)

    clustering = {n: 0.0 for n in node_ids}
    betweenness = {n: 0.0 for n in node_ids}
    if n <= cfg.max_struct_features_nodes:
        clustering = nx.clustering(G)
        k = min(cfg.betweenness_sample_k, max(4, int(math.sqrt(n))))
        betweenness = nx.betweenness_centrality(G, k=k, seed=42)

    struct_feats = []
    trojan_labels = []
    for nid in node_ids:
        deg_norm = degs[nid] / max_deg if max_deg > 0 else 0.0
        struct_feats.append(
            [
                deg_norm,
                clustering.get(nid, 0.0),
                betweenness.get(nid, 0.0),
                float(G.nodes[nid].get("trigger_prob", 0.0)),
                float(G.nodes[nid].get("rare_signal", 0.0)),
            ]
        )
        trojan_labels.append(int(G.nodes[nid].get("trojan", 0)))
    return np.asarray(struct_feats, dtype=np.float32), np.asarray(trojan_labels, dtype=np.int64)


def _graph_to_data(G: nx.Graph, trojan_type: str, cfg: GenerationConfig) -> GraphWithMeta:
    struct_feats, trojan_labels = _compute_structural_features(G, cfg)
    gate_vecs = []
    for n in sorted(G.nodes()):
        gate_vecs.append(G.nodes[n].get("gate_vec", np.zeros(len(GATE_TYPES), dtype=np.float32)))
    gate_vecs = np.stack(gate_vecs).astype(np.float32)
    x = np.concatenate([struct_feats, gate_vecs], axis=1)
    edge_index = np.asarray(list(G.edges()), dtype=np.int64).T
    # Add reverse edges to encourage undirected message passing
    rev_edges = edge_index[[1, 0], :]
    edge_index = np.concatenate([edge_index, rev_edges], axis=1)
    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        y_graph=torch.tensor([1 if trojan_labels.sum() > 0 else 0], dtype=torch.long),
        y_node=torch.from_numpy(trojan_labels),
    )
    data.trojan_type = trojan_type
    return GraphWithMeta(data=data, trojan_type=trojan_type)


def generate_graph(cfg: GenerationConfig, rng: random.Random) -> GraphWithMeta:
    G = _pick_base_graph(cfg, rng)
    trojan_type = "clean"
    if rng.random() < cfg.trojan_ratio:
        primary = rng.choice(PATTERN_FUNCS)
        trojan_type = primary(G, rng)
        if rng.random() < cfg.trojan_mix_prob:
            secondary = rng.choice(PATTERN_FUNCS)
            # Avoid duplicate type names in mixes
            secondary_type = secondary(G, rng)
            trojan_type = f"mix:{trojan_type}+{secondary_type}"
    return _graph_to_data(G, trojan_type, cfg)


def generate_dataset(cfg: GenerationConfig) -> List[GraphWithMeta]:
    rng = random.Random(cfg.seed)
    graphs: List[GraphWithMeta] = []
    for _ in range(cfg.num_graphs):
        graphs.append(generate_graph(cfg, rng))
    return graphs


def generate_and_save_dataset(cfg: GenerationConfig, out_dir: str, val_frac: float = 0.1, test_frac: float = 0.1, filename_prefix: str = "graph_") -> None:
    """Generate graphs and save them directly to disk as `.pt` files without keeping them in memory.

    Creates `out_dir/train`, `out_dir/val`, `out_dir/test` and writes each generated
    `torch_geometric.data.Data` object using `torch.save`. The split is determined
    by `val_frac` and `test_frac` (fractions of total `cfg.num_graphs`).
    """
    import math
    import torch

    os.makedirs(out_dir, exist_ok=True)
    train_dir = os.path.join(out_dir, "train")
    val_dir = os.path.join(out_dir, "val")
    test_dir = os.path.join(out_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    rng = random.Random(cfg.seed)
    n = cfg.num_graphs
    n_val = int(math.floor(n * val_frac))
    n_test = int(math.floor(n * test_frac))
    n_train = n - n_val - n_test

    # Counters for naming files in each split
    c_train = 0
    c_val = 0
    c_test = 0

    for i in trange(n):
        meta = generate_graph(cfg, rng)
        if i < n_train:
            d = train_dir
            idx = c_train
            c_train += 1
        elif i < n_train + n_val:
            d = val_dir
            idx = c_val
            c_val += 1
        else:
            d = test_dir
            idx = c_test
            c_test += 1

        path = os.path.join(d, f"{filename_prefix}{idx:06d}.pt")
        torch.save(meta.data, path)

    return None


class SyntheticTrojanDataset(Dataset):
    """On-the-fly synthetic dataset to avoid storing tens of thousands of graphs in memory."""

    def __init__(self, cfg: GenerationConfig, length: int, seed_offset: int = 0, warmup_scan: int = 256):
        self.cfg = cfg
        self.length = length
        self.seed_offset = seed_offset
        self.type_to_id = {"clean": 0}

        # Warmup pass to stabilize type IDs (captures mixes probabilistically)
        rng = random.Random(cfg.seed + seed_offset)
        for _ in range(min(warmup_scan, length)):
            meta = generate_graph(cfg, rng)
            if meta.trojan_type not in self.type_to_id:
                self.type_to_id[meta.trojan_type] = len(self.type_to_id)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = random.Random(self.cfg.seed + self.seed_offset + idx)
        meta = generate_graph(self.cfg, rng)
        t_id = self.type_to_id.get(meta.trojan_type)
        if t_id is None:
            t_id = len(self.type_to_id)
            self.type_to_id[meta.trojan_type] = t_id
        meta.data.trojan_type_id = torch.tensor([t_id], dtype=torch.long)
        return meta.data


class PregeneratedDiskDataset(Dataset):
    """Dataset that loads pre-generated `torch_geometric.data.Data` objects from disk.

    Files are expected to be individual torch-saved Data objects (e.g. via
    `torch.save(data, path)`). The dataset lists files in `data_dir` and lazily
    loads them on __getitem__.
    """

    def __init__(self, data_dir: str, pattern: str = "*.pt"):
        self.data_dir = data_dir
        # collect files sorted for deterministic ordering
        files = [os.path.join(data_dir, p) for p in os.listdir(data_dir) if p.endswith(".pt")]
        self.files = sorted(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        path = self.files[idx]
        # torch.load will return the Data object saved by the generator script
        return torch.load(path, weights_only=False)
