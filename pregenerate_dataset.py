#!/usr/bin/env python3
"""Generate a pregenerated dataset of graphs and save to disk as individual .pt files.

Creates `out_dir/train`, `out_dir/val`, `out_dir/test` and writes torch-saved
`torch_geometric.data.Data` objects into them.

Usage example:
  python scripts/pregenerate_dataset.py --out-dir pregen_data --num-graphs 2000 --min-nodes 200 --max-nodes 500 --trojan-ratio 0.5
"""
import argparse
import os
import math
import torch
from data_gen import GenerationConfig, generate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-graphs", type=int, default=2000)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--min-nodes", type=int, default=80)
    parser.add_argument("--max-nodes", type=int, default=320)
    parser.add_argument("--trojan-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_dir = os.path.join(args.out_dir, "train")
    val_dir = os.path.join(args.out_dir, "val")
    test_dir = os.path.join(args.out_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    cfg = GenerationConfig(
        num_graphs=args.num_graphs,
        trojan_ratio=args.trojan_ratio,
        node_range=(args.min_nodes, args.max_nodes),
        seed=args.seed,
    )

    # Stream-generate and save graphs directly to disk without retaining them in memory.
    from data_gen import generate_and_save_dataset

    generate_and_save_dataset(cfg, args.out_dir, val_frac=args.val_frac, test_frac=args.test_frac)
    print(f"Saved pregenerated dataset to {args.out_dir} (train/val/test)")


if __name__ == "__main__":
    main()
