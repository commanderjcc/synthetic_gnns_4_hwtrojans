# Graph Benchmarking and Dataset Generation

This repository contains code and pre-generated data for graph neural network experiments, dataset generation utilities, training/evaluation scripts, and plotting helpers used to compare models (GCN, GAT, GraphSAGE, H2GNN).

## Quick Start

- Install dependencies:

```bash
# Using uv (if you use it)
uv sync

# Or with pip (editable install from pyproject.toml)
python -m pip install -e .
```

- Pre-generate datasets:

```bash
python pregenerate_dataset.py  --help  # modify to create the datasets you want
```

- Run training matrix:

```bash
./run_matrix_seq.sh
```

## Repository Layout

- `pregenerate_dataset.py` — utilities to pre-generate and persist datasets in `pregen_data/`.
- `data_gen.py` — dataset creation and augmentation utilities.
- `train_and_eval.py` — training and evaluation entrypoint for experiments.
- `matrix_runner.py` — run many experiments (parameter matrix).
- `concat_results.py` — combine CSV results into summary tables.
- `models.py` — model definitions for GCN / GAT / GraphSAGE / H2GNN
