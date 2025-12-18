import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# ensure project root is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models import GCNClassifier, GATClassifier, GraphSAGEClassifier, H2GNNClassifier


def parse_hidden(source_file: str):
    m = re.search(r'_h(\d+)_', source_file)
    if m:
        return int(m.group(1))
    m = re.search(r'h(\d+)', source_file)
    return int(m.group(1)) if m else None


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    df = pd.read_csv('results/combined_results.csv')
    df.columns = df.columns.str.strip()
    df['trojan_ratio'] = pd.to_numeric(df['trojan_ratio'], errors='coerce')
    df['graph_f1'] = pd.to_numeric(df['graph_f1'], errors='coerce')
    df['node_f1'] = pd.to_numeric(df['node_f1'], errors='coerce')

    df05 = df[df['trojan_ratio'] == 0.5]
    if df05.empty:
        print('No rows for trojan_ratio 0.5')
        return

    print(len(df05), 'rows for trojan_ratio=0.5')

    rows = []
    models_map = {
        'GCN': GCNClassifier,
        'GAT': GATClassifier,
        'GraphSAGE': GraphSAGEClassifier,
        'H2GNN': H2GNNClassifier,
    }

    in_channels = 13

    # For each recorded config at ratio=0.5, compute params and keep graph/node F1
    for _, r in df05.iterrows():
        model_name = r['model']
        if model_name not in models_map:
            continue
        hidden = parse_hidden(r.get('source_file', '')) or 96
        cls = models_map[model_name]
        if model_name == 'GAT':
            model = cls(in_channels=in_channels, hidden=hidden, heads=4)
        else:
            model = cls(in_channels=in_channels, hidden=hidden)
        params = count_params(model)
        rows.append({
            'model': model_name,
            'hidden': int(hidden),
            'params': int(params),
            'graph_f1': r['graph_f1'],
            'node_f1': r['node_f1'],
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(['model', 'hidden'])

    # Melt to long form: metric (Graph/Node) and value
    long_rows = []
    for _, r in out.iterrows():
        long_rows.append({'model': r['model'], 'hidden': r['hidden'], 'params': r['params'], 'metric': 'Graph F1', 'value': r['graph_f1']})
        long_rows.append({'model': r['model'], 'hidden': r['hidden'], 'params': r['params'], 'metric': 'Node F1', 'value': r['node_f1']})
    long_df = pd.DataFrame(long_rows)
    long_df['value_pct'] = long_df['value'] * 100

    # Plot using seaborn for consistent styling
    sns.set_theme(style='whitegrid', context='paper')
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.scatterplot(data=long_df, x='params', y='value_pct', hue='model', style='metric', s=100, ax=ax)
    # annotate with hidden size near each point
    for _, r in long_df.iterrows():
        ax.annotate(f"h{int(r['hidden'])}", (r['params'], r['value_pct']), xytext=(4, -6), textcoords='offset points', fontsize=7)

    ax.set_xscale('log')
    ax.set_xlabel('Number of parameters (log scale)')
    ax.set_ylabel('F1 score (%)')
    ax.set_title('Model Size vs Performance (trojan_ratio=0.5)')
    ax.legend()
    plt.tight_layout()

    out_dir = Path('writeup/imgs')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'params_vs_f1.png'
    fig.savefig(out_path, dpi=200)
    print('Saved plot to', out_path)

if __name__ == '__main__':
    main()