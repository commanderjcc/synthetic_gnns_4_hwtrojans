import re
import pandas as pd

def parse_hidden(source_file: str):
    m = re.search(r'_h(\d+)_', source_file)
    if m:
        return int(m.group(1))
    m = re.search(r'h(\d+)', source_file)
    return int(m.group(1)) if m else None

def fmt_pct(x, acc=False):
    if pd.isna(x):
        return 'N/A'
    if acc:
        return f"{100.0 * x:.2f}%"
    return f"{100.0 * x:.1f}%"

def best_per_model(df, ratio, level='graph'):
    sub = df[df['trojan_ratio'] == ratio]
    assert not sub.empty, 'No rows for given ratio'
    if level == 'graph':
        key = 'graph_f1'
        acc_col = 'graph_acc'
        prec_col = 'graph_precision'
        rec_col = 'graph_recall'
    else:
        key = 'node_f1'
        acc_col = 'node_acc'
        prec_col = 'node_precision'
        rec_col = 'node_recall'

    rows = []
    for model, g in sub.groupby('model'):
        idx = g[key].idxmax()
        r = g.loc[idx]
        hidden = parse_hidden(r['source_file'])
        rows.append({
            'Model': model,
            'Hidden': hidden,
            'Accuracy': fmt_pct(r[acc_col], acc=True),
            'Precision': fmt_pct(r[prec_col]),
            'Recall': fmt_pct(r[rec_col]),
            'F1': fmt_pct(r[key])
        })
    return pd.DataFrame(rows)

def main():
    df = pd.read_csv('results/combined_results.csv')
    # strip whitespace from column names (CSV has padded headers)
    df.columns = df.columns.str.strip()
    # Ensure numeric columns
    for col in ['trojan_ratio','graph_f1','node_f1','graph_acc','node_acc',
                'graph_precision','graph_recall','node_precision','node_recall']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    graph_best = best_per_model(df, 0.5, level='graph')
    node_best = best_per_model(df, 0.5, level='node')

    print('Graph-level best configs at trojan_ratio=0.5:')
    print(graph_best.to_string(index=False))
    print('\nNode-level best configs at trojan_ratio=0.5:')
    print(node_best.to_string(index=False))

    # Print LaTeX table rows for convenience
    print('\n--- LaTeX rows (graph-level) ---')
    for _, r in graph_best.iterrows():
        print(f"{r['Model']} & {r['Hidden']} & {r['Accuracy']} & {r['Precision']} & {r['Recall']} & {r['F1']} \\")

    print('\n--- LaTeX rows (node-level) ---')
    for _, r in node_best.iterrows():
        print(f"{r['Model']} & {r['Hidden']} & {r['Accuracy']} & {r['Precision']} & {r['Recall']} & {r['F1']} \\")

if __name__ == '__main__':
    main()
