#!/usr/bin/env python3
"""Concatenate result CSVs into a single CSV.

Usage:
  python scripts/concat_results.py --input-dir results --pattern 'result_*.csv' --out combined.csv --add-source

By default this will look for `results/result_*.csv` and write `results/combined_results.csv`.
"""
import argparse
import glob
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Concatenate result CSV files into one CSV")
    parser.add_argument("--input-dir", type=str, default="results", help="Directory containing result CSV files")
    parser.add_argument("--pattern", type=str, default="result_*.csv", help="Glob pattern for result files")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (defaults to <input-dir>/combined_results.csv)")
    parser.add_argument("--add-source", action="store_true", default=True, help="Add a column `source_file` with the original filename")
    args = parser.parse_args()

    inp = os.path.abspath(args.input_dir)
    pattern = os.path.join(inp, args.pattern)
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found for pattern: {pattern}")
        return

    dfs = []
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
            continue
        if args.add_source:
            df["source_file"] = os.path.basename(p)
        dfs.append(df)

    if not dfs:
        print("No readable CSV files found.")
        return

    # Concatenate and reset index
    out_df = pd.concat(dfs, ignore_index=True, sort=False)

    out_path = args.out or os.path.join(inp, "combined_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Wrote combined CSV with {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
