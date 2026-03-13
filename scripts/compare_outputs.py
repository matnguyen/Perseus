#!/usr/bin/env python3

"""
Compare two Perseus output files with tolerance for floating-point differences.

Usage
-----
python scripts/compare_outputs.py file1.txt file2.txt

Exit codes
----------
0 : files match within tolerance
1 : files differ
"""

import sys
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

RTOL = 1e-4
ATOL = 1e-5


def load_file(path):
    return pd.read_csv(path, sep="\t")


def compare_dataframes(df1, df2):
    if df1.shape != df2.shape:
        print(f"Shape mismatch: {df1.shape} vs {df2.shape}")
        return False

    if list(df1.columns) != list(df2.columns):
        print("Column mismatch:")
        print("file1 columns:", list(df1.columns))
        print("file2 columns:", list(df2.columns))
        return False

    for col in df1.columns:
        s1 = df1[col]
        s2 = df2[col]

        if is_numeric_dtype(s1):
            a = s1.to_numpy()
            b = s2.to_numpy()

            if not np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
                diff = np.abs(a - b)
                idx = int(np.nanargmax(diff))
                print(f"Numeric mismatch in column '{col}'")
                print(f"Row: {idx}")
                print(f"file1: {a[idx]}")
                print(f"file2: {b[idx]}")
                print(f"abs diff: {diff[idx]}")
                return False
        else:
            # Treat missing values consistently before comparison
            a = s1.fillna("<NA>").astype(str)
            b = s2.fillna("<NA>").astype(str)

            mismatches = a != b
            if mismatches.any():
                idx = int(np.flatnonzero(mismatches.to_numpy())[0])
                print(f"Mismatch in column '{col}'")
                print(f"Row: {idx}")
                print(f"file1: {a.iloc[idx]}")
                print(f"file2: {b.iloc[idx]}")
                return False

    return True


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_outputs.py file1 file2")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    df1 = load_file(file1)
    df2 = load_file(file2)

    ok = compare_dataframes(df1, df2)

    if ok:
        print("✓ Outputs match within tolerance")
        sys.exit(0)
    else:
        print("✗ Outputs differ")
        sys.exit(1)


if __name__ == "__main__":
    main()