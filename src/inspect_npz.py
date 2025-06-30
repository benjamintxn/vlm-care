#!/usr/bin/env python3
"""
inspect_npz.py

Quickly list the contents of one or more .npz feature batches:
  $ python inspect_npz.py data/features/testing/batch_001.npz
  $ python inspect_npz.py data/features/testing/
"""
import os
import argparse
import numpy as np

def inspect_file(path):
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"✗ Could not open {path}: {e}")
        return

    print(f"\n📦 {path}")
    for key in data.files:
        arr = data[key]
        # basic info
        print(f"  • {key!r}: shape={arr.shape}, dtype={arr.dtype}")
        # if it’s an object or string array, show a few samples
        if arr.dtype == object or arr.dtype.kind in {'U','S'}:
            sample = arr.flatten()[:5]
            print(f"      sample values: {sample.tolist()}")
    data.close()

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("paths", nargs="+",
                   help=".npz file(s) or directory(ies) to inspect")
    args = p.parse_args()

    to_inspect = []
    for p in args.paths:
        if os.path.isdir(p):
            for fn in sorted(os.listdir(p)):
                if fn.endswith(".npz"):
                    to_inspect.append(os.path.join(p, fn))
        elif os.path.isfile(p) and p.endswith(".npz"):
            to_inspect.append(p)
        else:
            print(f"⚠️  skipping {p} (not .npz or not found)")

    for fn in to_inspect:
        inspect_file(fn)

if __name__ == "__main__":
    main()