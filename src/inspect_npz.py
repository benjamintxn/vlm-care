#!/usr/bin/env python3
"""
inspect_npz.py

Quickly list the contents of one or more .npz feature batches,
or find which batch contains a specific video ID.

Usage:
  # just inspect files
  python inspect_npz.py data/features/testing/batch_001.npz
  python inspect_npz.py data/features/testing/

  # search for video ID "000456"
  python inspect_npz.py data/features/testing/ --find 000456
"""
import os
import argparse
import numpy as np

def inspect_file(path):
    """Print keys/shapes/dtypes in a single .npz"""
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"✗ Could not open {path}: {e}")
        return
    print(f"\n📦 {path}")
    for key in data.files:
        arr = data[key]
        print(f"  • {key!r}: shape={arr.shape}, dtype={arr.dtype}")
        if arr.dtype == object or arr.dtype.kind in {'U','S'}:
            sample = arr.flatten()[:5]
            print(f"      sample values: {sample.tolist()}")
    data.close()

def find_in_file(path, video_id):
    """Check if 'ID' array in .npz contains video_id (as str or bytes)."""
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    if "ID" not in data:
        data.close()
        return None

    ids = data["ID"]
    # decode bytes to str if necessary
    ids = [i.decode() if isinstance(i, (bytes, bytearray)) else str(i) for i in ids]
    data.close()
    for idx, vid in enumerate(ids):
        if vid == video_id:
            return idx
    return None

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("paths", nargs="+",
                        help=".npz file(s) or directory(ies) to inspect")
    parser.add_argument("--find", metavar="VIDEO_ID",
                        help="if set, only search for this video ID (instead of dumping contents)")
    args = parser.parse_args()

    # collect all .npz under the given paths
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

    if args.find:
        found_any = False
        for fn in to_inspect:
            idx = find_in_file(fn, args.find)
            if idx is not None:
                print(f"✅ Found ID '{args.find}' in {fn} at index {idx}")
                found_any = True
        if not found_any:
            print(f"❌ VIDEO ID '{args.find}' not found in any of the scanned batches.")
    else:
        for fn in to_inspect:
            inspect_file(fn)

if __name__ == "__main__":
    main()