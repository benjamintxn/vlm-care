#!/usr/bin/env python3
"""
batch_or_inspect_features.py

  • Batch per-clip *.npz feature files into grouped *.npz files that mirror the
    directory structure data/{training,testing}/{positive,negative}/

  • OR: Inspect a small sample of existing batch files so you can
    sanity-check shapes, dtypes, ID fields, etc., before training.

Usage
-----

# 1) Batch features  (same behaviour as before)
python batch_or_inspect_features.py \
       --feature-root ./data/new_features \
       --out-dir      ./data/feature_batches \
       --batch-size   10

# 2) Inspect two batch files per sub-folder
python batch_or_inspect_features.py \
       --feature-root ./data/feature_batches \
       --inspect \
       --num-files 2
"""
import os
import glob
import random
import argparse
import numpy as np


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch feature .npz files OR inspect existing batches."
    )
    p.add_argument(
        "--feature-root",
        required=True,
        help="Root directory where *.npz files live "
             "(either per-clip files or previously batched files).",
    )

    # ----------------------- batching arguments ---------------------------- #
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output dir for batched files (required unless --inspect).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of clips per batch (batch mode only).",
    )

    # ----------------------- inspection arguments -------------------------- #
    p.add_argument(
        "--inspect",
        action="store_true",
        help="If set, do **NOT** batch.  Instead, pick a few existing batch "
             "files and list their contents.",
    )
    p.add_argument(
        "--num-files",
        type=int,
        default=2,
        help="Number of batch files to inspect per split/class folder.",
    )

    return p


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
def batch_features(feature_root: str, out_dir: str, batch_size: int) -> None:
    """Group per-clip feature files into batched .npz files."""
    splits = ["training", "testing"]
    classes = ["positive", "negative"]

    for split in splits:
        for cls in classes:
            pattern = os.path.join(feature_root, split, cls, "*.npz")
            files = sorted(glob.glob(pattern))
            if not files:
                continue

            # Ensure output dir exists
            target_dir = os.path.join(out_dir, split, cls)
            os.makedirs(target_dir, exist_ok=True)

            # Batch
            for i in range(0, len(files), batch_size):
                batch_files = files[i: i + batch_size]

                data_list, det_list, lbl_list, id_list = [], [], [], []
                for fn in batch_files:
                    arr = np.load(fn, allow_pickle=True)
                    data_list.append(arr["data"])        # (F, 20, 4096)
                    det_list.append(arr["det"])          # (F, 20, 4)
                    lbl_list.append(arr["labels"][0])    # (2,)
                    # handle bytes vs. str edge-cases
                    clip_id = arr["ID"].item()
                    if isinstance(clip_id, bytes):
                        clip_id = clip_id.decode("utf-8")
                    id_list.append(clip_id)

                # Stack
                data_stack = np.stack(data_list, axis=0)   # (B, F, 20, 4096)
                det_stack = np.stack(det_list, axis=0)     # (B, F, 20, 4)
                lbl_stack = np.stack(lbl_list, axis=0)     # (B, 2)
                id_array  = np.array(id_list, dtype="S32") # (B,)

                # Save
                batch_idx = (i // batch_size) + 1
                out_file = os.path.join(
                    target_dir, f"batch_{batch_idx:03d}.npz"
                )
                np.savez(
                    out_file,
                    data=data_stack,
                    det=det_stack,
                    labels=lbl_stack,
                    ID=id_array,
                )
                print(f"[+] Wrote {out_file:>60}  ({len(batch_files)} clips)")


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def inspect_batches(feature_root: str, num_files: int = 2) -> None:
    """Print shapes / dtypes / ID values for a sample of batch files."""
    splits = ["training", "testing"]
    classes = ["positive", "negative"]

    for split in splits:
        for cls in classes:
            pattern = os.path.join(feature_root, split, cls, "batch_*.npz")
            all_files = glob.glob(pattern)
            if not all_files:
                continue

            sample = random.sample(
                all_files, k=min(num_files, len(all_files))
            )
            header = f"\n=== {split}/{cls} : inspecting {len(sample)} file(s) ==="
            print(header)

            for fn in sample:
                arr = np.load(fn, allow_pickle=True)
                print(f"\n{fn}")
                for key in arr.files:
                    val = arr[key]
                    print(f"  {key:>8}: shape {val.shape}, dtype {val.dtype}")
                    if key == "ID":
                        # Show the list of IDs for quick sanity-check
                        ids = [x.decode("utf-8") for x in val]
                        print(f"           IDs → {ids}")
                print("-" * 72)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = make_parser().parse_args()

    if args.inspect:
        # Inspection mode
        inspect_batches(args.feature_root, args.num_files)
    else:
        # Batch-creation mode
        if args.out_dir is None:
            raise ValueError("--out-dir is required when not using --inspect.")
        batch_features(args.feature_root, args.out_dir, args.batch_size)


if __name__ == "__main__":
    main()