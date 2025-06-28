#!/usr/bin/env python3
import os
import glob
import numpy as np

# Directory containing the original batched .npz files
ORIG_FEAT_DIR = './data/features/training'
# The clip IDs to inspect
TARGET_IDS = [f'{i:06d}' for i in range(1,6)]  # '000001' to '000005'

def main():
    printed = set()
    # search all batch files
    batch_files = sorted(glob.glob(os.path.join(ORIG_FEAT_DIR, 'batch_*.npz')))
    if not batch_files:
        print("No batch_*.npz files found in", ORIG_FEAT_DIR)
        return

    for batch_file in batch_files:
        arr = np.load(batch_file, allow_pickle=True)
        feats = arr['data']    # (B, n_frames, n_det, feat_dim)
        dets  = arr['det']     # (B, n_frames, n_det, 4)
        ids   = arr['ID']      # array of bytestrings
        ids = [i.decode() if isinstance(i, (bytes,bytearray)) else str(i) for i in ids]

        for target in TARGET_IDS:
            if target in printed:
                continue
            if target in ids:
                idx = ids.index(target)
                f = feats[idx]
                b = dets[idx]
                B, NF, ND, FD = feats.shape[0], f.shape[0], f.shape[1], f.shape[2]  # batch size ignored
                print(f"Found {target} in {batch_file} (index {idx}):")
                print(f"  feats.shape = {f.shape}, dets.shape = {b.shape}")
                print(f"  feats.min/max/mean = {f.min():.4f}/{f.max():.4f}/{f.mean():.4f}\n")
                printed.add(target)
                if len(printed) == len(TARGET_IDS):
                    return

    missing = set(TARGET_IDS) - printed
    if missing:
        print("Could not find these IDs in any batch:", sorted(missing))

if __name__=='__main__':
    main()
