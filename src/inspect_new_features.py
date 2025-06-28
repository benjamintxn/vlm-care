#!/usr/bin/env python3
import os
import glob
import numpy as np

# adjust this to wherever your new per-video .npz files are:
NEW_FEAT_DIR = './data/new_features/training/positive'

def main():
    fnames = sorted(glob.glob(os.path.join(NEW_FEAT_DIR, '*.npz')))
    if not fnames:
        print("No .npz files found in", NEW_FEAT_DIR)
        return

    print("Inspecting first 5 of your new feature files:")
    for path in fnames[:5]:
        arr = np.load(path, allow_pickle=True)
        feats = arr['data']   # (n_frames, n_det, feat_dim)
        dets  = arr['det']    # (n_frames, n_det, 4)
        vid   = os.path.splitext(os.path.basename(path))[0]

        print(f"Video ID={vid}:")
        print(f"    feats.shape = {feats.shape}, dets.shape = {dets.shape}")
        print(f"    feats.min/max/mean = {feats.min():.4f}/{feats.max():.4f}/{feats.mean():.4f}")
        print()

if __name__=='__main__':
    main()