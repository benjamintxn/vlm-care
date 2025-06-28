#!/usr/bin/env python3
import os, glob, argparse
import numpy as np

def make_parser():
    p = argparse.ArgumentParser(description="Batch feature .npz files into grouped .npz with same directory structure")
    p.add_argument("--feature-root", required=True,
                   help="Root directory where per-clip .npz features live (e.g. ./data/new_features)")
    p.add_argument("--out-dir", required=True,
                   help="Where to write batched feature files")
    p.add_argument("--batch-size", type=int, default=10,
                   help="Number of clips per batch")
    return p


def batch_features(feature_root, out_dir, batch_size):
    # Expected subfolders: training/{positive,negative}, testing/{positive,negative}
    splits = ["training", "testing"]
    classes = ["positive", "negative"]

    for split in splits:
        for cls in classes:
            fg = os.path.join(feature_root, split, cls, "*.npz")
            files = sorted(glob.glob(fg))
            if not files:
                continue

            # ensure output subfolder exists
            out_sub = os.path.join(out_dir, split, cls)
            os.makedirs(out_sub, exist_ok=True)

            # batch them
            for i in range(0, len(files), batch_size):
                batch_files = files[i:i+batch_size]
                batch_data = []
                batch_det  = []
                batch_lbl  = []
                batch_id   = []
                for fn in batch_files:
                    arr = np.load(fn, allow_pickle=True)
                    batch_data.append(arr['data'])   # shape (F, D, C)
                    batch_det.append(arr['det'])     # shape (F, D, 4)
                    labels = arr['labels'].reshape(-1,2)  # (1,2)
                    batch_lbl.append(labels[0])
                    ID = arr['ID'].item() if arr['ID'].dtype.type is np.bytes_ else arr['ID'].item()
                    batch_id.append(ID)

                # stack into arrays
                data_stack = np.stack(batch_data, axis=0)  # (B, F, D, C)
                det_stack  = np.stack(batch_det,  axis=0)  # (B, F, D, 4)
                lbl_stack  = np.stack(batch_lbl,  axis=0)  # (B, 2)
                id_array   = np.array(batch_id, dtype='S6')  # (B,)

                # filename: batch_{split}_{cls}_{idx:03d}.npz
                batch_idx = i // batch_size + 1
                out_fn = os.path.join(out_sub, f"batch_{batch_idx:03d}.npz")
                np.savez(out_fn,
                         data=data_stack,
                         det=det_stack,
                         labels=lbl_stack,
                         ID=id_array)
                print(f"Wrote {out_fn}  ({len(batch_files)} clips)")


def main():
    args = make_parser().parse_args()
    batch_features(args.feature_root, args.out_dir, args.batch_size)

if __name__ == '__main__':
    main()