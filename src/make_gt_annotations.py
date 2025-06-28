#!/usr/bin/env python3
import os, glob, argparse
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(
        description="Take feature batches and your per-video .txts and emit annotated batches"
    )
    p.add_argument('--feature-root', required=True,
                   help='Root dir with feature batches under training/{pos,neg} and testing/{pos,neg}')
    p.add_argument('--annotation-dir', required=True,
                   help='Dir of your 620 per-video .txt annotation files (only positives)')
    p.add_argument('--out-dir', required=True,
                   help='Where to write the new annotated batches')
    p.add_argument('--batch-size', type=int, default=10,
                   help='Videos per output batch (should match how you made your feature batches)')
    return p.parse_args()

def load_gt(ann_dir):
    """Load all your positive annotations into a dict vid -> { frame -> set of accident-tracks }."""
    gt = {}
    for fn in glob.glob(os.path.join(ann_dir, '*.txt')):
        vid = os.path.splitext(os.path.basename(fn))[0]
        d = {}
        for L in open(fn):
            f, track, _cls, x1,y1,x2,y2,acc = L.strip().split()
            if int(acc)==1:
                d.setdefault(int(f), set()).add(int(track))
        gt[vid] = d
    return gt

# ─────────── helper: IoU ─────────────────────────────────────────────
def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter  = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    areaA  = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB  = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)

# ─────────── load GT with boxes ──────────────────────────────────────
def load_gt(ann_dir):
    gt = {}
    for fn in glob.glob(os.path.join(ann_dir, '*.txt')):
        vid = os.path.splitext(os.path.basename(fn))[0]
        vid_dict = {}
        for line in open(fn):
            f, _tid, _cls, x1,y1,x2,y2,acc = line.strip().split()
            if int(acc) == 1:                      # accident flag only
                bbox = (int(x1),int(y1),int(x2),int(y2))
                vid_dict.setdefault(int(f), []).append(bbox)
        gt[vid] = vid_dict
    return gt

# ─────────── annotate one batch file ─────────────────────────────────
def annotate_batch(fn, gt, iou_thr=0.5):
    arr    = np.load(fn, allow_pickle=True)
    data   = arr['data'];  dets = arr['det']
    labels = arr['labels']; ids  = arr['ID']
    B, F, D, _ = data.shape
    attn  = np.zeros((B, F, D), np.float32)

    for i in range(B):
        vid = ids[i].decode()
        if labels[i,1] == 0:                          # negative clip
            continue
        vid_gt = gt.get(vid, {})
        for f in range(1, F+1):
            gt_boxes = vid_gt.get(f, [])
            if not gt_boxes: continue
            det_boxes = dets[i, f-1]                 # (D,4)
            for gbox in gt_boxes:
                best = max(range(D), key=lambda j: iou(det_boxes[j], gbox))
                if iou(det_boxes[best], gbox) >= iou_thr:
                    attn[i, f-1, best] = 1.0

    out = dict(arr)          # keep original keys
    out['attn_sup'] = attn
    return out

def main():
    args = parse_args()
    gt = load_gt(args.annotation_dir)
    splits = ['training','testing']
    classes= ['positive','negative']

    for split in splits:
        for cls in classes:
            in_pattern = os.path.join(args.feature_root, split, cls, 'batch_*.npz')
            out_dir    = os.path.join(args.out_dir, split, cls)
            os.makedirs(out_dir, exist_ok=True)

            for fn in sorted(glob.glob(in_pattern)):
                out = annotate_batch(fn, gt, split, cls, args.batch_size)
                base = os.path.basename(fn)
                np.savez(os.path.join(out_dir, base), **out)
                print(f"Annotated {split}/{cls}/{base}")

if __name__=='__main__':
    main()