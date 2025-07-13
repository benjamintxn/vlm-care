#!/usr/bin/env python3
# risk_logger.py  ────────────────────────────────────────────
# Log per-frame risk+attention for all videos in testing/positive and testing/negative
#
import argparse, os, glob, csv
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from accident import build_model, n_classes   # or from new_training import build_model

# constants (must match training)
n_frames     = 100
n_detection  = 20

def parse_args():
    p = argparse.ArgumentParser(
        description="Log per-frame risk+attention for all test videos"
    )
    p.add_argument('--video_dir', default='./data/raw/videos/testing/',
                   help='Root folder containing "positive" and "negative" subdirs')
    p.add_argument('--out_dir', default='./outputs/risk_logs/',
                   help='Directory where CSV logs will be written')
    p.add_argument('--model',  default='./model/',
                   help='Checkpoint dir or prefix')
    p.add_argument('--gpu',    default='0', help='CUDA_VISIBLE_DEVICES id')
    return p.parse_args()

def find_feature_file(video_id, root='./data/features/testing/'):
    for fn in sorted(glob.glob(os.path.join(root, "batch_*.npz"))):
        with np.load(fn, allow_pickle=True) as data:
            ids = data['ID']
            ids = [i.decode() if isinstance(i,(bytes,bytearray)) else str(i) for i in ids]
            if video_id in ids:
                return fn, ids.index(video_id)
    return None, None

def process_one(video_path, out_csv, sess, x_ph, keep_ph, y_ph, soft_pred, all_alphas):
    vid_name = os.path.splitext(os.path.basename(video_path))[0]
    npz_path, idx = find_feature_file(vid_name)
    if npz_path is None:
        print(f"[!] No features for {vid_name}, skipping.")
        return
    with np.load(npz_path, allow_pickle=True) as data:
        feats = data['data']   # (B, frames, det, feat)
        dets  = data['det']    # (B, frames, det, 4)
        ids   = data['ID']
        ids   = [i.decode() if isinstance(i,(bytes,bytearray)) else str(i) for i in ids]

    bs = feats.shape[0]
    dummy_y    = np.zeros((bs, n_classes), np.float32)
    dummy_keep = np.zeros((bs,),      np.float32)

    risk_batch, alpha_batch = sess.run(
        [soft_pred, all_alphas],
        feed_dict={
            x_ph:    feats,
            y_ph:    dummy_y,
            keep_ph: dummy_keep
        }
    )
    # shapes:
    #  - risk_batch: (B, frames)
    #  - alpha_batch: (frames, det-1, B)
    #  - dets:       (B, frames, det, 4)

    risk_curve = risk_batch[idx]            # (frames,)
    attentions = alpha_batch[:, :, idx]     # (frames, det-1)
    bboxes     = dets[idx]                  # (frames, det, 4)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['frame','risk']
        for k in range(3):
            header += [
                f'att{k+1}_score',
                f'att{k+1}_x0', f'att{k+1}_y0',
                f'att{k+1}_x1', f'att{k+1}_y1'
            ]
        w.writerow(header)

        for fr in range(n_frames):
            row = [fr, float(risk_curve[fr])]
            scores = attentions[fr]         # length n_detection-1
            top3   = np.argsort(-scores)[:3]
            for bi in top3:
                score = float(scores[bi])
                x0, y0, x1, y1 = bboxes[fr, bi, :4]  # note: bbox index aligns
                row += [score, int(x0), int(y0), int(x1), int(y1)]
            w.writerow(row)

    print(f"✓ Logged {vid_name} → {out_csv}")

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # build & restore
    x_ph, keep_ph, y_ph, _, _, _, soft_pred, all_alphas = build_model()
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True)
    ))
    ckpt = args.model
    if os.path.isdir(ckpt):
        ckpt = tf.train.latest_checkpoint(ckpt)
    saver.restore(sess, ckpt)
    print("✓ Restored", ckpt)

    # gather all videos
    vids = sorted(glob.glob(os.path.join(args.video_dir, 'positive','*.mp4')))
    vids += sorted(glob.glob(os.path.join(args.video_dir, 'negative','*.mp4')))

    for video_path in vids:
        vid_name = os.path.splitext(os.path.basename(video_path))[0]
        out_csv = os.path.join(args.out_dir, f"{vid_name}.csv")
        process_one(video_path, out_csv, sess,
                    x_ph, keep_ph, y_ph, soft_pred, all_alphas)

    sess.close()
    print("✓ All done.")

if __name__ == '__main__':
    main()