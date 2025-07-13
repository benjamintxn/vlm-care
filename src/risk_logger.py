# risk_logger.py

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
        description="Log per-frame risk+attention for one video"
    )
    p.add_argument('video',   help='Path to input mp4 (e.g. ./dataset/.../000469.mp4)')
    p.add_argument('out_csv', default='./outputs/risk_logs/' , help='Where to write the CSV log')
    p.add_argument('--model',  default='./model/',
                   help='Checkpoint dir or prefix')
    p.add_argument('--gpu',    default='0', help='CUDA_VISIBLE_DEVICES id')
    return p.parse_args()

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 1) Build TF graph
    x_ph, keep_ph, y_ph, is_train_ph, _, _, soft_pred, all_alphas = build_model()
    saver = tf.train.Saver()

    # 2) Launch session & restore weights
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True)
    ))
    ckpt = args.model
    if os.path.isdir(ckpt):
        ckpt = tf.train.latest_checkpoint(ckpt)
    saver.restore(sess, ckpt)
    print("Restored", ckpt)

    # 3) Find feature‐batch .npz containing our video ID
    vid_name = os.path.splitext(os.path.basename(args.video))[0]
    feature_file = None
    for fn in sorted(glob.glob('./data/features/testing/batch_*.npz')):
        ids = np.load(fn, allow_pickle=True)['ID']
        # decode bytes → str
        ids = [i.decode() if isinstance(i, (bytes,bytearray)) else str(i) for i in ids]
        if vid_name in ids:
            feature_file = fn
            break
    if feature_file is None:
        raise FileNotFoundError(f"No feature batch for {vid_name}")
    print("Using features from", feature_file)

    # 4) Load data
    data = np.load(feature_file, allow_pickle=True)
    feats = data['data']   # shape (batch_size, n_frames, n_detection, n_input)
    dets  = data['det']    # shape (batch_size, n_frames, n_detection, 4)
    ids   = data['ID']
    ids   = [i.decode() if isinstance(i, (bytes,bytearray)) else str(i) for i in ids]
    idx   = ids.index(vid_name)

    # 5) Dummy labels just to satisfy the y_ph placeholder
    feature_batch = data['data']                # shape: (bs, n_frames, ...)
    bs = feature_batch.shape[0]
    dummy_y    = np.zeros((bs, n_classes), dtype=np.float32)
    dummy_keep = np.zeros((bs,), dtype=np.float32)

    # 6) Run inference
    risk_batch, alpha_batch = sess.run(
        [soft_pred, all_alphas],
        feed_dict={
            x_ph:        feature_batch,
            y_ph:        dummy_y,
            keep_ph:     dummy_keep
        }
    )
    # risk_batch shape: (batch_size, n_frames)
    # alpha_batch shape: (n_frames, n_detection-1, batch_size)
    risk_curve = risk_batch[idx]              # (n_frames,)
    attentions = alpha_batch[:,:,idx]         # (n_frames, n_detection-1)
    bboxes     = dets[idx]                    # (n_frames, n_detection, 4)

    # 7) Write CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        # header
        header = ['frame','risk']
        for k in range(3):
            header += [
                f'att{k+1}_score',
                f'att{k+1}_x0', f'att{k+1}_y0',
                f'att{k+1}_x1', f'att{k+1}_y1'
            ]
        w.writerow(header)

        # rows
        for fr in range(n_frames):
            row = [fr, float(risk_curve[fr])]
            scores = attentions[fr]         # length n_detection-1
            # pick top-3 boxes
            top3 = np.argsort(-scores)[:3]
            for box_idx in top3:
                score = float(scores[box_idx])
                coords = bboxes[fr, box_idx]
                x0, y0, x1, y1 = coords[:4]
                row += [score, int(x0), int(y0), int(x1), int(y1)]
            w.writerow(row)

    print("Wrote log to", args.out_csv)


if __name__=='__main__':
    main()