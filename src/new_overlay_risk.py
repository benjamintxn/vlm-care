#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Overlay per-frame accident risk on dash-cam video using the training graph
# Usage:
#   python new_overlay_risk.py INPUT.mp4 OUTPUT.mp4 --model MODEL_DIR_OR_PREFIX --gpu 0
# ---------------------------------------------------------------------
import os, glob, argparse
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from accident import build_model  # ensure this points to your trained graph definition

tf.disable_v2_behavior()
FEATURE_DIR = "./data/annotated_batches/testing/"
N_CLASSES   = 2
GRAPH_H     = 150

# ---------------------------------------------------------------------
# Find feature batch containing this video
# ---------------------------------------------------------------------
def find_feature_file(video_id, root=FEATURE_DIR):
    # search recursively in positive/negative subfolders
    pattern = os.path.join(root, '**', 'batch_*.npz')
    for path in sorted(glob.glob(pattern, recursive=True)):
        with np.load(path, allow_pickle=True) as data:
            ids = [i.decode() if isinstance(i, (bytes, bytearray)) else str(i)
                   for i in data['ID']]
            if video_id in ids:
                return path, ids.index(video_id)
    return None, None

# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay per-frame risk on a dash-cam video"
    )
    parser.add_argument('video',  help='Path to input MP4')
    parser.add_argument('output', help='Path to output MP4')
    parser.add_argument('--model', default='./model',
                        help='Checkpoint directory or prefix')
    parser.add_argument('--gpu', default='0',
                        help='CUDA_VISIBLE_DEVICES')
    return parser.parse_args()

# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 1) Build the graph as in training
    x_ph, keep_ph, y_ph, attn_ph, _, _, soft_pred, _ = build_model()

    # 2) Prepare saver: only restore variables present in checkpoint
    ckpt = args.model
    if os.path.isdir(ckpt):
        ckpt = tf.train.latest_checkpoint(ckpt)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in {args.model}")
    ckpt_vars = {name for name, _ in tf.train.list_variables(ckpt)}
    var_list  = [v for v in tf.global_variables() if v.op.name in ckpt_vars]
    saver     = tf.train.Saver(var_list=var_list)

    # 3) Create session, init all, then restore
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True)
    ))
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, ckpt)
    print(f"✓ Initialized and restored {len(var_list)} vars from {ckpt}")

    # 4) Load feature batch for the given video
    vid = os.path.splitext(os.path.basename(args.video))[0]
    npz_path, idx = find_feature_file(vid)
    if npz_path is None:
        raise FileNotFoundError(f"No feature batch found for video ID {vid}")
    print(f"✓ Loaded features from {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    feat_batch = data['data']  # shape: (B, N_FRAMES, N_DET, N_INPUT)

        # 5) Run inference to get risk curve
    risk_batch = sess.run(soft_pred, feed_dict=feed)
    print("Risk batch shape:", risk_batch.shape)
    print("Sample risk values:", np.round(risk_batch[idx, :10], 3))

    # Normalize so that 0% risk is baseline at bottom of panel
    raw_curve = risk_batch[idx]
    min_val = raw_curve.min()
    norm_curve = raw_curve - min_val
    max_val = norm_curve.max()
    if max_val > 0:
        norm_curve = norm_curve / max_val
    risk_curve = norm_curve
    print("Normalized risk sample:", np.round(risk_curve[:10], 3))

        # Optional: smooth the normalized risk curve to reduce spikiness
    # Simple moving average with window size w
    w = 5
    kernel = np.ones(w) / w
    risk_curve = np.convolve(risk_curve, kernel, mode='same')

    # 6) Open video and setup writer
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {args.video}")
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # stretch curve if different length
    if total != len(risk_curve):
        xs = np.linspace(0, 1, len(risk_curve))
        xd = np.linspace(0, 1, total)
        risk_curve = np.interp(xd, xs, risk_curve)

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h + GRAPH_H)
    )
    print(f"✓ Writing overlay to {args.output}")

    # 7) Draw frames with overlay
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(risk_curve):
            break

        panel = np.zeros((GRAPH_H, w, 3), np.uint8)
        # draw axes
        cv2.line(panel, (0, GRAPH_H-1), (w, GRAPH_H-1), (200,200,200), 1)
        cv2.line(panel, (0, 0), (0, GRAPH_H-1), (200,200,200), 1)
        # threshold at 0.5
        y_th = GRAPH_H-1 - int(0.5 * (GRAPH_H-1))
        cv2.line(panel, (0, y_th), (w, y_th), (100,100,100), 1)
        # risk polyline
        pts = [(
            int(i * w / len(risk_curve)),
            GRAPH_H-1 - int(risk_curve[i] * (GRAPH_H-1))
        ) for i in range(frame_idx+1)]
        if len(pts) > 1:
            cv2.polylines(panel, [np.array(pts)], False, (0,0,255), 2)
        cv2.putText(panel, "Risk", (5,15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,255), 2)

        out_frame = np.vstack((frame, panel))
        writer.write(out_frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print("✓ Finished overlay")

if __name__ == '__main__':
    main()
