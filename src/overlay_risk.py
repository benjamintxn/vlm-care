#!/usr/bin/env python
# ---------------------------------------------------------------------
# Enhanced Risk Visualization with Object-Level Attention
# (Without annotation visualization)
# ---------------------------------------------------------------------
import os, glob, argparse
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from collections import deque

tf.disable_v2_behavior()

# ------------- Configuration ------------------------------------------
FEATURE_DIR = "./data/features/training/"
ANNOTATION_DIR = "./data/raw/annotations/"  # Keep for parsing but won't visualize
N_CLASSES = 2
GRAPH_H = 150
OBJECT_OVERLAY_ALPHA = 0.3  # Transparency for risk heatmap
# ---------------------------------------------------------------------

from accident import build_model

def parse_annotations(video_id):
    """Parse but don't visualize annotations"""
    return {}  # Return empty dict to skip visualization

def calculate_ttc(prev_boxes, current_boxes, fps=30):
    """Approximate Time-to-Collision using bounding box dynamics"""
    if prev_boxes is None or current_boxes is None:
        return 99.9
    
    min_ttc = 99.9
    for i in range(current_boxes.shape[0]):
        if np.all(current_boxes[i] == 0) or (prev_boxes[i] is None) or np.all(prev_boxes[i] == 0):
            continue
            
        dx = current_boxes[i, 0] - prev_boxes[i, 0]
        dy = current_boxes[i, 1] - prev_boxes[i, 1]
        velocity = np.sqrt(dx**2 + dy**2)
        
        height = current_boxes[i, 3] - current_boxes[i, 1]
        if height > 10 and velocity > 0.5:
            ttc = height / velocity / fps
            min_ttc = min(min_ttc, ttc)
            
    return min_ttc

def find_feature_file(video_id: str, root: str = FEATURE_DIR):
    for npz_path in sorted(glob.glob(os.path.join(root, "batch_*.npz"))):
        with np.load(npz_path, allow_pickle=True) as data:
            ids = [i.decode("utf-8") if isinstance(i, (bytes, bytearray)) else str(i)
                   for i in data["ID"]]
            if video_id in ids:
                return npz_path, ids.index(video_id)
    
    return None, None

def parse_args():
    ap = argparse.ArgumentParser("Accident Risk Visualization")
    ap.add_argument("video",  help="input .mp4 path")
    ap.add_argument("output", help="output .mp4 path")
    ap.add_argument("--model", default="./model/demo_model",
                    help="checkpoint prefix OR directory")
    ap.add_argument("--gpu",   default="0", help="CUDA_VISIBLE_DEVICES id")
    ap.add_argument("--show_ttc", action="store_true", 
                    help="Display Time-to-Collision metrics")
    return ap.parse_args()

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 1. Build graph & restore weights
    x_ph, keep_ph, y_ph, _, _, _, soft_pred, all_alphas = build_model()

    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True)))
    saver = tf.train.Saver()

    ckpt = args.model
    if os.path.isdir(ckpt):
        ckpt = tf.train.latest_checkpoint(ckpt)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in {args.model}")
    saver.restore(sess, ckpt)
    print("✓ Restored weights from", ckpt)

    # 2. Load features (skip annotation visualization)
    vid_path = args.video
    vid_id = os.path.splitext(os.path.basename(vid_path))[0]
    
    npz_path, idx_in_batch = find_feature_file(vid_id)
    if npz_path is None:
        raise FileNotFoundError(f"Features for {vid_id} not found")
    print("✓ Using feature file:", npz_path)

    with np.load(npz_path, allow_pickle=True) as npz:
        feat_batch = npz["data"]
        det_batch = npz["det"]

    # Add this after loading features in overlay_risk.py
    print("Feature stats - Mean:", np.mean(feat_batch), "Std:", np.std(feat_batch))

    detections = det_batch[idx_in_batch]
    prev_boxes = None
    ttc_history = deque(maxlen=10)

    # 3. Run inference
    dummy_y = np.zeros((feat_batch.shape[0], N_CLASSES), np.float32)
    risk_batch, alpha_batch = sess.run(
        [soft_pred, all_alphas],
        feed_dict={x_ph: feat_batch, keep_ph: [0.0], y_ph: dummy_y}
    )
    risk_curve = risk_batch[idx_in_batch]
    attentions = alpha_batch[:, :, idx_in_batch] if alpha_batch.ndim == 3 else alpha_batch

    # 4. Prepare video I/O
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {vid_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames != len(risk_curve):
        xs_src = np.linspace(0, 1, len(risk_curve))
        xs_dst = np.linspace(0, 1, total_frames)
        risk_curve = np.interp(xs_dst, xs_src, risk_curve)
        attentions = np.vstack([attentions, [attentions[-1]]*(total_frames - len(attentions))])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h + GRAPH_H))
    print("✓ Writing visualization to", args.output)

    # 5. Frame processing - simplified without annotations
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= total_frames:
            break

        current_risk = risk_curve[frame_idx]
        current_attn = attentions[frame_idx]
        current_dets = detections[frame_idx]

        # Create object risk heatmap overlay only
        overlay = frame.copy()
        for obj_idx in range(len(current_attn)):
            attn_score = current_attn[obj_idx]
            if attn_score < 0.1:
                continue
                
            box = current_dets[obj_idx][:4]
            x0, y0, x1, y1 = map(int, box)
            color = (0, 0, int(255 * min(attn_score, 1.0)))
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
        
        frame = cv2.addWeighted(overlay, OBJECT_OVERLAY_ALPHA, frame, 1 - OBJECT_OVERLAY_ALPHA, 0)

        # TTC display (optional)
        if args.show_ttc and frame_idx > 0:
            ttc = calculate_ttc(prev_boxes, current_dets, fps)
            ttc_history.append(ttc)
            smoothed_ttc = sum(ttc_history) / len(ttc_history)
            cv2.putText(frame, f"TTC: {smoothed_ttc:.1f}s", (w - 200, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        prev_boxes = current_dets

        # Risk graph panel
        panel = np.zeros((GRAPH_H, w, 3), np.uint8)
        cv2.line(panel, (0, GRAPH_H-1), (w, GRAPH_H-1), (200, 200, 200), 1)
        cv2.line(panel, (0, 0), (0, GRAPH_H-1), (200, 200, 200), 1)
        y_th = GRAPH_H-1 - int(0.5 * (GRAPH_H-1))
        cv2.line(panel, (0, y_th), (w, y_th), (100, 100, 100), 1)

        pts = [(int(i * w / total_frames), 
               GRAPH_H-1 - int(r * (GRAPH_H-1)))
              for i, r in enumerate(risk_curve[:frame_idx+1])]
        
        if len(pts) > 1:
            cv2.polylines(panel, [np.array(pts)], False, (0, 0, 255), 2)

        if pts:
            cv2.circle(panel, pts[-1], 5, (0, 255, 0), -1)

        cv2.putText(panel, f"Frame: {frame_idx}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(panel, f"Risk: {current_risk:.2f}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        out_frame = np.vstack((frame, panel))
        out.write(out_frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("✓ Visualization complete!")

if __name__ == "__main__":
    main()