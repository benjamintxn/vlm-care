#!/usr/bin/env python
# ---------------------------------------------------------------------
# Enhanced Risk Visualization with Object-Level Attention and Accident Markers
#
# Improvements:
# 1. Object-level risk visualization using attention weights
# 2. Accident involvement markers from annotation files
# 3. Time-to-Collision (TTC) display
# 4. Physics-based risk validation
# 5. Frame-specific risk information overlay
# ---------------------------------------------------------------------
import os, glob, argparse
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from collections import deque

tf.disable_v2_behavior()

# ------------- Configuration ------------------------------------------
FEATURE_DIR = "./data/features/testing/"
ANNOTATION_DIR = "./data/raw/annotations/"  # Directory for annotation files
N_CLASSES = 2
GRAPH_H = 150
OBJECT_OVERLAY_ALPHA = 0.3  # Transparency for risk heatmap
# ---------------------------------------------------------------------

from new_accident import build_model

def parse_annotations(video_id):
    ann_path = os.path.join(ANNOTATION_DIR, f"{video_id}.txt")
    frame_objs = {}
    
    if not os.path.exists(ann_path):
        return frame_objs
    
    with open(ann_path, 'r') as f:
        for line in f:
            # Handle tab-separated values
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue
                
            # Convert frame number (000001 -> 0)
            frame_num = int(parts[0]) - 1
            is_accident = int(parts[7])
            
            if is_accident == 1:
                if frame_num not in frame_objs:
                    frame_objs[frame_num] = []
                frame_objs[frame_num].append({
                    'track_id': int(parts[1]),
                    'class': parts[2].strip(),
                    'bbox': [float(x) for x in parts[3:7]]  # Convert to floats
                })
    
    print(f"Parsed {sum(len(v) for v in frame_objs.values())} accident objects")
    return frame_objs

def calculate_ttc(prev_boxes, current_boxes, fps=30):
    """Approximate Time-to-Collision using bounding box dynamics"""
    if prev_boxes is None or current_boxes is None:
        return 99.9
    
    min_ttc = 99.9
    for i in range(current_boxes.shape[0]):
        if np.all(current_boxes[i] == 0) or (prev_boxes[i] is None) or np.all(prev_boxes[i] == 0):
            continue
            
        # Calculate relative movement (pixels/frame)
        dx = current_boxes[i, 0] - prev_boxes[i, 0]
        dy = current_boxes[i, 1] - prev_boxes[i, 1]
        velocity = np.sqrt(dx**2 + dy**2)
        
        # Use bounding box height as distance proxy
        height = current_boxes[i, 3] - current_boxes[i, 1]
        if height > 10 and velocity > 0.5:  # Ignore small boxes and minimal movement
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
    ap = argparse.ArgumentParser("Enhanced Accident Risk Visualization")
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
    # Get attention weights from the model
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

    # 2. Load features and annotations
    vid_path = args.video
    vid_id = os.path.splitext(os.path.basename(vid_path))[0]
    
    # Load annotation data for accident objects
    accident_objects = parse_annotations(vid_id)
    print(f"✓ Loaded annotations for {len(accident_objects)} accident frames")

    npz_path, idx_in_batch = find_feature_file(vid_id)
    if npz_path is None:
        raise FileNotFoundError(f"Features for {vid_id} not found")
    print("✓ Using feature file:", npz_path)

    with np.load(npz_path, allow_pickle=True) as npz:
        feat_batch = npz["data"]
        det_batch = npz["det"]  # Detection boxes [batch, frames, objects, 4]

    # Extract detection boxes for this video
    detections = det_batch[idx_in_batch]  # [frames, objects, 4]
    prev_boxes = None
    ttc_history = deque(maxlen=10)  # For smoothing TTC values

    # 3. Run inference to get risk and attention
    dummy_y = np.zeros((feat_batch.shape[0], N_CLASSES), np.float32)
    risk_batch, alpha_batch = sess.run(
        [soft_pred, all_alphas],
        feed_dict={x_ph: feat_batch, keep_ph: [0.0], y_ph: dummy_y}
    )
    risk_curve = risk_batch[idx_in_batch]  # [frames]
    if alpha_batch.ndim == 3:
        attentions = alpha_batch[:, :, idx_in_batch]        # old demo model
    else:  # ndim == 2
        attentions = alpha_batch  

    # 4. Prepare video I/O
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {vid_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Handle frame count mismatch
    if total_frames != len(risk_curve):
        xs_src = np.linspace(0, 1, len(risk_curve))
        xs_dst = np.linspace(0, 1, total_frames)
        risk_curve = np.interp(xs_dst, xs_src, risk_curve)
        # Repeat last attention for additional frames
        attentions = np.vstack([attentions, [attentions[-1]]*(total_frames - len(attentions))])
        detections = np.vstack([detections, [detections[-1]]*(total_frames - len(detections))])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h + GRAPH_H))
    print("✓ Writing enhanced visualization to", args.output)

    # 5. Frame-by-frame enhanced overlay
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= total_frames:
            break

        current_risk = risk_curve[frame_idx]
        current_attn = attentions[frame_idx]
        current_dets = detections[frame_idx]

        # Create object risk heatmap overlay
        overlay = frame.copy()
        for obj_idx in range(len(current_attn)):
            attn_score = current_attn[obj_idx]
            if attn_score < 0.1:  # Only visualize significant risks
                continue
                
            # Get bounding box coordinates
            box = current_dets[obj_idx][:4]          # keep only x0,y0,x1,y1
            x0, y0, x1, y1 = map(int, box)
            
            # Draw semi-transparent overlay
            color = (0, 0, int(255 * min(attn_score, 1.0)))
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
        
        # Blend overlay with original frame
        frame = cv2.addWeighted(overlay, OBJECT_OVERLAY_ALPHA, frame, 1 - OBJECT_OVERLAY_ALPHA, 0)

        # Mark accident-involved objects
        for fr in range(total_frames):
            if fr in accident_objects:
                for obj in accident_objects[fr]:
                    try:
                        # Safely extract coordinates
                        x0, y0, x1, y1 = [int(round(coord)) for coord in obj['bbox']]
                        
                        # Draw bounding box (yellow for accidents)
                        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
                        
                        # Label with class and track ID
                        label = f"{obj['class']} {obj['track_id']}"
                        cv2.putText(frame, label, (x0, y0-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                        
                    except Exception as e:
                        print(f"Error drawing object {obj} at frame {fr}: {str(e)}")
                        continue
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0,255,255), 3)

        # Calculate and display TTC
        if args.show_ttc and frame_idx > 0:
            ttc = calculate_ttc(prev_boxes, current_dets, fps)
            ttc_history.append(ttc)
            smoothed_ttc = sum(ttc_history) / len(ttc_history)
            
            ttc_text = f"TTC: {smoothed_ttc:.1f}s"
            risk_text = f"Risk: {current_risk:.2f}"
            
            cv2.putText(frame, ttc_text, (w - 200, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, risk_text, (w - 200, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (0, 0, 255) if current_risk > 0.5 else (0, 255, 0), 2)

        prev_boxes = current_dets  # Store for next frame

        # Create risk graph panel
        panel = np.zeros((GRAPH_H, w, 3), np.uint8)

        # Draw axes
        cv2.line(panel, (0, GRAPH_H-1), (w, GRAPH_H-1), (200, 200, 200), 1)
        cv2.line(panel, (0, 0), (0, GRAPH_H-1), (200, 200, 200), 1)

        # Threshold line
        y_th = GRAPH_H-1 - int(0.5 * (GRAPH_H-1))
        cv2.line(panel, (0, y_th), (w, y_th), (100, 100, 100), 1)

        # Risk curve
        pts = [(int(i * w / total_frames), 
               GRAPH_H-1 - int(r * (GRAPH_H-1)))
              for i, r in enumerate(risk_curve[:frame_idx+1])]
        
        if len(pts) > 1:
            cv2.polylines(panel, [np.array(pts)], False, (0, 0, 255), 2)

        # Current position marker
        if pts:
            cv2.circle(panel, pts[-1], 5, (0, 255, 0), -1)

        # Text labels
        cv2.putText(panel, f"Frame: {frame_idx}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(panel, f"Risk: {current_risk:.2f}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Stack and write frame
        out_frame = np.vstack((frame, panel))
        out.write(out_frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("✓ Enhanced visualization complete!")

if __name__ == "__main__":
    main()