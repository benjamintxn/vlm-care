#!/usr/bin/env python
# ---------------------------------------------------------------------
# Batch Accident Risk Visualization with Object‐Level Attention
# ---------------------------------------------------------------------
import os, glob, argparse
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from collections import deque

tf.disable_v2_behavior()

# ------------- Configuration ------------------------------------------
FEATURE_DIR = "./data/features/testing/"
N_CLASSES = 2
GRAPH_H = 150
OBJECT_OVERLAY_ALPHA = 0.3  # Transparency for risk heatmap
# ---------------------------------------------------------------------

from accident import build_model

def find_feature_file(video_id: str, root: str = FEATURE_DIR):
    for npz_path in sorted(glob.glob(os.path.join(root, "batch_*.npz"))):
        with np.load(npz_path, allow_pickle=True) as data:
            ids = [i.decode("utf-8") if isinstance(i, (bytes, bytearray)) else str(i)
                   for i in data["ID"]]
            if video_id in ids:
                return npz_path, ids.index(video_id)
    return None, None

def calculate_ttc(prev_boxes, current_boxes, fps=30):
    if prev_boxes is None or current_boxes is None:
        return 99.9
    min_ttc = 99.9
    for i in range(current_boxes.shape[0]):
        if np.all(current_boxes[i]==0) or prev_boxes[i] is None:
            continue
        dx = current_boxes[i,0] - prev_boxes[i,0]
        dy = current_boxes[i,1] - prev_boxes[i,1]
        vel = np.hypot(dx,dy)
        h = current_boxes[i,3] - current_boxes[i,1]
        if h>10 and vel>0.5:
            ttc = h/vel/fps
            min_ttc = min(min_ttc, ttc)
    return min_ttc

def parse_args():
    p = argparse.ArgumentParser("Batch Accident Risk Visualization")
    p.add_argument("video_dir",
                   help="root folder containing `positive/` and `negative/` subfolders")
    p.add_argument("out_dir",
                   help="where to write the overlaid videos, preserving structure")
    p.add_argument("--model", default="./model/demo_model",
                   help="checkpoint prefix OR directory")
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES id")
    p.add_argument("--show_ttc", action="store_true",
                   help="Display Time-to-Collision metrics")
    return p.parse_args()

def process_single(video_path, out_path, sess, ckpt, model_tensors):
    x_ph, keep_ph, y_ph, _, _, _, soft_pred, all_alphas = model_tensors

    # 1) find feature batch
    vid_id = os.path.splitext(os.path.basename(video_path))[0]
    npz_path, idx = find_feature_file(vid_id)
    if npz_path is None:
        print(f"⚠️  No features for {vid_id}, skipping")
        return
    with np.load(npz_path, allow_pickle=True) as npz:
        feats = npz["data"]
        dets  = npz["det"]
    # 2) inference
    dummy_y = np.zeros((feats.shape[0], N_CLASSES), np.float32)
    risk_b, alpha_b = sess.run([soft_pred, all_alphas],
        feed_dict={x_ph:feats, keep_ph:[0.0], y_ph:dummy_y})
    risk_curve = risk_b[idx]
    # handle attention dims
    attentions = alpha_b[:,:,idx] if alpha_b.ndim==3 else alpha_b
    detections = dets[idx]

    # 3) open video + writer
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # stretch to match frames if needed
    if total != len(risk_curve):
        xs = np.linspace(0,1,len(risk_curve))
        xd = np.linspace(0,1,total)
        risk_curve = np.interp(xd,xs,risk_curve)
        attentions = np.vstack([attentions, attentions[-1:]]*(total-len(attentions)))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H+GRAPH_H))

    prev_boxes = None
    ttc_hist = deque(maxlen=10)
    idx_frame=0
    while True:
        ret, frame = cap.read()
        if not ret or idx_frame>=len(risk_curve): break
        # overlay objects
        overlay = frame.copy()
        for i,att in enumerate(attentions[idx_frame]):
            if att<0.1: continue
            x0,y0,x1,y1 = map(int, detections[idx_frame][i][:4])
            color = (0,0,int(255*min(att,1.0)))
            cv2.rectangle(overlay,(x0,y0),(x1,y1),color,-1)
        frame = cv2.addWeighted(overlay, OBJECT_OVERLAY_ALPHA, frame,1-OBJECT_OVERLAY_ALPHA,0)
        # ttc
        if args.show_ttc and prev_boxes is not None:
            t = calculate_ttc(prev_boxes, detections[idx_frame], fps)
            ttc_hist.append(t)
            cv2.putText(frame,f"TTC:{np.mean(ttc_hist):.1f}s",(W-200,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        prev_boxes = detections[idx_frame]
        # risk panel
        panel = np.zeros((GRAPH_H,W,3),np.uint8)
        # axes + thr
        cv2.line(panel,(0,GRAPH_H-1),(W,GRAPH_H-1),(200,200,200),1)
        ythr= GRAPH_H-1-int(0.5*(GRAPH_H-1))
        cv2.line(panel,(0,ythr),(W,ythr),(100,100,100),1)
        pts=[(int(i*W/len(risk_curve)),GRAPH_H-1-int(r*(GRAPH_H-1)))
             for i,r in enumerate(risk_curve[:idx_frame+1])]
        if len(pts)>1: cv2.polylines(panel,[np.array(pts)],False,(0,0,255),2)
        if pts: cv2.circle(panel,pts[-1],5,(0,255,0),-1)
        cv2.putText(panel,f"Risk:{risk_curve[idx_frame]:.2f}",(10,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
        # stack & write
        out.write( np.vstack((frame,panel)) )
        idx_frame+=1

    cap.release(); out.release()
    print(f"✓ wrote {out_path}")

if __name__=="__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    # build once
    model_tensors = build_model()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    saver = tf.train.Saver()
    ckpt = args.model
    if os.path.isdir(ckpt):
        ckpt = tf.train.latest_checkpoint(ckpt)
    saver.restore(sess, ckpt)
    print("✓ Restored", ckpt)

    # walk both subdirs
    for sub in ["positive","negative"]:
        in_dir  = os.path.join(args.video_dir, sub)
        out_sub = os.path.join(args.out_dir,   sub)
        for vid in sorted(glob.glob(os.path.join(in_dir,"*.mp4"))):
            rel = os.path.relpath(vid, args.video_dir)
            outp= os.path.join(args.out_dir, rel)
            process_single(vid, outp, sess, ckpt, model_tensors)
    print("🏁 All done!")