#!/usr/bin/env python
# ---------------------------------------------------------------------
# Overlay the per–time-step accident risk produced by accident.py
# on top of a dash-cam video, writing out a new .mp4 clip.
#
# Usage:
#   python overlay_risk.py \
#       ./dataset/videos/testing/positive/000469.mp4 \
#       ./output/000469_with_risk.mp4 \
#       --model ./model/demo_model            # or a checkpoint dir
#       --gpu   0
# ---------------------------------------------------------------------
import os, glob, argparse
import cv2
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()          # keep TF-1 semantics

# ------------- project-specific paths --------------------------------
FEATURE_DIR = "./data/features/testing/"   # *.npz files
N_CLASSES   = 2
GRAPH_H     = 150                             # pixels for risk panel
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Helper: locate which batch_XXX.npz contains this video’s features
# ---------------------------------------------------------------------
def find_feature_file(video_id: str, root: str = FEATURE_DIR):
    """
    Return (npz_path, index_in_batch) or (None, None) if not found.
    """
    for npz_path in sorted(glob.glob(os.path.join(root, "batch_*.npz"))):
        with np.load(npz_path, allow_pickle=True) as data:
            ids = [i.decode("utf-8") if isinstance(i, (bytes, bytearray)) else str(i)
                   for i in data["ID"]]
            if video_id in ids:
                return npz_path, ids.index(video_id)
    return None, None


# ---------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser("Overlay accident‐risk curve on a video")
    ap.add_argument("video",  help="input .mp4 path")
    ap.add_argument("output", help="output .mp4 path")
    ap.add_argument("--model", default="./model/demo_model",
                    help="checkpoint prefix OR directory (demo_model, final_model, …)")
    ap.add_argument("--gpu",   default="0", help="CUDA_VISIBLE_DEVICES id")
    ap.add_argument("--new_arch", action="store_true",
                    help="Use the upgraded geometry+velocity network "
                         "(checkpoint trained with new_accident.py)")
    return ap.parse_args()


# ---------------------------------------------------------------------
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # choose the correct graph
    if args.new_arch:
        from new_accident import build_model as _build
        NEW = True
    else:
        from accident import build_model as _build
        NEW = False

    outs = _build()                 # tensors list (old = 8, new = 9)

    soft_pred = outs[-2]            # ← second-to-last is always soft_pred

    # correct index for keep_ph
    keep_ph = outs[4] if NEW else outs[1]

    if NEW:                          # 9-tensor signature
        x_fc7, x_geom, x_vel = outs[0:3]
        y_ph                 = outs[3]
    else:                            # 8-tensor signature
        x_fc7                = outs[0]
        x_geom = x_vel       = None
        y_ph                 = outs[2]

    saver = tf.train.Saver()
    sess  = tf.Session(config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True)))

    ckpt = args.model
    if os.path.isdir(ckpt):
        ckpt = tf.train.latest_checkpoint(ckpt)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in dir {args.model}")
    saver.restore(sess, ckpt)
    print("✓ Restored weights from", ckpt)

    # ----------------------------------------------------------
    # 2. Load the feature batch containing this video
    # ----------------------------------------------------------
    vid_path  = args.video
    vid_id    = os.path.splitext(os.path.basename(vid_path))[0]

    npz_path, idx_in_batch = find_feature_file(vid_id)
    if npz_path is None:
        raise FileNotFoundError(f"Features for video ID {vid_id} not found in {FEATURE_DIR}")
    print("✓ Using feature file:", npz_path)

    with np.load(npz_path, allow_pickle=True) as npz:
        feat_batch = npz["data"]                       # (batch, 100, 20, 4096)

    # dummy labels just to satisfy the placeholder
    dummy_y = np.zeros((feat_batch.shape[0], N_CLASSES), np.float32)

    # ----------------------------------------------------------
    # 3. Run inference once to obtain the 100-step risk curve
    # ----------------------------------------------------------
    if NEW:
        with np.load(npz_path, allow_pickle=True) as npz:
            det_raw = npz["det"].astype("float32")

        # build geom & vel exactly like new_accident.load_npz_batch
        if det_raw.shape[2] == 19:
            B,T,_,_ = det_raw.shape
            det = np.concatenate([np.zeros((B,T,1,6), det_raw.dtype), det_raw], axis=2)
        else:
            det = det_raw
        x0,y0,x1,y1 = det[...,0], det[...,1], det[...,2], det[...,3]
        w = np.maximum(x1-x0,1);  h = np.maximum(y1-y0,1)
        geom_batch = np.stack([x0,y0,w,h], axis=-1)
        cx = 0.5*(x0+x1);  cy = 0.5*(y0+y1)
        d_cx = np.diff(cx, axis=1, prepend=cx[:,:1])
        d_cy = np.diff(cy, axis=1, prepend=cy[:,:1])
        vel_batch = np.stack([d_cx,d_cy], axis=-1)

    # ---------- build feed dict & run ----------
    feed = {x_fc7: feat_batch,
            keep_ph: [0.0],
            y_ph:    np.zeros((feat_batch.shape[0], N_CLASSES), np.float32)}

    if NEW:
        feed.update({x_geom: geom_batch,
                    x_vel:  vel_batch})

    risk_batch = sess.run(soft_pred, feed_dict=feed)   # (batch,100)
    risk_curve = risk_batch[idx_in_batch]  
    # ----------------------------------------------------------
    # 4. Prepare video I/O
    # ----------------------------------------------------------
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {vid_path}")

    fps  = cap.get(cv2.CAP_PROP_FPS)
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_vis_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # If the raw video is longer than 100 frames, just stretch the curve
    if total_vis_frames != len(risk_curve):
        xs_src = np.linspace(0, 1, len(risk_curve))
        xs_dst = np.linspace(0, 1, total_vis_frames)
        risk_curve = np.interp(xs_dst, xs_src, risk_curve)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(args.output, fourcc, fps, (w, h + GRAPH_H))
    print("✓ Writing overlay to", args.output)

    # ----------------------------------------------------------
    # 5. Frame-by-frame overlay
    # ----------------------------------------------------------
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(risk_curve):
            break

        # ---- draw risk panel --------------------------------
        panel = np.zeros((GRAPH_H, w, 3), np.uint8)

        # axes
        cv2.line(panel, (0, GRAPH_H-1), (w, GRAPH_H-1), (200,200,200), 1)
        cv2.line(panel, (0, 0), (0, GRAPH_H-1), (200,200,200), 1)

        # threshold = 0.5
        y_th = GRAPH_H-1 - int(0.5 * (GRAPH_H-1))
        cv2.line(panel, (0, y_th), (w, y_th), (100,100,100), 1)

        # current curve
        pts = [(
            int(i * w / len(risk_curve)),
            GRAPH_H-1 - int(r * (GRAPH_H-1))
        ) for i, r in enumerate(risk_curve[:frame_idx+1])]
        if len(pts) > 1:
            cv2.polylines(panel, [np.array(pts)], False, (0,0,255), 2)

        cv2.putText(panel, "Risk", (5,15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,255), 2)

        # ---- stack & write ----------------------------------
        out_frame = np.vstack((frame, panel))
        out.write(out_frame)

        frame_idx += 1

    cap.release()
    out.release()
    print("✓ Finished!")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()