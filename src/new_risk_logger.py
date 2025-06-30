import argparse, os, glob, csv
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from new_accident import build_model, n_classes

# Constants (must match training)
n_frames = 100
n_detection = 20

def parse_annotations(video_id, annotation_dir="./data/raw/annotations"):
    """Parse annotation file for accident-involved objects"""
    ann_path = os.path.join(annotation_dir, f"{video_id}.txt")
    frame_objs = {}
    
    if not os.path.exists(ann_path):
        return frame_objs
    
    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
                
            # Extract frame number from filename (e.g., "frame_00045" -> 44)
            frame_name = parts[0]
            try:
                frame_num = int(frame_name.split('_')[1]) - 1
            except:
                continue
                
            # Check if object is involved in accident
            if int(parts[7]) == 1:
                bbox = list(map(float, parts[3:7]))
                track_id = int(parts[1])
                if frame_num not in frame_objs:
                    frame_objs[frame_num] = []
                frame_objs[frame_num].append({
                    'track_id': track_id,
                    'bbox': bbox
                })
                
    return frame_objs

def calculate_ttc(prev_boxes, current_boxes, fps=30):
    """Approximate Time-to-Collision using bounding box dynamics"""
    if prev_boxes is None or current_boxes is None:
        return []
    
    ttc_list = [99.9] * current_boxes.shape[0]
    for obj_idx in range(current_boxes.shape[0]):
        if np.all(current_boxes[obj_idx] == 0) or (prev_boxes[obj_idx] is None) or np.all(prev_boxes[obj_idx] == 0):
            continue
            
        # Calculate relative movement (pixels/frame)
        dx = current_boxes[obj_idx, 0] - prev_boxes[obj_idx, 0]
        dy = current_boxes[obj_idx, 1] - prev_boxes[obj_idx, 1]
        velocity = np.sqrt(dx**2 + dy**2)
        
        # Use bounding box height as distance proxy
        height = current_boxes[obj_idx, 3] - current_boxes[obj_idx, 1]
        if height > 10 and velocity > 0.5:  # Ignore small boxes and minimal movement
            ttc_list[obj_idx] = height / velocity / fps
            
    return ttc_list

def parse_args():
    p = argparse.ArgumentParser(
        description="Enhanced per-frame risk logging with object details"
    )
    p.add_argument('video',   help='Path to input mp4')
    p.add_argument('out_csv', help='Output CSV path')
    p.add_argument('--model',  default='./model/demo_model',
                   help='Checkpoint dir or prefix')
    p.add_argument('--gpu',    default='0', help='CUDA_VISIBLE_DEVICES id')
    p.add_argument('--annotation_dir', default='./data/annotations',
                   help='Directory containing annotation files')
    return p.parse_args()

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 1) Build TF graph - get attention weights
    x_ph, keep_ph, y_ph, _, _, _, soft_pred, all_alphas = build_model()
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
    print("✓ Restored", ckpt)

    # 3) Get video ID and load annotations
    vid_name = os.path.splitext(os.path.basename(args.video))[0]
    accident_objects = parse_annotations(vid_name, args.annotation_dir)
    print(f"✓ Loaded annotations for {len(accident_objects)} accident frames")

    # 4) Find feature batch containing this video
    feature_file = None
    for fn in sorted(glob.glob('./data/features/testing/batch_*.npz')):
        data = np.load(fn, allow_pickle=True)
        ids = data['ID']
        # decode bytes → str
        ids = [i.decode() if isinstance(i, (bytes, bytearray)) else str(i) for i in ids]
        if vid_name in ids:
            feature_file = fn
            break
            
    if feature_file is None:
        raise FileNotFoundError(f"No feature batch for {vid_name}")
    print("✓ Using features from", feature_file)

    # 5) Load feature data
    data = np.load(feature_file, allow_pickle=True)
    feats = data['data']   # (batch_size, n_frames, n_detection, n_input)
    dets  = data['det']    # (batch_size, n_frames, n_detection, 4)
    ids   = data['ID']
    ids   = [i.decode() if isinstance(i, (bytes, bytearray)) else str(i) for i in ids]
    idx   = ids.index(vid_name)

    # 6) Prepare data for inference
    bs = feats.shape[0]
    dummy_y    = np.zeros((bs, n_classes), dtype=np.float32)
    dummy_keep = np.zeros((bs,), dtype=np.float32)

    # 7) Run inference
    risk_batch, alpha_batch = sess.run(
        [soft_pred, all_alphas],
        feed_dict={
            x_ph:        feats,
            y_ph:        dummy_y,
            keep_ph:     dummy_keep
        }
    )
    # Extract results for this video
    risk_curve = risk_batch[idx]              # (n_frames,)
    attentions = alpha_batch[:, :, idx]       # (n_frames, n_detection-1)
    bboxes     = dets[idx]                    # (n_frames, n_detection, 4)
    n_det = bboxes.shape[1]

    # 8) Prepare TTC calculation
    prev_boxes = None
    ttc_history = []

    # 9) Write enhanced CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        # Enhanced header
        header = [
            'frame', 'global_risk', 'min_ttc', 'max_attention',
            'has_accident', 'accident_objects'
        ]
        for obj_idx in range(n_det):     # use n_det, not n_detection
            header += [
                f'obj_{obj_idx}_attention',
                f'obj_{obj_idx}_ttc',
                f'obj_{obj_idx}_x0', f'obj_{obj_idx}_y0',
                f'obj_{obj_idx}_x1', f'obj_{obj_idx}_y1',
                f'obj_{obj_idx}_in_accident'
            ]
        w.writerow(header)

        # Process each frame
        for fr in range(n_frames):
            # Calculate TTC for this frame
            current_boxes = bboxes[fr]
            ttcs = calculate_ttc(prev_boxes, current_boxes)
            prev_boxes = current_boxes  # Store for next frame
            
            # Check for accident in this frame
            has_accident = 1 if fr in accident_objects else 0
            accident_obj_ids = []
            if has_accident:
                accident_obj_ids = [obj['track_id'] for obj in accident_objects[fr]]
            
            # Prepare per-object data
            obj_details = []
            for obj_idx in range(n_det):
                # attention array has no dummy slot → shift index by −1
                if obj_idx == 0:
                    attention = 0.0                      # the image-level feature
                else:
                    att_idx   = obj_idx - 1
                    attention = attentions[fr, att_idx] if att_idx < attentions.shape[1] else 0.0

                ttc  = ttcs[obj_idx] if obj_idx < len(ttcs) else 99.9
                bbox = current_boxes[obj_idx]
                in_accident = 1 if obj_idx in accident_obj_ids else 0

                obj_details.extend([
                    float(attention), float(ttc),
                    int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                    in_accident
                ])
            
            # Find max attention and min TTC
            max_attention = np.max(attentions[fr]) if fr < len(attentions) else 0
            min_ttc = min(ttcs) if ttcs else 99.9
            
            # Write frame row
            row = [
                fr,
                float(risk_curve[fr]),
                float(min_ttc),
                float(max_attention),
                has_accident,
                ",".join(map(str, accident_obj_ids)) if accident_obj_ids else ""
            ]
            row += obj_details
            w.writerow(row)

    print("✓ Wrote enhanced log to", args.out_csv)

if __name__=='__main__':
    main()