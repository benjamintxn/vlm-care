import argparse
import pandas as pd
import numpy as np


def iou(boxA, boxB):
    # box = [x0, y0, x1, y1]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0


class Track:
    def __init__(self, tid, box, score, frame):
        self.id = tid
        self.box = box
        self.scores = [score]
        self.frames = [frame]
        self.missed = 0

    def update(self, box, score, frame):
        self.box = box
        self.scores.append(score)
        self.frames.append(frame)
        self.missed = 0

    def mark_missed(self):
        self.missed += 1


def track_attention(input_csv, output_csv, iou_thresh=0.3, max_missed=1):
    df = pd.read_csv(input_csv)

    tracks = []
    next_id = 0

    for frame in sorted(df['frame'].unique()):
        sub = df[df['frame'] == frame]
        # collect detections for this frame
        dets = []
        for k in [1, 2, 3]:
            score_col = f'att{k}_score'
            x0_col = f'att{k}_x0'; y0_col = f'att{k}_y0'
            x1_col = f'att{k}_x1'; y1_col = f'att{k}_y1'
            if score_col in sub and not sub[score_col].isna().all():
                score = sub.iloc[0][score_col]
                box = [
                    sub.iloc[0][x0_col], sub.iloc[0][y0_col],
                    sub.iloc[0][x1_col], sub.iloc[0][y1_col]
                ]
                dets.append((box, score))

        # assignment
        assigned = set()
        for tr in tracks:
            best_iou = 0.0
            best_idx = None
            for idx, (box, score) in enumerate(dets):
                if idx in assigned:
                    continue
                cur_iou = iou(tr.box, box)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_idx = idx
            if best_iou >= iou_thresh and best_idx is not None:
                box, score = dets[best_idx]
                tr.update(box, score, frame)
                assigned.add(best_idx)
            else:
                tr.mark_missed()

        # create new tracks
        for idx, (box, score) in enumerate(dets):
            if idx not in assigned:
                tracks.append(Track(next_id, box, score, frame))
                next_id += 1

        # remove dead tracks
        tracks = [tr for tr in tracks if tr.missed <= max_missed]

    # prepare stats
    stats = []
    for tr in tracks:
        length = len(tr.frames)
        start = min(tr.frames)
        end = max(tr.frames)
        avg_score = np.mean(tr.scores)
        peak_idx = int(np.argmax(tr.scores))
        peak_score = tr.scores[peak_idx]
        peak_frame = tr.frames[peak_idx]
        stats.append({
            'track_id': tr.id,
            'length': length,
            'start_frame': start,
            'end_frame': end,
            'avg_score': avg_score,
            'peak_score': peak_score,
            'peak_frame': peak_frame
        })

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_csv, index=False)
    print(f"Wrote track stats to {output_csv}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Track-level attention clustering")
    p.add_argument('input_csv', help='Per-frame attention CSV')
    p.add_argument('output_csv', help='Output track stats CSV')
    p.add_argument('--iou_thresh', type=float, default=0.3)
    p.add_argument('--max_missed', type=int, default=1)
    args = p.parse_args()
    track_attention(args.input_csv, args.output_csv,
                    args.iou_thresh, args.max_missed)