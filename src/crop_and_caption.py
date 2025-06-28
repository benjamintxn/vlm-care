#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop the top-attended object at its peak frame and caption it with a VLM"
    )
    parser.add_argument(
        "video", help="Path to input video (e.g. ./dataset/videos/testing/positive/000469.mp4)"
    )
    parser.add_argument(
        "track_stats", help="CSV with per-track stats (e.g. output/track_stats_000469.csv)"
    )
    parser.add_argument(
        "risk_logs", help="CSV with per-frame attention & bboxes (e.g. output/risk_logs/000469.csv)"
    )
    parser.add_argument(
        "--output_dir", default="output", help="Where to save the crop and caption.txt"
    )
    parser.add_argument(
        "--vlm_model",
        default="Salesforce/blip-image-captioning-base",
        help="HuggingFace VLM checkpoint"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) pick best track from track_stats
    stats = pd.read_csv(args.track_stats)
    best = stats.loc[stats.peak_score.idxmax()]
    peak_frame = int(best.peak_frame)
    peak_score = float(best.peak_score)
    print(f"[+] Best track {best.track_id} peaks at frame {peak_frame} (score={peak_score:.3f})")

    # 2) load per-frame attention log, find the row for peak_frame
    df = pd.read_csv(args.risk_logs)
    row = df[df["frame"] == peak_frame]
    if len(row)==0:
        raise ValueError(f"No entry for frame {peak_frame} in {args.risk_logs}")
    row = row.iloc[0]

    # 3) find which of the top-3 attention boxes matches our peak_score (fallback to att1)
    chosen = 1
    for i in (1,2,3):
        if abs(row[f"att{i}_score"] - peak_score) < 1e-6:
            chosen = i
            break
    print(f"[+] Using bbox att{chosen}_x0..y1")

    x0 = int(row[f"att{chosen}_x0"])
    y0 = int(row[f"att{chosen}_y0"])
    x1 = int(row[f"att{chosen}_x1"])
    y1 = int(row[f"att{chosen}_y1"])

    # 4) grab that frame from the video and crop
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, peak_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {peak_frame} from {args.video}")

    h, w, _ = frame.shape
    pad = 10
    x0p, y0p = max(0, x0 - pad), max(0, y0 - pad)
    x1p, y1p = min(w, x1 + pad), min(h, y1 + pad)
    crop = frame[y0p:y1p, x0p:x1p]
    crop_path = os.path.join(args.output_dir, f"peak_crop_{peak_frame}.jpg")
    cv2.imwrite(crop_path, crop)
    print(f"[+] Saved crop to {crop_path}")

    # 5) load the BLIP VLM and caption the crop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Loading VLM model {args.vlm_model} on {device}")
    processor = BlipProcessor.from_pretrained(args.vlm_model)
    model     = BlipForConditionalGeneration.from_pretrained(args.vlm_model).to(device)

    img = Image.open(crop_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    out    = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"[+] VLM caption: {caption}")

    # 6) save caption
    caption_path = os.path.join(args.output_dir, "caption.txt")
    with open(caption_path, "w") as f:
        f.write(caption + "\n")
    print(f"[+] Caption written to {caption_path}")

if __name__ == "__main__":
    main()