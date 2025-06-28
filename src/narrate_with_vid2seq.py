#!/usr/bin/env python3
"""
Generate a short narration for the accident clip using Vid2Seq (or any Seq2Seq LM).
Input  : video.mp4 + risk_logs.csv + track_stats.csv
Output : narration.txt
"""

import os
import argparse
import cv2
import pandas as pd
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a short narration for the accident clip using a Seq2Seq model"
    )
    p.add_argument("video", help="Path to input .mp4 file")
    p.add_argument("track_stats", help="CSV with per-track stats (including peak_frame)")
    p.add_argument("risk_logs", help="CSV with per-frame risk scores and bboxes")
    p.add_argument(
        "--out_dir", default="output", help="Directory to save narration.txt"
    )
    p.add_argument(
        "--model", default="google/vit-b-prompt-video-to-text",
        help="Seq2Seq model ID (e.g. google/vit-b-prompt-video-to-text)"
    )
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES id")
    return p.parse_args()


def extract_window(video_path, center_frame, length=32):
    """Extract `length` frames centered on `center_frame` as RGB arrays."""
    cap = cv2.VideoCapture(video_path)
    start = max(0, center_frame - length // 2)
    frames = []
    for idx in range(start, start + length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load track stats and find peak risk frame
    stats = pd.read_csv(args.track_stats)
    if 'peak_frame' not in stats.columns:
        raise ValueError("track_stats missing 'peak_frame' column")
    peak_frame = int(stats.loc[stats.peak_score.idxmax()].peak_frame)
    print(f"[+] Peak frame: {peak_frame}")

    # 2) Extract frames around the peak
    clip = extract_window(args.video, peak_frame, length=32)
    if not clip:
        raise RuntimeError(f"Could not extract frames around frame {peak_frame}")
    print(f"[+] Extracted {len(clip)} frames")

    # 3) Load risk value at peak frame
    risk_df = pd.read_csv(args.risk_logs)
    risk_row = risk_df.loc[risk_df.frame == peak_frame]
    if risk_row.empty:
        raise ValueError(f"No risk entry for frame {peak_frame}")
    risk_val = float(risk_row.risk.values[0]) * 100
    print(f"[+] Risk at peak: {risk_val:.1f}%")

    # 4) Initialize model and processor
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Attempt remote, fallback to local-only
    try:
        processor = AutoProcessor.from_pretrained(args.model)
    except OSError:
        if os.path.isdir(args.model):
            processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        else:
            raise

    model     = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16
    ).to(device)

    # 5) Build prompt
    prompt = (
        f"Dash-cam video segment centered at frame {peak_frame}. "
        f"Estimated risk {risk_val:.0f} percent. "
        "Describe events that could lead to an accident."
    )
    print(f"[+] Prompt: {prompt}")

    # 6) Prepare inputs and generate narration
    inputs = processor(
        videos=clip,
        text=prompt,
        return_tensors="pt"
    ).to(device)
    outputs = model.generate(
        **inputs,
        max_length=64,
        num_beams=4,
        early_stopping=True
    )
    narration = processor.decode(outputs[0], skip_special_tokens=True)
    print(f"[+] Narration: {narration}")

    # 7) Save narration
    out_path = os.path.join(args.out_dir, "narration.txt")
    with open(out_path, 'w') as f:
        f.write(prompt + "\n" + narration + "\n")
    print(f"[+] Saved narration to {out_path}")

if __name__ == "__main__":
    main()
