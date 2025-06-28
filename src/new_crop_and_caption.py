#!/usr/bin/env python3
"""
Peak-frame crop + InstructBLIP deep caption with improved input quality.

Example
-------
python src/new_crop_and_caption.py \
    data/raw/videos/testing/positive/000469.mp4 \
    outputs/track_stats_000469.csv \
    outputs/risk_logs_000469.csv \
    --enlarge 1.5 --pad 20 \
    --prompt "Let’s think step by step: identify objects, trajectories, collision point, contextual factors, and mitigation strategies." \
    --full                # feed entire frame instead of cropping
"""
import os
import argparse
import cv2
import pandas as pd
import torch
import numpy as np
from PIL import Image
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration
)

def parse_args():
    p = argparse.ArgumentParser("Peak-frame crop + InstructBLIP deep caption")
    p.add_argument("video", help="Path to input video")
    p.add_argument("track_stats", help="CSV with per-track stats")
    p.add_argument("risk_logs", help="CSV with per-frame attention & bboxes")
    p.add_argument("--output_dir", default="outputs",
                   help="Where to save the frames and captions")
    p.add_argument("--vlm_model",
                   default="Salesforce/instructblip-flan-t5-xxl",
                   help="InstructBLIP checkpoint (flan-t5-xxl, opt-2.7b, …)")
    p.add_argument("--prompt",
        default=(
            "Let’s think step by step:\n"
            "1) Identify all objects and their directions.\n"
            "2) Describe any trajectory intersections.\n"
            "3) Explain contextual factors (e.g. speed, environment).\n"
            "4) Summarize potential outcomes and avoidance measures."
        ),
        help="Deep reasoning prompt for the VLM"
    )
    p.add_argument("--full", action="store_true",
                   help="Use the full frame instead of cropping")
    p.add_argument("--enlarge", type=float, default=1.0,
                   help="Scale bbox about its center (1.0=no scale, >1 adds context)")
    p.add_argument("--pad", type=int, default=0,
                   help="Extra pixel padding around the scaled box")
    return p.parse_args()

def expand_box(x0, y0, x1, y1, scale, W, H):
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = (x1 - x0) * scale, (y1 - y0) * scale
    nx0 = max(0, int(cx - bw/2))
    ny0 = max(0, int(cy - bh/2))
    nx1 = min(W-1, int(cx + bw/2))
    ny1 = min(H-1, int(cy + bh/2))
    return nx0, ny0, nx1, ny1

def sharpen_image(img_np: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img_np, -1, kernel)

def upscale_crop(crop: np.ndarray, factor: float = 2.0) -> np.ndarray:
    h, w = crop.shape[:2]
    return cv2.resize(crop, (int(w*factor), int(h*factor)),
                      interpolation=cv2.INTER_CUBIC)

def main():
    args = parse_args()
    peak_dir = os.path.join(args.output_dir, "peak_frames")
    cap_dir  = os.path.join(args.output_dir, "captions")
    os.makedirs(peak_dir, exist_ok=True)
    os.makedirs(cap_dir, exist_ok=True)

    # 1) pick the peak frame
    stats = pd.read_csv(args.track_stats)
    best  = stats.loc[stats.peak_score.idxmax()]
    pf, ps = int(best.peak_frame), float(best.peak_score)
    print(f"[+] Track {best.track_id} peaks at frame {pf} (score={ps:.3f})")

    # 2) load that frame
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, pf)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {pf} from {args.video}")
    H, W = frame.shape[:2]

    # 3) crop or full
    if args.full:
        crop = frame
        print(f"[+] Using full frame ({W}×{H})")
    else:
        log = pd.read_csv(args.risk_logs).query("frame==@pf").iloc[0]
        chosen = next(i for i in (1,2,3)
                      if abs(log[f"att{i}_score"]-ps)<1e-6)
        x0,y0,x1,y1 = (int(log[f"att{chosen}_{c}"]) for c in ("x0","y0","x1","y1"))
        x0s,y0s,x1s,y1s = expand_box(x0,y0,x1,y1,args.enlarge,W,H)
        x0p = max(0, x0s-args.pad)
        y0p = max(0, y0s-args.pad)
        x1p = min(W, x1s+args.pad)
        y1p = min(H, y1s+args.pad)
        crop = frame[y0p:y1p, x0p:x1p]
        print(f"[+] Cropped: scale×{args.enlarge}, pad={args.pad}px → {crop.shape[1]}×{crop.shape[0]}")

    # 4) upscale & sharpen
    crop_up = upscale_crop(crop, factor=2.0)
    crop_up = sharpen_image(crop_up)

    # save as PNG
    png_path = os.path.join(peak_dir, f"frame_{pf}.png")
    cv2.imwrite(png_path, crop_up, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"[+] Saved PNG → {png_path}")

    # resize to the model’s patch grid (384×384)
    pil = Image.open(png_path).convert("RGB")
    pil = pil.resize((384, 384), Image.BICUBIC)

    # 5) load and run InstructBLIP on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Loading VLM {args.vlm_model} on {device}")
    proc  = InstructBlipProcessor.from_pretrained(
                args.vlm_model,
                trust_remote_code=True,
                use_fast=True
            )
    model = InstructBlipForConditionalGeneration.from_pretrained(
                args.vlm_model,
                torch_dtype=(torch.float16 if device.type=="cuda" else torch.float32),
                device_map=("auto" if device.type=="cuda" else None),
                trust_remote_code=True
            )
    if device.type=="cpu":
        model.to(device)

    batch = proc(images=pil, text=args.prompt, return_tensors="pt").to(model.device)
    out_ids = model.generate(
        **batch,
        max_new_tokens=256,
        num_beams=8,
        no_repeat_ngram_size=4,
        length_penalty=1.0,
        early_stopping=True,
        do_sample=False,
    )

    full   = proc.decode(out_ids[0], skip_special_tokens=True).strip()
    prompt = args.prompt.strip()
    caption = full[len(prompt):].strip() if full.startswith(prompt) else full

    # 6) write out
    cap_path = os.path.join(cap_dir, f"caption_{pf}.txt")
    with open(cap_path, "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    print(f"[✓] Caption written → {cap_path}\n{caption}")

if __name__ == '__main__':
    main()