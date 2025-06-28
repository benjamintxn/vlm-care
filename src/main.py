#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys

# ── USER CONFIG ────────────────────────────────────────────────────────────────
MODEL_DIR    = "./data/external/demo_model/demo_model"
VLM_MODEL    = "./data/external/hf_models/blip_base"   # e.g. Salesforce/blip2-flan-t5-xl
GPU_ID       = "0"
OUTPUT_ROOT  = "./outputs"
# ────────────────────────────────────────────────────────────────────────────────

def run(cmd):
    print(f"\n> {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if res.returncode != 0:
        sys.exit(res.returncode)

def main():
    p = argparse.ArgumentParser(
        description="Full pipeline: risk → tracks → crop & caption"
    )
    p.add_argument("video", help="Path to your video (e.g. ./.../000476.mp4)")
    args = p.parse_args()

    vid       = os.path.splitext(os.path.basename(args.video))[0]
    rl_csv    = os.path.join(OUTPUT_ROOT, "risk_logs",   f"{vid}.csv")
    ts_csv    = os.path.join(OUTPUT_ROOT, "track_stats", f"{vid}.csv")
    cap_dir   = os.path.join(OUTPUT_ROOT, "captions",    vid)

    # make sure output dirs exist
    for sub in ("risk_logs", "track_stats"):
        os.makedirs(os.path.join(OUTPUT_ROOT, sub), exist_ok=True)
    os.makedirs(cap_dir, exist_ok=True)

    # 1) per-frame risk + attention logging
    run([
        "python", "src/risk_logger.py",
        args.video, rl_csv,
        "--model", MODEL_DIR,
        "--gpu", GPU_ID
    ])

    # 2) track-level attention clustering
    run([
        "python", "src/track_attention.py",
        rl_csv, ts_csv
    ])

    # 3) crop-and-caption best-attended object with new script
    run([
        "python", "src/new_crop_and_caption.py",
        args.video,                   # the MP4
        ts_csv,                       # track-stats CSV
        rl_csv,                       # risk-log CSV
        "--output_dir", cap_dir,
        "--vlm_model", VLM_MODEL,
        "--prompt", "Is there an accident in this scene? If so, what is happening?"
    ])

    print(f"\n✅ Done! Captions are in {cap_dir}")

if __name__ == "__main__":
    main()