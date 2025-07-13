#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2  # type: ignore
import numpy as np
import pandas as pd

try:
    from cv2 import dnn_superres  # OpenCV ≥ 4.5
except ImportError:
    dnn_superres = None  # SR unavailable; we’ll warn at runtime

BBox = Tuple[int, int, int, int]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _boxes_to_pixels(box: List[float], w: int, h: int) -> BBox:
    x0, y0, x1, y1 = box
    if max(x0, x1) <= 1.0 and max(y0, y1) <= 1.0:
        x0, x1 = x0 * w, x1 * w
        y0, y1 = y0 * h, y1 * h
    return int(x0), int(y0), int(x1), int(y1)


def _find_onset(risk: pd.Series, smooth: int = 10, tau: float = 0.6) -> int:
    sm = risk.rolling(smooth, min_periods=1).mean()
    thr = tau * sm.max()
    cand = np.flatnonzero(sm.to_numpy() > thr)
    return int(cand[0]) if cand.size else int(risk.idxmax())


def _fixed_square_crop(frame: np.ndarray, bbox: BBox, crop_dim: int) -> np.ndarray:
    h, w, _ = frame.shape
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = crop_dim // 2

    l, t = cx - half, cy - half
    r, b = l + crop_dim, t + crop_dim

    pad_l, pad_t = max(0, -l), max(0, -t)
    pad_r, pad_b = max(0, r - w), max(0, b - h)
    l, t, r, b = max(0, l), max(0, t), min(w, r), min(h, b)

    crop = frame[t:b, l:r]
    if any((pad_l, pad_t, pad_r, pad_b)):
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT)
    return crop

# -----------------------------------------------------------------------------
# Super‑resolution wrapper
# -----------------------------------------------------------------------------

def _init_superres(model_path: str, scale: int):
    if dnn_superres is None:
        raise RuntimeError("OpenCV built without dnn_superres; reinstall opencv‑contrib-python")
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    # Model name inferred from filename: "edsr", "lapsrn", etc.
    algo = Path(model_path).stem.split("_")[0].lower()
    sr.setModel(algo, scale)
    return sr

# -----------------------------------------------------------------------------
# Core routine
# -----------------------------------------------------------------------------

def extract_and_crop_keyframes(
    csv_path: str,
    video_path: str,
    output_dir: str,
    *,
    padding: int = 50,
    crop_dim: int = 700,
    superres_model: str | None = None,
    sr_scale: int = 4,
) -> List[Tuple[str, int, str]]:

    # ~~~ load SR model once ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    sr = None
    if superres_model:
        sr = _init_superres(superres_model, sr_scale)
        print(f"🔍 Super‑resolution enabled ({Path(superres_model).name}, ×{sr_scale})")

    # ~~~ parse CSV ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    df = pd.read_csv(csv_path)
    req = {"risk"} | {f"att{i}_{c}" for i in (1, 2) for c in ("x0", "y0", "x1", "y1")}
    if (missing := req - set(df.columns)):
        raise KeyError(f"CSV missing columns: {sorted(missing)}")

    # ~~~ open video ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ~~~ key‑frame selection ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    f_peak = int(df["risk"].idxmax())
    f_on   = _find_onset(df["risk"])
    key_idx: Dict[str, int] = {
        "1_start": f_on - 30,
        "2_post_start": f_on - 20,
        "3_pre_onset": max(0, f_on - 10),
        "4_onset": f_on,
        "5_leadup": int(f_on + (f_peak - f_on) * 0.5),
        "6_peak": f_peak,
        "7_reaction": min(n_frames - 1, f_peak + 3),
        "8_aftermath": min(n_frames - 1, f_peak + 5),
    }
    seen, final_idx = set(), {}
    for lbl in sorted(key_idx, key=key_idx.get):
        idx = key_idx[lbl]
        while idx in seen and idx < n_frames - 1:
            idx += 1
        final_idx[lbl] = idx; seen.add(idx)

    # ~~~ crop loop ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved: List[Tuple[str, int, str]] = []

    for lbl, idx in sorted(final_idx.items(), key=lambda x: x[1]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"⚠️  frame {idx} unreadable – skipped"); continue

        # ---------- ❶ save full-frame variant -----------------------------------
        full_out = Path(output_dir) / f"{lbl}_{idx:04d}_full.jpg"
        cv2.imwrite(str(full_out), frame)
        saved.append((f"{lbl}_full", idx, str(full_out)))

        # ---------- ❷ compute crop as before ------------------------------------
        row = df.iloc[idx]
        b1 = _boxes_to_pixels(row[[f"att1_{c}" for c in ("x0","y0","x1","y1")]].tolist(), W, H)
        b2 = _boxes_to_pixels(row[[f"att2_{c}" for c in ("x0","y0","x1","y1")]].tolist(), W, H)
        x0 = max(0, min(b1[0], b2[0]) - padding); y0 = max(0, min(b1[1], b2[1]) - padding)
        x1 = min(W, max(b1[2], b2[2]) + padding); y1 = min(H, max(b1[3], b2[3]) + padding)

        crop = _fixed_square_crop(frame, (x0, y0, x1, y1), crop_dim)

        if sr is not None:                       # optional super-resolution
            up   = sr.upsample(crop)
            crop = cv2.resize(up, (crop_dim, crop_dim), interpolation=cv2.INTER_LANCZOS4)

        crop_out = Path(output_dir) / f"{lbl}_{idx:04d}_crop.jpg"
        cv2.imwrite(str(crop_out), crop)
        saved.append((f"{lbl}_crop", idx, str(crop_out)))
        print(f"  ✔️  {crop_out.name}  +  {full_out.name}")
    return saved

# -----------------------------------------------------------------------------
# CLI helper (auto‑paths)
# -----------------------------------------------------------------------------

def _autopaths(vid: str, video: str | None, csv: str | None, out: str | None):
    vid = vid.strip().removesuffix(".mp4")
    # video
    if video is None:
        for sub in ("positive", "negative"):
            p = Path("./data/raw/videos/testing") / sub / f"{vid}.mp4"
            if p.exists():
                video = str(p); break
        else:
            raise FileNotFoundError("video not found for ID " + vid)
    # csv
    if csv is None:
        csv = f"./outputs/risk_logs/{vid}.csv"
        if not Path(csv).exists():
            raise FileNotFoundError(csv)
    # out dir
    if out is None:
        out = f"./outputs/risk_frames/{vid}"
    return video, csv, out

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="7 key‑frame crops with optional super‑resolution")
    ap.add_argument("video_id", help="Video ID – e.g. 000467")
    ap.add_argument("--video", help="Explicit video path (.mp4)")
    ap.add_argument("--csv", help="Explicit risk-log CSV path")
    ap.add_argument("--out", help="Output directory for crops")
    ap.add_argument("--padding", type=int, default=110, help="Padding around union bbox (px)")
    ap.add_argument("--crop-dim", type=int, default=700, help="Final square crop size (px)")
    ap.add_argument("--superres", help="Path to .pb super-resolution model (optional)")
    ap.add_argument("--sr-scale", type=int, default=8, help="Model scale factor (2, 4, 8…)")
    args = ap.parse_args()

    video, csv_log, out_dir = _autopaths(args.video_id, args.video, args.csv, args.out)

    extract_and_crop_keyframes(
        csv_log,
        video,
        out_dir,
        padding=args.padding,
        crop_dim=args.crop_dim,
        superres_model=args.superres,
        sr_scale=args.sr_scale,
    )