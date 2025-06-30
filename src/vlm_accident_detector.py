#!/usr/bin/env python3
# vlm_accident_detector.py  ────────────────────────────────────────────
# Decide “accident / no-accident” with InstructBLIP
#
# 1.  Smooth the risk curve from risk_logger CSV
# 2.  Find the first sustained rise above --t_low  (or fall back to peak)
# 3.  Pull six **separate** frames:  t-6,-3,-1,0,+2,+5   (0 = onset)
# 4.  For each frame
#        ▸ crop the top-attention box (+40 % padding, 2× upsample)
#        ▸ ask InstructBLIP a binary question on   • the crop
#                                                 • the full frame
#        ▸ take “yes” if either answer is “yes”
# 5.  Verdict = YES  if ≥ 2 of the 6 frames say “yes” (majority vote)
# ---------------------------------------------------------------------
import os, re, json, argparse, time
import cv2, numpy as np, pandas as pd, torch
from PIL import Image
from transformers import (InstructBlipProcessor,
                          InstructBlipForConditionalGeneration)

# ──────────────── command-line ───────────────────────────────────────
def cli():
    p = argparse.ArgumentParser("Accident yes/no with InstructBLIP")
    p.add_argument("video"); p.add_argument("track_stats"); p.add_argument("risk_logs")
    p.add_argument("--out_dir", default="outputs/vlm_decision/")
    p.add_argument("--vlm",     default="Salesforce/instructblip-flan-t5-xxl")
    p.add_argument("--t_low",   type=float, default=0.50,
                   help="risk threshold for onset detection")
    p.add_argument("--win",     type=int,   default=6,
                   help="# rising frames needed to declare an onset")
    return p.parse_args()

# COCO ids → short names  (enough for traffic scenes)
COCO = {1:"person",2:"bicycle",3:"car",4:"motorcycle",6:"bus",8:"truck"}

# ──────────────── small helpers ──────────────────────────────────────
def median_smooth(arr, k=5):
    pad = k//2
    ext = np.pad(arr, (pad, pad), mode="edge")
    return np.array([np.median(ext[i:i+k]) for i in range(len(arr))])

def derivative(arr):
    return np.diff(arr, prepend=arr[0])

def first_onset(risk, grad, t_low, win):
    rising = (risk > t_low) & (grad > 0)
    hits   = np.where(np.convolve(rising, np.ones(win, int), "valid") == win)[0]
    return int(hits[0]) if hits.size else int(np.argmax(risk))

def build_prompt(risk, obj_names):
    objs = ", ".join(obj_names) if obj_names else "unknown objects"
    return (f"Answer with a single word (yes or no): "
            f"Is there a traffic collision in this frame? "
            f"Objects: {objs}. Risk={risk:.2f}")

CRASH_RE = re.compile(r"\b(crash|collision|impact|accident|hit|smash|ram)\b", re.I)

# ───────────────── main ──────────────────────────────────────────────
def main():
    args = cli()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- 1) risk curve ----------------------------------------
    curve_df = pd.read_csv(args.risk_logs).set_index("frame")
    risk_raw = curve_df["risk"].values
    risk     = median_smooth(risk_raw, k=5)
    grad     = derivative(risk)
    onset    = first_onset(risk, grad, args.t_low, args.win)

    offsets = np.array([-6, -3, -1, 0, +2, +5])
    frames  = np.clip(onset + offsets, 0, len(risk)-1)

    # ---------- 2) video & VLM ---------------------------------------
    cap = cv2.VideoCapture(args.video)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));  W = int(cap.get(3))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc   = InstructBlipProcessor.from_pretrained(args.vlm, use_fast=True)
    model  = InstructBlipForConditionalGeneration.from_pretrained(
                args.vlm,
                device_map="auto" if device.type == "cuda" else None,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)

    yes_votes = 0
    out_json  = []

    # ---------- 3) loop over the six frames --------------------------
    for t in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, frame = cap.read();  assert ok
        row  = curve_df.loc[int(t)]
        risk_t = float(risk[int(t)])

        # crop top-attention box with 40 % padding
        x0,y0,x1,y1 = (int(row.att1_x0), int(row.att1_y0),
                       int(row.att1_x1), int(row.att1_y1))
        pad = int(0.40 * (y1 - y0))
        x0=max(0,x0-pad); y0=max(0,y0-pad); x1=min(W,x1+pad); y1=min(H,y1+pad)
        crop = cv2.resize(frame[y0:y1, x0:x1], None, fx=2, fy=2,
                          interpolation=cv2.INTER_CUBIC)

        # object names (optional column written by risk_logger)
        try:
            ids   = [int(i) for i in str(row.obj_classes).split(",")]
            names = [COCO.get(i, "obj") for i in ids]
        except Exception:
            names = []

        prompt = build_prompt(risk_t, names)

        def ask(img):
            inputs = proc(images=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                          text=prompt, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=3,
                                 num_beams=1, do_sample=False)
            return proc.decode(out[0], skip_special_tokens=True).strip().lower()

        ans_crop = ask(crop)
        ans_full = ask(frame)
        final_ans = "yes" if ("yes" in ans_crop or "yes" in ans_full) else "no"
        yes_votes += (final_ans == "yes")

        out_json.append({"frame": int(t),
                         "risk":  risk_t,
                         "prompt": prompt,
                         "answer_crop": ans_crop,
                         "answer_full": ans_full,
                         "final": final_ans})
        print(f"[{t:3d}] {final_ans}")

    cap.release()

    verdict = ("YES – accident detected" if yes_votes >= 2
               else "NO – no accident")
    print("="*60 + f"\nVLM verdict:  {verdict}\n" + "="*60)

    with open(os.path.join(args.out_dir, "vlm_result.json"), "w") as f:
        json.dump({"verdict": verdict,
                   "yes_votes": int(yes_votes),
                   "samples": out_json}, f, indent=2)
    print("✓ result JSON written to", args.out_dir)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()