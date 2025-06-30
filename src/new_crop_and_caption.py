#!/usr/bin/env python3
"""
multi_frame_caption.py
──────────────────────
1. Find peak-risk frame  t  from <track_stats>.csv
2. Grab six frames  [t-5, t-3, t(crop), t(full), t+3, t+5]
3. Caption each with InstructBLIP using role-specific prompts
4. If peak-risk < THRESH or captions say "no accident" → stop.
5. Else send captions to an LLM to get final driver advice.

USAGE
─────
python multi_frame_caption.py \
       <video>.mp4  <track_stats>.csv  <risk_logs>.csv \
       --peak_dir outputs/peak_frames \
       --caption_dir outputs/captions \
       --risk_thresh 0.4 \
       --openai_key $OPENAI_API_KEY
"""

import os, argparse, re, json, time
import numpy as np, pandas as pd, cv2, torch, openai
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

# ──────────────────────────────── CLI ────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("Multi-frame caption + advice")
    p.add_argument("video"); p.add_argument("track_stats"); p.add_argument("risk_logs")
    p.add_argument("--peak_dir",    default="outputs/peak_frames")
    p.add_argument("--caption_dir", default="outputs/captions")
    p.add_argument("--vlm_model",   default="Salesforce/instructblip-flan-t5-xxl")
    p.add_argument("--risk_thresh", type=float, default=0.4,
                   help="Skip LLM if peak risk below this")
    p.add_argument("--openai_key",  default=None,
                   help="If set, use ChatGPT for the advice step")
    p.add_argument("--gpt_model",   default="gpt-3.5-turbo")
    return p.parse_args()

# ---------------- helper ------------------------------------------------
OFFSETS = [-5, -3, 0, 0, +3, +5]        # frames relative to peak
ROLE    = ["pre", "pre", "peak-crop", "peak-full", "post", "post"]

def build_prompt(role, risk):
    risk_pct = f"{risk:.2f}"
    if role == "pre":
        return (f"The risk in this frame is {risk_pct}. "
                "Can you anticipate an accident occurring here? "
                "If so, describe in detail how it might happen.")
    if role.startswith("peak"):
        return (f"The peak risk in this frame is {risk_pct}. "
                "Is a collision occurring?  Describe it in detail.")
    return (f"The risk in this frame is {risk_pct}. "
            "Has an accident already happened?  If so, describe it in detail.")

def is_negative_caption(txt):
    """very loose heuristic → True iff caption *explicitly* denies accident"""
    neg = re.search(r"\b(no accident|no collision|nothing .*? happen)", txt, re.I)
    return bool(neg)

# ───────────────────────────── main ────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.peak_dir,    exist_ok=True)
    os.makedirs(args.caption_dir, exist_ok=True)

    # 1) peak frame & score
    stats = pd.read_csv(args.track_stats)
    peak  = stats.loc[stats.peak_score.idxmax()]
    t     = int(peak.peak_frame);   peak_score = float(peak.peak_score)

    # full risk curve for quick lookup
    curve = pd.read_csv(args.risk_logs).set_index("frame")["risk"]

    # 2) open video once
    cap = cv2.VideoCapture(args.video);   H = int(cap.get(4));  W = int(cap.get(3))

    # 3) prepare VLM
    dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = InstructBlipProcessor.from_pretrained(args.vlm_model, use_fast=True)
    model= InstructBlipForConditionalGeneration.from_pretrained(
               args.vlm_model,
               device_map="auto" if dev.type=="cuda" else None,
               torch_dtype=torch.float16 if dev.type=="cuda" else torch.float32)

    captions = []
    accident_flag = False

    # 4) iterate six frames
    for off, role in zip(OFFSETS, ROLE):
        frm = max(0, min(int(cap.get(7))-1, t+off))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frm)
        ok, frame = cap.read();   assert ok, f"Cannot read frame {frm}"
        risk = float(curve.get(frm, peak_score))         # fallback

        # crop only for the “peak-crop” slot
        if role=="peak-crop":
            log = pd.read_csv(args.risk_logs).query("frame==@t").iloc[0]
            x0,y0,x1,y1 = (int(log[f"att1_{k}"]) for k in ("x0","y0","x1","y1"))
            pad = 10;  x0=max(0,x0-pad); y0=max(0,y0-pad); x1=min(W,x1+pad); y1=min(H,y1+pad)
            frame = frame[y0:y1, x0:x1]

        fn = os.path.join(args.peak_dir, f"{role}_{frm}.jpg")
        cv2.imwrite(fn, frame)

        # caption
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prompt = build_prompt(role.replace("peak-",""), risk)
        inputs = proc(images=img, text=prompt, return_tensors="pt").to(model.device)
        out    = model.generate(**inputs, max_new_tokens=120)
        cap_txt= proc.decode(out[0], skip_special_tokens=True).strip()
        captions.append((role, cap_txt, risk))
        if not is_negative_caption(cap_txt):
            accident_flag = True
        print(f"[{role}] {cap_txt}")

    cap.release()

    # 5) If nothing interesting → short file & exit
    if (peak_score < args.risk_thresh) or (not accident_flag):
        out_path = os.path.join(args.caption_dir, "no_accident.txt")
        with open(out_path,"w") as f:
            f.write(f"No accident detected (peak risk {peak_score:.2f}).\n")
        print("✓ No collision – skipping LLM summary.")
        return

    # 6) build context for the LLM
    prompt_chunks = [f"{i+1}. {r} frame: {txt}"
                     for i,(r,txt,_) in enumerate(captions)]
    gpt_prompt = (
      "You are an advanced driving-assistant analyst.\n"
      "Below are six numbered observations from a vision-language model "
      "watching a dash-cam accident clip (frames before, during, after).\n\n"
      + "\n".join(prompt_chunks) +
      "\n\nTask:\n"
      "• Briefly summarise what collision happened (vehicles, direction, point of impact).\n"
      "• Explain the main contributing factors (speed, weather, blind spot, etc.).\n"
      "• Give **three concrete pieces of advice** the driver could have followed "
      "to avoid or mitigate the crash.\n"
      "Write the answer in plain English, 2-3 short paragraphs."
    )

    if args.openai_key:
        openai.api_key = args.openai_key
        rsp = openai.ChatCompletion.create(
            model=args.gpt_model,
            messages=[{"role":"user","content": gpt_prompt}],
            max_tokens=300, temperature=0.7)
        advice = rsp["choices"][0]["message"]["content"].strip()
    else:
        advice = "(LLM step skipped – set --openai_key to enable)"

    # 7) save everything
    time_tag = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.caption_dir, f"captions_{time_tag}.json"),"w") as f:
        json.dump({"peak_score":peak_score, "captions":captions,
                   "driver_advice":advice}, f, indent=2)
    print("✓ Written captions + advice JSON")

if __name__=="__main__":
    main()