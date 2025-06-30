#!/usr/bin/env python3
"""
Peak-frame strip + InstructBLIP caption with extra context.

Adds:
  • 5-frame horizontal strip  (t-2 … t+2)
  • Optional attention heat-map overlay
  • Structured text pre-amble  + few-shot primer
  • Checklist-style prompt
"""

import os, argparse, cv2, pandas as pd, numpy as np, torch, textwrap
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

# ────────────────────────── CLI ────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("Peak-frame strip + InstructBLIP caption")
    p.add_argument("video");         p.add_argument("track_stats");  p.add_argument("risk_logs")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--vlm_model", default="Salesforce/instructblip-flan-t5-xxl")
    p.add_argument("--full", action="store_true", help="use full frame instead of crop")
    p.add_argument("--enlarge", type=float, default=1.0)
    p.add_argument("--pad",     type=int,   default=0)
    p.add_argument("--overlay", action="store_true", help="draw attention heat-map")
    return p.parse_args()

# ─────────────────── helpers ───────────────────────────────────────
def expand_box(x0,y0,x1,y1,scale,W,H):
    cx,cy=(x0+x1)/2,(y0+y1)/2; bw, bh=(x1-x0)*scale,(y1-y0)*scale
    return max(0,int(cx-bw/2)), max(0,int(cy-bh/2)), \
           min(W,int(cx+bw/2)), min(H,int(cy+bh/2))

def overlay_heatmap(img, boxes, alphas, color=(0,0,255)):
    """boxes:(N,4), alphas:(N,) in [0,1]."""
    heat = np.zeros_like(img, np.uint8)
    for (x0,y0,x1,y1),a in zip(boxes, alphas):
        cv2.rectangle(heat,(x0,y0),(x1,y1), color, -1)
        heat[...,2] = (heat[...,2].astype(float)*0.0 + a*255).astype(np.uint8)
    return cv2.addWeighted(img,0.7, heat,0.3,0)

def make_strip(frames):
    h = max(f.shape[0] for f in frames)
    strip = np.hstack([cv2.copyMakeBorder(f,0,h-f.shape[0],0,0,cv2.BORDER_CONSTANT)
                       for f in frames])
    return strip

# ─────────────────── main ──────────────────────────────────────────
def main():
    args = parse_args(); os.makedirs(args.output_dir, exist_ok=True)

    # 0️⃣ read stats → peak frame
    peak = pd.read_csv(args.track_stats).loc[lambda d: d.peak_score.idxmax()]
    pf, ps = int(peak.peak_frame), float(peak.peak_score)

    cap = cv2.VideoCapture(args.video);  total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H,W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # 1️⃣ collect 5 frames (with boundary checks)
    idxs = [max(0,min(total-1,pf+i)) for i in (-2,-1,0,1,2)]
    frames=[]
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok,f = cap.read()
        if not ok: raise RuntimeError(f"cant read frame {idx}")
        frames.append(f)
    cap.release()

    # 2️⃣ choose crop or full on the *centre* frame
    centre = frames[2]
    if args.full:
        crop = centre
        att_boxes, att_alphas = None, None
    else:
        row = pd.read_csv(args.risk_logs).query("frame==@pf").iloc[0]
        chosen = next(i for i in (1,2,3) if abs(row[f'att{i}_score']-ps)<1e-6)
        x0,y0,x1,y1 = (int(row[f'att{chosen}_{c}']) for c in ('x0','y0','x1','y1'))
        x0,y0,x1,y1 = expand_box(x0,y0,x1,y1,args.enlarge,W,H)
        x0,y0,x1,y1 = max(0,x0-args.pad),max(0,y0-args.pad),min(W,x1+args.pad),min(H,y1+args.pad)
        crop = centre[y0:y1, x0:x1]
        att_boxes = [(x0,y0,x1,y1)]; att_alphas=[1.0]

    # 3️⃣ build strip & optional heat-map
    strip = make_strip([cv2.resize(f,crop.shape[1::-1]) for f in frames])
    if args.overlay and att_boxes is not None:
        strip = overlay_heatmap(strip, att_boxes, att_alphas)

    # 4️⃣ save PNG lossless
    img_path = os.path.join(args.output_dir,f"strip_{pf}.png")
    cv2.imwrite(img_path, strip,[cv2.IMWRITE_PNG_COMPRESSION,0]); print("saved",img_path)
    pil = Image.fromarray(strip[...,::-1])   # BGR→RGB

    # 5️⃣ build prompt
    preface = f"Peak risk score: {ps:.2f}\n"
    fewshot = ("Q: Two cars nearly collide at an intersection.\n"
               "A: 1. Objects: red car left, blue car right …\n"
               "   2. Trajectories cross at centre …\n"
               "   3. Rain makes road slippery …\n"
               "   4. Braking early would avoid impact.\n\n")

    prompt  = fewshot + preface + textwrap.dedent("""\
        Q: Analyse this scene. Answer with numbered points:
        1. Key objects and where they are
        2. How each object is moving
        3. Why/where a collision could occur
        4. Contextual factors (speed, road, weather)
        5. How to avoid or mitigate the accident
        A:""")

    # 6️⃣ VLM
    # 5️⃣  Load InstructBLIP ------------------------------------------------
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dev.type == "cuda":
        # let Accelerate figure out a shard layout; DON'T call .to(dev) afterwards
        model = InstructBlipForConditionalGeneration.from_pretrained(
            args.vlm_model,
            torch_dtype=torch.float16,
            device_map="auto"          # ← creates meta / off-disk shards
        )
    else:                             # pure-CPU fallback
        model = InstructBlipForConditionalGeneration.from_pretrained(
            args.vlm_model,
            torch_dtype=torch.float32
        ).to(dev)                      # <─ only move when we didn’t use device_map

    proc  = InstructBlipProcessor.from_pretrained(args.vlm_model, use_fast=True)
    batch = proc(images=pil, text=prompt, return_tensors="pt").to(dev)
    ids   = model.generate(**batch, max_new_tokens=256,num_beams=6,
                           no_repeat_ngram_size=3,length_penalty=1.0)
    answer= proc.decode(ids[0], skip_special_tokens=True).strip()

    # strip any echoed prompt
    if answer.startswith(prompt.split("A:")[0]):
        answer = answer.split("A:")[-1].strip()

    cap_path = os.path.join(args.output_dir,f"caption_{pf}.txt")
    with open(cap_path,'w',encoding='utf-8') as f: f.write(answer+"\n")
    print("\n───────── CAPTION ─────────\n"+answer+"\nSaved →",cap_path)

if __name__=="__main__":
    main()