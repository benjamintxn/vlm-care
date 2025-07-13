#!/usr/bin/env python3
# vlm_basic_progress.py  – 8 frames → Qwen-VL 7B → plain-text file

from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch, glob, textwrap, os, time
from pathlib import Path
os.environ["TRANSFORMERS_NO_TF"] = "1"

ID   = "Qwen/Qwen2-VL-7B-Instruct"
VID  = "000572"                     # ← change to any clip ID

# ─── 0.  model & processor ──────────────────────────────────────
tic = time.time(); print("[0] Loading processor + model … ", end="", flush=True)
proc  = AutoProcessor.from_pretrained(ID)
model = AutoModelForVision2Seq.from_pretrained(
            ID,
            torch_dtype=torch.float16,
            device_map="auto",        # add load_in_4bit=True if RAM is tight
        )
print(f"done ({time.time()-tic:.1f}s)")

# ─── 1.  load images (8) ────────────────────────────────────────
tic = time.time()
frames = sorted(glob.glob(f"outputs/risk_frames/{VID}/*.jpg"))
stages = ["start","post_start","pre_onset","onset",
          "leadup","peak","reaction","aftermath"]
images = [Image.open(p).convert("RGB") for p in frames]
print(f"[1] Loaded {len(images)} images … ({time.time()-tic:.1f}s)")

# ─── 2.  build chat prompt ──────────────────────────────────────
msgs = [{"role":"system","content":"You detect road accidents."}]
user = []
for img, st in zip(images, stages):
    user += [{"image": img}, {"text": f"Stage: {st}"}]

user.append({"text": textwrap.dedent("""\
  TASK –
  For EACH stage reply on one line:
    Frame <stage>: <yes/no/unsure>, <≤12-word reason>.
  If you answer “yes” for ANY stage, add ONE extra line:
    Description: <≤30 words describing vehicles, motion, impact>.
  Then Overall: <yes/no>, earliest stage.
  Respond with exactly the required lines, nothing else.
  """)})
msgs.append({"role":"user","content": user})
print("[2] Prompt assembled.")

# ─── 3.  tokenise ───────────────────────────────────────────────
tic = time.time()
prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
toks   = proc(images=images, text=prompt, return_tensors="pt").to(model.device)
print(f"[3] Prompt & images tokenised … ({time.time()-tic:.1f}s)")

# ─── 4.  generate ───────────────────────────────────────────────
tic = time.time(); print("[4] Generating text … ", end="", flush=True)
with torch.inference_mode():
    out = model.generate(**toks, max_new_tokens=180)
print(f"done ({time.time()-tic:.1f}s)")

reply = proc.decode(out[0], skip_special_tokens=True).strip()
print("\n─────────  VLM OUTPUT  ─────────\n")
print(reply)
print("\n────────────────────────────────\n")

# ─── 5.  save to file ───────────────────────────────────────────
out_dir = Path(f"outputs/vlm_captions/{VID}")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / f"{VID}.txt").write_text(reply + "\n", encoding="utf-8")
print(f"[5] ✓ saved → {out_dir/(VID + '.txt')}")