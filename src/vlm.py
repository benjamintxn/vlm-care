from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch, glob, textwrap, time, os
from pathlib import Path
os.environ["TRANSFORMERS_NO_TF"] = "1"

VID = "000583"                      
ID  = "Qwen/Qwen2-VL-7B-Instruct"         

# ─── 0  load model ──────────────────────────────────────────────
t0 = time.time(); print("[0] loading model … ", end="", flush=True)
proc  = AutoProcessor.from_pretrained(ID)
model = AutoModelForVision2Seq.from_pretrained(ID, device_map="auto", torch_dtype=torch.float16) # load_in_4bit=True, bnb_4bit_quant_type="nf4, "
print(f"done ({time.time()-t0:.1f}s)")

# ─── 1  load 8 frames ───────────────────────────────────────────
frames = sorted(glob.glob(f"outputs/risk_frames/{VID}/*.jpg"))
stages = ["start","post_start","pre_onset","onset","leadup","peak","reaction","aftermath"]
images = [Image.open(p).convert("RGB") for p in frames]
print(f"[1] {len(images)} images loaded.")

# ─── 2  build chat messages ────────────────────────────────────
system_msg = (
    "You are an accident-detection assistant"
    "You are tasked to determine if there is an accident about to occur, occuring, or has occured."
    "Rewarded for concise, non-repetitive, stage-specific safety insights."
)

task = textwrap.dedent("""\
    You must output **exactly 11 lines**.

    ✱ For lines 1-8 use **different wording** each time; do **not** repeat an
    identical sentence or phrase from a previous line.
                
    Example  
    Stage start:   no, normal traffic flow.  
    ...  
    Stage peak:    yes, rider hits car rear-panel.  

    Now follow the same style; do not repeat identical words across lines.

    1  Stage start:        <yes/no/unsure>, <≤15-word reason>.
    2  Stage post_start:   <yes/no/unsure>, <≤15-word reason>.
    3  Stage pre_onset:    <yes/no/unsure>, <≤15-word reason>.
    4  Stage onset:        <yes/no/unsure>, <≤15-word reason>.
    5  Stage leadup:       <yes/no/unsure>, <≤15-word reason>.
    6  Stage peak:         <yes/no/unsure>, <≤15-word reason>.
    7  Stage reaction:     <yes/no/unsure>, <≤15-word reason>.
    8  Stage aftermath:    <yes/no/unsure>, <≤15-word reason>.

    •  Focus on what **changes** from the previous frame (speed, lane
    position, braking, skid-marks, rider posture, etc.).

    ✱ Line 9  Narrative: **40-50 words** summarising the sequence; avoid
    repeating any sentence used above.

    ✱ Line 10 Overall: <yes/no>, earliest stage marked “yes”.

    ✱ Line 11 must be completely blank.

    Do **not** output anything before line 1 or after line 11.
""")

user_parts = []
for img, st in zip(images, stages):
    user_parts += [
        {"image": img},
        {"text": f"Stage: {st}\n"}          # ← line-break after every stage
    ]
user_parts.append({"text": task})

messages = [
    {"role": "system", "content": system_msg},
    {"role": "user",   "content": user_parts},
]

# ─── 3  tokenize ────────────────────────────────────────────────
prompt = proc.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

tok = proc(images=images, text=prompt,
           return_tensors="pt").to(model.device)
print("[2] prompt tokenised.")

# ─── 4  generate ────────────────────────────────────────────────
GEN_TOKENS = 230          # plenty for 11 lines
print("[3] generating … ", end="", flush=True)
with torch.inference_mode():
    out = model.generate(
        **tok,
        max_new_tokens=GEN_TOKENS,
        do_sample=False,
        temperature=0.3,
        repetition_penalty=1.05,
        eos_token_id=proc.tokenizer.eos_token_id,  # stop on <|endoftext|>
    )
print(f"done ({time.time()-t0:.1f}s)")

raw = proc.decode(out[0], skip_special_tokens=True)

# ── isolate the assistant’s part ───────────────────────────────
# the template always contains the word “assistant” on its own line
try:
    assistant_only = raw.split("\nassistant\n", 1)[1].strip()
except IndexError:                         # model returned nothing new
    assistant_only = "(no assistant output!)"

# keep exactly the first 11 non-empty lines
reply_lines = [l for l in assistant_only.splitlines() if l.strip()]
reply       = "\n".join(reply_lines[:11])

# ─── 5  save ────────────────────────────────────────────────────
out_dir = Path(f"outputs/vlm_captions/{VID}")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / f"{VID}.txt").write_text(reply + "\n", encoding="utf-8")
print(f"[4] ✓ saved → {out_dir/(VID + '.txt')}")