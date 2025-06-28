# extract_features.py  (v2)
import os, glob, argparse, time
import numpy as np
import torch, torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import roi_align
from torchvision.models import resnet50
import cv2
from tqdm import tqdm

# ──────────────────────────── CLI ────────────────────────────
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--video_root",  required=True)
    p.add_argument("--out_root",    required=True)
    p.add_argument("--pos_list",    required=True)
    p.add_argument("--detector_weights", required=True,
                   help="local fasterrcnn_resnet50_fpn_coco-*.pth")
    p.add_argument("--n_frames", type=int, default=100)
    p.add_argument("--n_det",    type=int, default=20)
    p.add_argument("--chunk_size", type=int, default=6,
                   help="frames per CUDA batch (avoid OOM)")
    p.add_argument("--ids",
                   help="comma-separated clip IDs (e.g. 000001,000002) "
                        "for a quick debug run")
    p.add_argument("--no_flow", action="store_true",
                   help="extract only appearance (2048-D) features")
    return p

# ─────────────────────── helpers ─────────────────────────────
def load_frames(path:str, n:int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, total-1, n).astype(int)
    frames=[]
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def to_tensor(img, size=(300,300), device="cpu") -> torch.Tensor:
    t = torch.from_numpy(img).permute(2,0,1).float()/255.0
    t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=size, mode="bilinear",
            align_corners=False).squeeze(0).to(device)
    return t

# ───────────────── helper: parse --ids ───────────────────────
def parse_ids(arg: str) -> set[str]:
    """
    Accepts:
        "000001,000005"               → {"000001","000005"}
        "1-5,10,20-22"                → {"000001","000002",…,"000005","000010",
                                         "000020","000021","000022"}
    Returns a set of zero-padded 6-char IDs.
    """
    ids = set()
    for part in arg.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:                             # range a-b
            a, b = part.split('-')
            for i in range(int(a), int(b) + 1):
                ids.add(f"{i:06d}")
        else:                                       # single ID
            ids.add(part.zfill(6))
    return ids

# ───────────────────── main extractor ───────────────────────
@torch.no_grad()
def extract_one(vid_path:str,
                detector, backbone, flow_backbone,
                device, nF:int, nD:int, chunk:int,
                use_flow:bool):
    frames = load_frames(vid_path, nF)
    tensors = [to_tensor(f, device=device) for f in frames]

    # run detector/backbone in small chunks
    dets, feat_maps = [], []
    for i in range(0, len(tensors), chunk):
        batch = torch.stack(tensors[i:i+chunk])
        outs  = detector(batch)            # list[dict]
        feats = backbone(batch)            # OrderedDict of FPN maps

        dets += outs
        # take P5 (last) map  ➜ stride = 300 / H
        fmap = list(feats.values())[-1]    # (B,C,H,W)
        feat_maps += [fmap[j] for j in range(fmap.shape[0])]
        torch.cuda.empty_cache()

    # optical-flow magnitude images (optional)
    if use_flow:
        flows=[]
        for a,b in zip(frames[:-1], frames[1:]):
            f1 = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            f2 = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
            fl = cv2.calcOpticalFlowFarneback(f1,f2,None,0.5,3,15,3,5,1.2,0)
            mag,_ = cv2.cartToPolar(fl[...,0], fl[...,1])
            mag=cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb=np.stack([mag]*3,-1).astype(np.uint8)
            flows.append(to_tensor(rgb, device=device))
        flows.append(flows[-1])   # pad last frame

    concat_feats, all_boxes = [], []
    for f_map, det, flow_img in zip(feat_maps, dets,
                                    flows if use_flow else [None]*len(dets)):
        boxes  = det["boxes"]
        scores = det["scores"]
        keep   = scores.argsort(descending=True)[:nD]
        boxes  = boxes[keep]                      # (k,4)
        k      = boxes.shape[0]

        # scale boxes to feature-map coords
        H = f_map.shape[-1]          # P5 is square; input 300 → stride=300/H
        scale = 300.0 / H
        boxes_fm = boxes / scale

        # appearance ROI-Align
        app = roi_align(f_map.unsqueeze(0), [boxes_fm],
                        output_size=(1,1),
                        spatial_scale=1.0, aligned=True
                       ).squeeze(-1).squeeze(-1)      # (k,2048)

        # flow ROI-Align (optional)
        if use_flow:
            fmap_flow = flow_backbone(flow_img.unsqueeze(0))
            fmap_flow = list(fmap_flow.values())[-1] \
                        if isinstance(fmap_flow, dict) else fmap_flow
            flow = roi_align(fmap_flow, [boxes_fm],
                             output_size=(1,1),
                             spatial_scale=1.0, aligned=True
                            ).squeeze(-1).squeeze(-1)  # (k,2048)
            feats = torch.cat([app, flow], 1)          # (k,4096)
        else:
            feats = app                                # (k,2048)

        # pad to n_det
        if k < nD:
            padF = torch.zeros(nD-k, feats.shape[1], device=feats.device)
            feats = torch.cat([feats, padF], 0)
            padB = torch.zeros(nD-k, 4, device=boxes.device)
            boxes = torch.cat([boxes, padB], 0)

        concat_feats.append(feats.cpu().numpy())
        all_boxes.append(boxes.cpu().numpy())

    return np.stack(concat_feats), np.stack(all_boxes)

# ─────────────────────────── main ───────────────────────────
def main():
    args = make_parser().parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    sel_ids = parse_ids(args.ids) if args.ids else None
    pos_ids = set(open(args.pos_list).read().split())
    dev     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # detector + backbone
    det = fasterrcnn_resnet50_fpn(pretrained=False).to(dev)
    det.load_state_dict(torch.load(args.detector_weights, map_location=dev))
    det.eval()
    backbone = det.backbone.body

    # flow backbone (ResNet-50 conv5) if requested
    flow_backbone = None
    if not args.no_flow:
        r50 = resnet50(weights="IMAGENET1K_V1")
        flow_backbone = torch.nn.Sequential(*list(r50.children())[:-2]).eval().to(dev)

    splits = ["training/positive","training/negative",
              "testing/positive","testing/negative"]

    for sp in splits:
        vidlist = glob.glob(os.path.join(args.video_root, sp, "*.mp4"))
        outdir  = os.path.join(args.out_root, sp)
        os.makedirs(outdir, exist_ok=True)

        for vp in tqdm(vidlist, desc=sp):
            vid_id = os.path.splitext(os.path.basename(vp))[0]
            if sel_ids and vid_id not in sel_ids:
                continue
            start=time.time()
            feat, boxes = extract_one(
                vp, det, backbone, flow_backbone,
                dev, args.n_frames, args.n_det,
                args.chunk_size, not args.no_flow)
            fps=len(feat)/(time.time()-start)

            lbl = 1 if vid_id in pos_ids else 0
            np.savez(os.path.join(outdir, f"{vid_id}.npz"),
                     data=feat.astype(np.float32),
                     det=boxes.astype(np.float32),
                     labels=np.array([[1-lbl,lbl]], np.float32),
                     ID=np.array([vid_id], dtype="S6"))
            tqdm.write(f"{vid_id}: saved  feat={feat.shape}  fps≈{fps:.1f}")

if __name__ == "__main__":
    main()