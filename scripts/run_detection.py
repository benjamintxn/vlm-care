# scripts/run_detection.py

import os
import cv2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

INPUT_DIR  = "data/raw/coco2017/images"
OUTPUT_DIR = "data/processed/detections"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_model(conf_threshold=0.5):
    cfg = get_cfg()
    # Use a Mask R-CNN pre-trained on COCO
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    return DefaultPredictor(cfg)

def run_detection(predictor):
    for fname in sorted(os.listdir(INPUT_DIR))[:50]:  # limit to first 50
        img_path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(img_path)
        outputs = predictor(img)

        # Visualise and save
        v = Visualizer(
            img[:, :, ::-1],
            scale=1.0,
            instance_mode=ColorMode.IMAGE  # shows masks
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_img = v.get_image()[:, :, ::-1]
        cv2.imwrite(os.path.join(OUTPUT_DIR, fname), out_img)
        print(f"Processed {fname}")

if __name__ == "__main__":
    print("Setting up Mask R-CNN...")
    predictor = setup_model(0.5)
    run_detection(predictor)
    print("Done. Check images in", OUTPUT_DIR)
