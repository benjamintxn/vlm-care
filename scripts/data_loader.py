# scripts/data_loader.py

import os
from pycocotools.coco import COCO
from PIL import Image

class COCOPersonDataset:
    def __init__(self, data_dir, max_images=None):
        self.img_dir = os.path.join(data_dir, "images")
        ann_file   = os.path.join(data_dir, "annotations", "instances_val2017.json")
        self.coco = COCO(ann_file)
        cat_ids = self.coco.getCatIds(catNms=["person"])
        self.img_ids = self.coco.getImgIds(catIds=cat_ids)
        if max_images:
            self.img_ids = self.img_ids[:max_images]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_info = self.coco.loadImgs(self.img_ids[idx])[0]
        path = os.path.join(self.img_dir, img_info["file_name"])
        img  = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_info["id"], catIds=self.coco.getCatIds(["person"]))
        anns    = self.coco.loadAnns(ann_ids)

        # Extract bboxes as [x1,y1,x2,y2]
        bboxes = []
        for a in anns:
            x, y, w, h = a["bbox"]
            bboxes.append([x, y, x + w, y + h])

        return img_info["file_name"], img, bboxes

# Quick test
if __name__ == "__main__":
    ds = COCOPersonDataset(data_dir="data/raw/coco2017", max_images=50)
    print(f"Dataset size: {len(ds)} images")
    # Fetch a sample
    fname, img, bboxes = ds[0]
    print("First image:", fname)
    print("Number of person bboxes:", len(bboxes))