# scripts/download_coco_person.py

import os
import wget
import zipfile
from pycocotools.coco import COCO

# 1) Paths
BASE = os.getcwd()
DATA_DIR = os.path.join(BASE, "data", "raw", "coco2017")
IMGS_DIR = os.path.join(DATA_DIR, "images")
ANN_DIR  = os.path.join(DATA_DIR, "annotations")
ZIP_PATH = os.path.join(ANN_DIR, "annotations_trainval2017.zip")
JSON_NAME = "instances_val2017.json"
JSON_PATH = os.path.join(ANN_DIR, JSON_NAME)

os.makedirs(IMGS_DIR, exist_ok=True)
os.makedirs(ANN_DIR, exist_ok=True)

# 2) Download & extract annotations if needed
if not os.path.exists(JSON_PATH):
    # official ZIP URL
    zip_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    print("Downloading COCO annotations zip (~250 MB)...")
    wget.download(zip_url, ZIP_PATH)
    print("\nExtracting JSON...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        # extract only the val instances JSON
        member = f"annotations/{JSON_NAME}"
        z.extract(member, ANN_DIR)
        # move it up one level
        os.replace(os.path.join(ANN_DIR, member),
                   JSON_PATH)
    print("Cleaning up zip...")
    os.remove(ZIP_PATH)
    print("Annotations ready.")

# 3) Load COCO and pick images
coco = COCO(JSON_PATH)
person_cat = coco.getCatIds(catNms=["person"])
img_ids = coco.getImgIds(catIds=person_cat)[:200]  # first 200

# 4) Download each image
for img_id in img_ids:
    info = coco.loadImgs(img_id)[0]
    url = info["coco_url"]
    target = os.path.join(IMGS_DIR, info["file_name"])
    if not os.path.exists(target):
        print(f"Downloading {info['file_name']}...")
        wget.download(url, target)
        print()
print("All done—~200 person images downloaded.")
