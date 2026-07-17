# Fine-tuning YOLOv8 on the India Driving Dataset (IDD)

The vision service (`vision_service.py`) already runs real detection using
COCO-pretrained weights — it correctly finds car/bus/truck/motorcycle/
bicycle today. This guide is for the upgrade: fine-tuning on **IDD**, which
adds the classes COCO is missing for Indian roads (autorickshaw above all)
and adapts the model to Indian traffic density, vehicle mix, and camera
angles — this is also the more defensible dataset to cite in a research
paper, since it's an academic benchmark built specifically from Indian
roads including Bengaluru.

## 1. Get access
1. Go to **https://idd.insaan.iiit.ac.in**
2. Register for an account (free, academic use) and agree to the license.
3. Download the **"IDD Detection"** subset (not IDD-Segmentation — that's
   for pixel-level segmentation, not bounding boxes). It ships as images
   + Pascal-VOC-style XML annotations.
4. Extract it — you'll get a structure roughly like:
   ```
   IDD_Detection/
   ├── JPEGImages/
   │   └── <folder>/<image>.jpg
   └── Annotations/
       └── <folder>/<image>.xml
   ```

## 2. Convert annotations to YOLO format
Ultralytics YOLO expects one `.txt` file per image with lines of
`class_id x_center y_center width height` (all normalized 0–1), plus a
`data.yaml` describing the classes. Run this converter:

```python
# convert_idd_to_yolo.py
import os
import xml.etree.ElementTree as ET
from pathlib import Path

# IDD's detection classes relevant to vehicle counting (skip pedestrian/
# rider/traffic-sign classes not needed for this feature)
CLASS_MAP = {
    "car": 0,
    "motorcycle": 1,
    "bus": 2,
    "truck": 3,
    "autorickshaw": 4,
    "bicycle": 5,
    "vehicle fallback": 6,   # IDD's catch-all for ambiguous/occluded vehicles
}

def convert_one(xml_path: Path, img_w: int, img_h: int, out_txt: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[name]
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        xc = ((xmin + xmax) / 2) / img_w
        yc = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines))

def get_image_size(xml_path: Path):
    tree = ET.parse(xml_path)
    size = tree.getroot().find("size")
    return int(size.find("width").text), int(size.find("height").text)

def main(idd_root: str, out_root: str):
    idd_root, out_root = Path(idd_root), Path(out_root)
    ann_dir = idd_root / "Annotations"
    for xml_path in ann_dir.rglob("*.xml"):
        rel = xml_path.relative_to(ann_dir).with_suffix("")
        w, h = get_image_size(xml_path)
        out_txt = out_root / "labels" / f"{rel}.txt".replace("/", "_")
        convert_one(xml_path, w, h, out_txt)
    print("Conversion done ->", out_root / "labels")

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
```

Run it:
```bash
python convert_idd_to_yolo.py /path/to/IDD_Detection ./idd_yolo
```

Then organize images to match (copy/symlink `JPEGImages/**/*.jpg` into
`idd_yolo/images/`, using the same flattened naming as the labels), and
split into train/val (an 85/15 split is standard):

```bash
python - <<'EOF'
import random, shutil
from pathlib import Path

root = Path("idd_yolo")
images = list((root / "images").glob("*.jpg"))
random.seed(42)
random.shuffle(images)
split = int(len(images) * 0.85)
for subset, files in [("train", images[:split]), ("val", images[split:])]:
    for f in files:
        dest_img = root / subset / "images" / f.name
        dest_lbl = root / subset / "labels" / f.with_suffix(".txt").name
        dest_img.parent.mkdir(parents=True, exist_ok=True)
        dest_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dest_img)
        shutil.copy(root / "labels" / f.with_suffix(".txt").name, dest_lbl)
print("Split complete")
EOF
```

## 3. Write the data.yaml
```yaml
# idd_yolo/data.yaml
path: ./idd_yolo
train: train/images
val: val/images

names:
  0: car
  1: motorcycle
  2: bus
  3: truck
  4: autorickshaw
  5: bicycle
  6: vehicle_fallback
```

## 4. Train
```bash
pip install ultralytics
```
```python
# train_idd.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # start from COCO-pretrained weights (transfer learning)
model.train(
    data="idd_yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    project="runs/detect",
    name="idd_vehicles",
)
```
Run with a GPU (Colab T4 is enough for yolov8n on this scale — expect
roughly 2–4 hours for 50 epochs depending on dataset size). On CPU only,
budget considerably longer; consider `yolov8n` (not `s`/`m`) and fewer
epochs if you're CPU-bound.

Track your mAP the same way you did for Smart Traffic AI's earlier YOLOv8
work — `runs/detect/idd_vehicles/results.csv` gives per-epoch mAP50,
mAP50-95, precision, recall, ready to drop into your paper's results table.

## 5. Deploy the fine-tuned weights
Copy `runs/detect/idd_vehicles/weights/best.pt` into `backend/`, then in
`vision_service.py` change:
```python
MODEL_PATH = "runs/detect/idd_vehicles/weights/best.pt"
```
and update `COCO_VEHICLE_CLASSES` to your new IDD class map (from step 3),
including `autorickshaw`. Restart the service — everything else (the
`/api/vision/detect` endpoint, the response schema, the dashboard wiring)
stays exactly the same.
