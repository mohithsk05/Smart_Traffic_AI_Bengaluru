"""
Smart Traffic AI - Bengaluru
Vehicle Detection service (YOLOv8)

Serves:
  POST /api/vision/detect     -> upload one image, get back vehicle
                                  detections (counts + bounding boxes)
  GET  /health                -> liveness + which weights are loaded

--------------------------------------------------------------------------
What's real right now vs. what to upgrade
--------------------------------------------------------------------------
This service runs genuine YOLOv8 inference (ultralytics) - it is NOT
simulated. Out of the box it loads the standard COCO-pretrained yolov8n.pt
weights, which already detect car / motorcycle / bus / truck / bicycle
correctly on real images.

The gap: COCO has no "auto-rickshaw" class, which matters a lot for
Bengaluru traffic. To close that gap and match the vehicle-mix your city
actually has, fine-tune on the India Driving Dataset (IDD) - see
VISION_TRAINING.md in this folder for the full dataset download + training
pipeline. Once trained, just point MODEL_PATH below at your
`runs/detect/train/weights/best.pt` and everything else in this file is
unchanged.
"""

import io
from typing import Dict, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------
# TODO(after IDD fine-tuning): change this to your trained weights path,
# e.g. "runs/detect/idd_vehicles/weights/best.pt"
MODEL_PATH = "yolov8n.pt"

# COCO class ids we care about for traffic counting. Once fine-tuned on
# IDD, replace this map with your custom class list (which will include
# "autorickshaw" and can split "motorcycle"/"bicycle" as needed).
COCO_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
CONFIDENCE_THRESHOLD = 0.35

app = FastAPI(title="Smart Traffic AI - Bengaluru: Vehicle Detection Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"Loading YOLO weights: {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)
print("Model loaded. Classes:", model.names)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "classes_tracked": list(COCO_VEHICLE_CLASSES.values())}


@app.post("/api/vision/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    results = model.predict(image, verbose=False, conf=CONFIDENCE_THRESHOLD)
    r = results[0]

    counts: Dict[str, int] = {name: 0 for name in COCO_VEHICLE_CLASSES.values()}
    boxes: List[dict] = []

    for box in r.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in COCO_VEHICLE_CLASSES:
            continue
        label = COCO_VEHICLE_CLASSES[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        counts[label] += 1
        boxes.append({
            "label": label,
            "confidence": round(conf, 3),
            "box": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
        })

    total_vehicles = counts["car"] + counts["motorcycle"] + counts["bus"] + counts["truck"] + counts["bicycle"]

    return {
        "image_size": {"width": image.width, "height": image.height},
        "total_vehicles": total_vehicles,
        "counts": counts,
        "detections": boxes,
        "model": MODEL_PATH,
        "note": "COCO-pretrained model; fine-tune on IDD to add auto-rickshaw class and improve Indian-traffic accuracy.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
