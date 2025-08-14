# Propane Vision: Tank Detection Pipeline

End-to-end image recognition to find propane tanks for customer prospecting using YOLOv8.

## Project Layout
- `datasets/propane_tanks/` — YOLO dataset (images + YOLO labels)
- `config/dataset.yaml` — YOLO dataset config
- `scripts/train.py` — Train a YOLOv8 detector
- `scripts/infer.py` — Batch inference + JSON output
- `app/api.py` — FastAPI inference service
- `Dockerfile` — Containerized API

## Dataset Structure (YOLO format)
```
datasets/propane_tanks/
  images/
    train/  # .jpg/.png
    val/
  labels/
    train/  # .txt (YOLO: cls cx cy w h)
    val/
```
Class list: `propane_tank`

## 1) Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Add Data + Labels
- Use Label Studio or Roboflow to annotate propane tanks
- Export in YOLOv8 format and place files per structure above
- Update `config/dataset.yaml` if you change paths

## 3) Train
```bash
python scripts/train.py \
  --data config/dataset.yaml \
  --model yolov8n.pt \
  --epochs 50 --imgsz 640 --batch 16
```
Artifacts: `runs/train/<exp>/weights/best.pt`

## 4) Inference (batch)
```bash
python scripts/infer.py \
  --weights runs/train/<exp>/weights/best.pt \
  --source datasets/propane_tanks/images/val \
  --conf 0.25
```
Outputs: annotated images in `runs/predict/*` and `detections.json`.

## 4.1) Export for Deployment (optional)
```bash
python scripts/export.py \
  --weights runs/train/<exp>/weights/best.pt \
  --format onnx --imgsz 640 --dynamic
```
This produces `best.onnx` next to your weights. You can run it with ONNX Runtime.

## 4.2) Lead Scoring (optional)
Compute a simple prospect score per image from detections (size, count heuristics):
```bash
python scripts/score_leads.py \
  --weights runs/train/<exp>/weights/best.pt \
  --source datasets/propane_tanks/images/val \
  --conf 0.25 --iou 0.45 \
  --out_csv lead_scores.csv
```
Or consume a prior `detections.json`:
```bash
python scripts/score_leads.py \
  --json runs/predict/batch/detections.json \
  --source datasets/propane_tanks/images/val \
  --out_csv lead_scores.csv
```

## 5) API (local)
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```
`POST /detect` with an image file returns detections and optional annotated image.

Example with curl:
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

Python client example:
```bash
python app/client_example.py --image /path/to/image.jpg --url http://localhost:8000/detect
```

## 6) Docker
```bash
docker build -t propane-vision .
docker run --rm -p 8000:8000 -e MODEL_WEIGHTS=runs/train/<exp>/weights/best.pt propane-vision
```

## Labeling Tips
- Label only visible tanks; include above-ground horizontal and vertical tanks
- Exclude water heaters/pressure vessels that are not propane storage
- Aim for ≥500 images; augment with different distances, lighting, and backgrounds