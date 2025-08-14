import io
import os
from typing import List

import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "runs/train/propane_yolov8/weights/best.pt")
CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))
IOU_THRES = float(os.getenv("IOU_THRES", "0.45"))

app = FastAPI(title="Propane Vision API", version="1.0.0")

model: YOLO | None = None


def load_model() -> YOLO:
	global model
	if model is None:
		model = YOLO(MODEL_WEIGHTS)
	return model


@app.on_event("startup")
def _startup() -> None:
	load_model()


@app.post("/detect")
def detect(
	file: UploadFile = File(...),
	conf: float = Query(CONF_THRES, ge=0.0, le=1.0),
	iou: float = Query(IOU_THRES, ge=0.0, le=1.0),
):
	image_bytes = file.file.read()
	image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
	img_array = np.array(image)

	mdl = load_model()
	results = mdl.predict(source=img_array, conf=conf, iou=iou, save=False, verbose=False)

	detections = []
	if results:
		r = results[0]
		for b in r.boxes:
			xyxy = b.xyxy[0].tolist()
			class_id = int(b.cls[0]) if b.cls is not None else -1
			score = float(b.conf[0]) if b.conf is not None else 0.0
			detections.append({
				"xyxy": xyxy,
				"class_id": class_id,
				"score": score,
			})

	return JSONResponse({
		"num_detections": len(detections),
		"detections": detections,
	})