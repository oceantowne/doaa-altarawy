import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from ultralytics import YOLO


@dataclass
class Detection:
	image_path: str
	xyxy: Tuple[float, float, float, float]
	score: float
	class_id: int
	image_size: Tuple[int, int]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Score leads based on propane tank detections")
	parser.add_argument("--source", required=True, help="Image file or directory of images")
	parser.add_argument("--weights", default=None, help="Path to .pt weights; if omitted, use --json")
	parser.add_argument("--json", default=None, help="Optional detections.json to consume instead of running inference")
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--iou", type=float, default=0.45)
	parser.add_argument("--out_csv", default="lead_scores.csv")
	return parser.parse_args()


def load_image_size(path: str) -> Tuple[int, int]:
	with Image.open(path) as im:
		w, h = im.size
	return w, h


def load_from_json(json_path: str) -> List[Detection]:
	records: List[Detection] = []
	data = json.loads(Path(json_path).read_text())
	for item in data:
		img_path = item["image"]
		w, h = load_image_size(img_path)
		for det in item["detections"]:
			x1, y1, x2, y2 = det["xyxy"]
			records.append(Detection(
				image_path=img_path,
				xyxy=(x1, y1, x2, y2),
				score=float(det.get("score", 0.0)),
				class_id=int(det.get("class_id", -1)),
				image_size=(w, h),
			))
	return records


def run_model(weights: str, source: str, conf: float, iou: float) -> List[Detection]:
	model = YOLO(weights)
	results = model.predict(source=source, conf=conf, iou=iou, save=False, verbose=False)
	records: List[Detection] = []
	for r in results:
		img_path = str(Path(r.path))
		h, w = r.orig_shape[:2]
		for b in r.boxes:
			xyxy = b.xyxy[0].tolist()
			cls_id = int(b.cls[0]) if b.cls is not None else -1
			score = float(b.conf[0]) if b.conf is not None else 0.0
			records.append(Detection(
				image_path=img_path,
				xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
				score=score,
				class_id=cls_id,
				image_size=(w, h),
			))
	return records


def compute_score_for_image(detections: List[Detection]) -> Tuple[float, float]:
	if not detections:
		return 0.0, 0.0
	width, height = detections[0].image_size
	frame_area = float(width * height)
	if frame_area <= 0:
		return 0.0, 0.0

	area_ratios: List[float] = []
	confidences: List[float] = []
	for d in detections:
		x1, y1, x2, y2 = d.xyxy
		box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
		area_ratio = box_area / frame_area
		area_ratios.append(area_ratio)
		confidences.append(d.score)

	# Heuristic scoring: size + multiplicity influence lead potential
	# Size factor saturates around ~3% of frame area
	size_factors = [min(1.0, ar / 0.03) for ar in area_ratios]
	base_score = 70.0 * sum(size_factors)
	multi_bonus = 10.0 if len(detections) >= 2 else 0.0
	big_tank_bonus = 10.0 if max(area_ratios) >= 0.02 else 0.0
	final_score = min(100.0, base_score + multi_bonus + big_tank_bonus)
	avg_conf = sum(confidences) / len(confidences)
	return round(final_score, 2), round(avg_conf, 3)


def group_by_image(records: List[Detection]) -> Dict[str, List[Detection]]:
	by_image: Dict[str, List[Detection]] = {}
	for r in records:
		by_image.setdefault(r.image_path, []).append(r)
	return by_image


def write_csv(rows: List[Dict[str, Any]], out_csv: str) -> None:
	fieldnames = ["image", "num_detections", "avg_conf", "score"]
	with open(out_csv, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)
	print(f"Saved: {out_csv} ({len(rows)} rows)")


def main() -> None:
	args = parse_args()
	if args.json:
		records = load_from_json(args.json)
	elif args.weights:
		records = run_model(args.weights, args.source, args.conf, args.iou)
	else:
		raise SystemExit("Provide either --weights to run inference or --json to consume detections.json")

	by_image = group_by_image(records)
	rows: List[Dict[str, Any]] = []
	for image_path, dets in by_image.items():
		score, avg_conf = compute_score_for_image(dets)
		rows.append({
			"image": image_path,
			"num_detections": len(dets),
			"avg_conf": avg_conf,
			"score": score,
		})
	write_csv(rows, args.out_csv)


if __name__ == "__main__":
	main()