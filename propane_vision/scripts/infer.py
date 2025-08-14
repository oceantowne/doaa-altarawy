import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Batch inference with YOLOv8")
	parser.add_argument("--weights", type=str, required=True, help="Path to model weights .pt")
	parser.add_argument("--source", type=str, required=True, help="Image or directory of images")
	parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
	parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold")
	parser.add_argument("--project", type=str, default="runs/predict", help="Output project directory")
	parser.add_argument("--name", type=str, default="batch", help="Run name")
	parser.add_argument("--save_json", action="store_true", help="Save detections to detections.json")
	return parser.parse_args()


def save_json(pred_json: List[Dict[str, Any]], out_dir: Path) -> None:
	out_path = out_dir / "detections.json"
	with out_path.open("w") as f:
		json.dump(pred_json, f, indent=2)
	print(f"Saved: {out_path}")


def main() -> None:
	args = parse_args()
	out_dir = Path(args.project) / args.name
	out_dir.mkdir(parents=True, exist_ok=True)

	model = YOLO(args.weights)
	results = model.predict(
		source=args.source,
		conf=args.conf,
		iou=args.iou,
		project=args.project,
		name=args.name,
		save=True,
	)

	if args.save_json:
		pred_json: List[Dict[str, Any]] = []
		for r in results:
			img_path = str(Path(r.path))
			boxes = []
			for b in r.boxes:
				xyxy = b.xyxy[0].tolist()
				cls_id = int(b.cls[0]) if b.cls is not None else -1
				score = float(b.conf[0]) if b.conf is not None else 0.0
				boxes.append({"xyxy": xyxy, "class_id": cls_id, "score": score})
			pred_json.append({"image": img_path, "detections": boxes})
		save_json(pred_json, out_dir)


if __name__ == "__main__":
	main()