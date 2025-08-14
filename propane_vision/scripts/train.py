import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train YOLOv8 for propane tank detection")
	parser.add_argument("--data", type=str, default="config/dataset.yaml", help="Path to dataset yaml")
	parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights or yaml")
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--batch", type=int, default=16)
	parser.add_argument("--device", type=str, default="", help="CUDA device, i.e. '0' or '0,1' or 'cpu'")
	parser.add_argument("--project", type=str, default="runs/train", help="Project directory for runs")
	parser.add_argument("--name", type=str, default="propane_yolov8", help="Run name")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	project_dir = Path(args.project)
	project_dir.mkdir(parents=True, exist_ok=True)

	model = YOLO(args.model)
	results = model.train(
		data=args.data,
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		device=args.device,
		project=args.project,
		name=args.name,
	)
	print(results)


if __name__ == "__main__":
	main()