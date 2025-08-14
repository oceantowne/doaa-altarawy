import argparse
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Export YOLOv8 model to various formats")
	parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt weights")
	parser.add_argument("--format", type=str, default="onnx", choices=[
		"onnx", "openvino", "torchscript", "coreml", "engine", "tflite"
	])
	parser.add_argument("--imgsz", type=int, default=640, help="Export image size")
	parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axes (where supported)")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	model = YOLO(args.weights)
	export_file = model.export(format=args.format, imgsz=args.imgsz, dynamic=args.dynamic, simplify=True)
	print(f"Exported: {export_file}")


if __name__ == "__main__":
	main()