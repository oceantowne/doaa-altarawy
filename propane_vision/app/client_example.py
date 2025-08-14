import argparse
import json
import requests


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Call Propane Vision API /detect")
	parser.add_argument("--image", required=True, help="Path to an input image")
	parser.add_argument("--url", default="http://localhost:8000/detect", help="API endpoint")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	with open(args.image, "rb") as f:
		files = {"file": (args.image, f, "application/octet-stream")}
		r = requests.post(args.url, files=files, timeout=60)
		r.raise_for_status()
		print(json.dumps(r.json(), indent=2))


if __name__ == "__main__":
	main()