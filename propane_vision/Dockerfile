FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1 \
	UVICORN_WORKERS=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
	gcc \
	libgl1 \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app /app/app
COPY runs /app/runs

ENV MODEL_WEIGHTS="runs/train/propane_yolov8/weights/best.pt"

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]