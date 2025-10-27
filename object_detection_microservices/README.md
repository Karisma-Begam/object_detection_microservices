# Object Detection Microservices (UI backend + AI backend)

## Overview
Two microservices:
- **ui_backend**: FastAPI service accepting image uploads, forwards to AI backend, returns results and annotated image.
- **ai_backend**: FastAPI service performing object detection using YOLOv3 (OpenCV DNN), returns JSON and saves annotated image.

This project targets CPU-only execution and is containerized with Docker.

## Prerequisites
- Docker & docker-compose (recommended) **or** Python 3.9+ and pip.
- Internet access during setup to download YOLOv3 weights and coco names.

## Quick start (docker-compose)
1. From the project root run:
   ```
   docker-compose up --build
   ```
2. UI backend will be available at `http://localhost:8000`.
   - POST `/detect` with form field `file` (multipart) to upload an image.
   - Response JSON contains detections and a URL to annotated image saved by AI backend.

## Quick start (local, without Docker)
1. Create virtualenv and install:
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Download weights and config (script included):
   ```
   ./scripts/download_weights.sh
   ```
3. Start AI backend:
   ```
   uvicorn ai_backend.app:app --host 0.0.0.0 --port 8001
   ```
4. Start UI backend:
   ```
   uvicorn ui_backend.app:app --host 0.0.0.0 --port 8000
   ```

## Notes & References
- Detection implementation uses OpenCV's DNN module with YOLOv3 cfg and weights.
- Reference YOLOv3 repo (for model information): https://github.com/ultralytics/yolov3
- coco.names and yolov3.weights are downloaded by the provided script.
