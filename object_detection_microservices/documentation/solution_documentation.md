# Solution Documentation

## Approach
- Built two FastAPI services (UI backend and AI backend). The UI service accepts image uploads and proxies them to the AI backend.
- AI backend performs object detection using OpenCV's DNN module and YOLOv3 model files (cfg + weights + coco.names).
- AI backend saves annotated images to `ai_backend/output/` and returns structured JSON describing detections.

## Implementation Notes
- Detection threshold, NMS threshold and input size are configurable in code (defaults used).
- Uses CPU-only OpenCV DNN (no CUDA).

## References
- YOLOv3 paper and implementations. Example repo used for reference: https://github.com/ultralytics/yolov3
- OpenCV DNN documentation for Darknet models.

## How to reproduce
See README.md in project root. Use `scripts/download_weights.sh` to get model files.
