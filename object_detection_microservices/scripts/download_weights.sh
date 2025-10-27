#!/usr/bin/env bash
set -e
mkdir -p models
cd models
# coco.names
if [ ! -f coco.names ]; then
  echo "Downloading coco.names..."
  wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O coco.names
fi
# yolov3.cfg
if [ ! -f yolov3.cfg ]; then
  echo "Downloading yolov3.cfg..."
  wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O yolov3.cfg
fi
# yolov3.weights (large ~248MB)
if [ ! -f yolov3.weights ]; then
  echo "Downloading yolov3.weights (this may take several minutes)..."
  wget -q https://pjreddie.com/media/files/yolov3.weights -O yolov3.weights
fi
echo "Models downloaded to $(pwd)"
