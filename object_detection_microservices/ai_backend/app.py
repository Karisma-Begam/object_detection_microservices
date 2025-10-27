import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import numpy as np
import cv2
from typing import List, Dict

app = FastAPI()

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Paths (expect download_weights.sh to populate these)
CFG_PATH = os.path.join(MODELS_DIR, "yolov3.cfg")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "yolov3.weights")
NAMES_PATH = os.path.join(MODELS_DIR, "coco.names")

# Load model lazily
net = None
classes = []

def load_model():
    global net, classes
    if net is not None:
        return
    if not (os.path.exists(CFG_PATH) and os.path.exists(WEIGHTS_PATH) and os.path.exists(NAMES_PATH)):
        raise FileNotFoundError("Model files not found. Run the download script in /scripts to fetch weights and cfg.")
    net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    with open(NAMES_PATH, 'r') as f:
        classes = [c.strip() for c in f.readlines()]

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        load_model()
        height, width = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB=True, crop=False)
        net.setInput(blob)
        ln = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(ln)
        boxes, confidences, classIDs = [], [], []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = int(np.argmax(scores))
                conf = float(scores[classID])
                if conf > 0.5:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, w, h) = box.astype("int")
                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(conf)
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        detections = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x,y,w,h = boxes[i]
                detections.append({
                    "class": classes[classIDs[i]] if classIDs[i] < len(classes) else str(classIDs[i]),
                    "confidence": float(confidences[i]),
                    "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                })
                # draw boxes
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                text = f"{classes[classIDs[i]]}:{confidences[i]:.2f}"
                cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        out_path = os.path.join(OUTPUT_DIR, f"annotated_{file.filename}")
        cv2.imwrite(out_path, img)
        result = {"filename": file.filename, "detections": detections, "annotated_image": f"/output/{os.path.basename(out_path)}"}
        return JSONResponse(result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/output/{fname}")
def get_output(fname: str):
    p = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(p, media_type="image/jpeg")
