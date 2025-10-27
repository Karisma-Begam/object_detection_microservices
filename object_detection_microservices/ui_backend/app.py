import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import os

app = FastAPI()
AI_BACKEND_URL = os.environ.get("AI_BACKEND_URL", "http://ai_backend:8001/detect")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    files = {"file": (file.filename, contents, file.content_type)}
    try:
        resp = requests.post(AI_BACKEND_URL, files=files, timeout=30)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reach AI backend: {e}")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return JSONResponse(resp.json())
