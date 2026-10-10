"""FastAPI backend.   Run:  uvicorn app:app --reload
Docs UI:  http://127.0.0.1:8000/docs
"""
import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from predict import Classifier, analyze

app = FastAPI(title="AgriVision AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_clf = None


def get_classifier():
    global _clf
    if _clf is None:
        try:
            _clf = Classifier("models")
        except FileNotFoundError:
            raise HTTPException(503, "Model not found. Run train_model.py first.")
    return _clf


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = Form("en"), question: str = Form("")):
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Could not read the image.")
    return analyze(get_classifier(), img, question, lang)
