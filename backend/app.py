# backend/app.py
from fastapi import FastAPI, UploadFile, File, WebSocket
from pydantic import BaseModel
from model_server import ensemble_predict
from PIL import Image
import io

app = FastAPI(title="FakeCastX API")

class TextReq(BaseModel):
    text: str

@app.post("/predict_text")
async def predict_text(req: TextReq):
    return ensemble_predict(text=req.text)

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    return ensemble_predict(pil_image=img)

@app.websocket("/ws/predict")
async def ws_predict(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_json()
        text = data.get("text")
        res = ensemble_predict(text=text)
        await ws.send_json(res)