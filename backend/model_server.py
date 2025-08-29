# backend/model_server.py
import os, joblib, numpy as np
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from backend.utils import fact_check_stub, detect_lang, translate_to_en

BASE = os.path.dirname(__file__)
ARTIFACTS = os.path.join(BASE, "train", "artifacts")  # put models here after download

# Safe defaults
_text_clf = None
_text_model_name = None
_embedder = None
_image_model = None

def _load_text_model():
    global _text_clf, _text_model_name, _embedder
    path = os.path.join(ARTIFACTS, "text_clf.joblib")
    if os.path.exists(path):
        _text_model_name, _text_clf = joblib.load(path)
        _embedder = SentenceTransformer(_text_model_name)
    else:
        print("Warning: text_clf.joblib not found at", path)

def _load_image_model():
    global _image_model
    path = os.path.join(ARTIFACTS, "image_model.h5")
    if os.path.exists(path):
        _image_model = tf.keras.models.load_model(path)
    else:
        print("Warning: image_model.h5 not found at", path)

# lazy load
def _ensure_models():
    if _text_clf is None:
        _load_text_model()
    if _image_model is None:
        _load_image_model()

def predict_text_raw(text):
    _ensure_models()
    if _text_clf is None or _embedder is None:
        return {"label":"unknown","prob_real":0.5,"prob_fake":0.5}
    emb = _embedder.encode([text], convert_to_numpy=True)
    probs = _text_clf.predict_proba(emb)[0]   # must be [prob_fake, prob_real]
    # ensure order: adapt if different
    prob_fake, prob_real = float(probs[0]), float(probs[1])
    return {"label":"real" if prob_real>prob_fake else "fake", "prob_real":prob_real, "prob_fake":prob_fake}

def predict_image_pil(pil_image):
    _ensure_models()
    if _image_model is None:
        return {"label":"unknown","prob_real":0.5,"prob_fake":0.5}
    img = pil_image.resize((224,224))
    arr = (np.array(img)/255.0).astype("float32")
    arr = np.expand_dims(arr,0)
    prob_real = float(_image_model.predict(arr)[0][0])  # sigmoid output
    return {"label":"real" if prob_real>0.5 else "fake", "prob_real":prob_real, "prob_fake":1-prob_real}

def ensemble_predict(text=None, pil_image=None):
    # simple weighted ensemble: text weight 0.7 if text exists, else 0.3
    text_res = predict_text_raw(text) if text else {"prob_real":0.5,"prob_fake":0.5}
    img_res = predict_image_pil(pil_image) if pil_image else {"prob_real":0.5,"prob_fake":0.5}
    w_text = 0.7 if text else 0.3
    w_img = 0.3 if pil_image else 0.7
    prob_real = w_text * text_res["prob_real"] + w_img * img_res["prob_real"]
    prob_fake = 1 - prob_real
    fc = fact_check_stub(text) if text else {"checked":False}
    return {"label": "real" if prob_real>=prob_fake else "fake",
            "prob_real": prob_real, "prob_fake": prob_fake, "fact_check": fc}