"""api.py — REST API (FastAPI) untuk EmoSense-ID Pro.

Memberi interface HTTP agar orkestrator non-Python (mis. **n8n** untuk bot
Telegram) bisa memakai model NLP yang sama dengan aplikasi Streamlit.

Endpoint:
  GET  /health   → cek hidup + daftar kelas.
  POST /analyze  → {text} → Sentimen + Emosi + ABSA (+ ringkasan & saran balasan AI
                   bila OPENROUTER_API_KEY diset di server).

Reuse penuh modul di src/: preprocessing.clean_text, absa.analyze_aspects, llm.analyze.
TIDAK meng-import streamlit_app.py (agar tidak mengeksekusi Streamlit).

Jalankan:
  uvicorn api:app --host 0.0.0.0 --port 8800
  # aktifkan fitur AI:
  OPENROUTER_API_KEY="sk-or-..." uvicorn api:app --port 8800
"""

import json
import sys
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Tambahkan src/ ke path agar modul inti bisa diimpor.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import clean_text  # noqa: E402
from absa import analyze_aspects  # noqa: E402
from llm import analyze as llm_analyze  # noqa: E402
from llm import get_api_key  # noqa: E402

MODELS_DIR = ROOT / "models"

# Label Indonesia untuk kenyamanan klien (bot/n8n).
SENTIMENT_ID = {"Positive": "Positif", "Negative": "Negatif"}
EMOTION_ID = {
    "Happy": "Senang", "Sadness": "Sedih", "Anger": "Marah",
    "Fear": "Takut", "Love": "Cinta",
}

# --- Muat artefak SEKALI saat startup --------------------------------------
if not (MODELS_DIR / "tfidf_vectorizer.joblib").exists():
    raise RuntimeError(
        f"Artefak model tidak ditemukan di {MODELS_DIR}. Jalankan dulu: python src/train.py"
    )

vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
sentiment_model = joblib.load(MODELS_DIR / "classic_sentiment.joblib")
emotion_model = joblib.load(MODELS_DIR / "classic_emotion.joblib")
metadata = json.loads((MODELS_DIR / "metadata.json").read_text())

app = FastAPI(title="EmoSense-ID API", version="1.0")


class AnalyzeRequest(BaseModel):
    text: str


@app.get("/")
def root():
    """Info ringkas + arahan ke endpoint yang tersedia (mencegah 404 membingungkan)."""
    return {
        "service": "EmoSense-ID API",
        "endpoints": {
            "GET /health": "cek status & kelas",
            "POST /analyze": "analisis ulasan (body JSON: {\"text\": \"...\"})",
            "GET /docs": "dokumentasi interaktif (Swagger UI)",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "sentiment_classes": list(sentiment_model.classes_),
        "emotion_classes": list(emotion_model.classes_),
        "ai_enabled": get_api_key() is not None,
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' kosong.")

    cleaned = clean_text(text)
    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail="Teks terlalu pendek / tidak ada kata bermakna setelah pra-pemrosesan.",
        )

    x = vectorizer.transform([cleaned])

    sent = sentiment_model.predict(x)[0]
    sent_conf = float(sentiment_model.predict_proba(x)[0].max())
    emo = emotion_model.predict(x)[0]
    emo_conf = float(emotion_model.predict_proba(x)[0].max())

    aspects_raw = analyze_aspects(text, vectorizer, sentiment_model)
    aspects = [
        {
            "aspect": aspek,
            "sentiment": info["sentiment"],
            "sentiment_id": SENTIMENT_ID.get(info["sentiment"], info["sentiment"]),
            "confidence": round(info["confidence"], 4),
            "clause": info["clause"],
        }
        for aspek, info in aspects_raw.items()
    ]

    # Lapisan AI (opsional) — hanya jika OPENROUTER_API_KEY tersedia di server.
    ai = None
    if get_api_key():
        try:
            ai = llm_analyze(text, sent, emo, aspects_raw)
        except RuntimeError as exc:
            ai = {"error": str(exc)}

    return {
        "text": text,
        "cleaned": cleaned,
        "sentiment": {
            "label": sent,
            "label_id": SENTIMENT_ID.get(sent, sent),
            "confidence": round(sent_conf, 4),
        },
        "emotion": {
            "label": emo,
            "label_id": EMOTION_ID.get(emo, emo),
            "confidence": round(emo_conf, 4),
        },
        "aspects": aspects,
        "ai": ai,
    }
