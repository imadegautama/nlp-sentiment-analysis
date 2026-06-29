"""src/llm.py — Lapisan LLM (OpenRouter) untuk EmoSense-ID Pro.

Modul ringan berbasis `requests` (tanpa SDK). Tugasnya BUKAN menggantikan model
NLP buatan sendiri, melainkan menambah lapisan augmentasi: setelah model TF-IDF
memberi Sentimen + Emosi + analisis Aspek (ABSA), LLM menyusun:
  1. ringkasan natural berbahasa Indonesia, dan
  2. saran balasan sopan untuk penjual.

Pola (klien, fallback antar-model gratis, resolusi API key, guardrail) diadaptasi
dari proyek UAS Pembelajaran Mesin penulis. Streamlit tidak diimpor di level atas
agar modul bisa dipakai juga oleh FastAPI/bot tanpa konteks Streamlit.

API key dibaca berurutan: st.session_state → st.secrets → env OPENROUTER_API_KEY.
"""

import os

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model default (gratis). Bisa diganti; model ":free" kadang kena rate-limit (429),
# karena itu disediakan daftar cadangan yang dicoba berurutan.
DEFAULT_MODEL = "google/gemma-4-31b-it:free"
FALLBACK_MODELS = [
    "google/gemma-4-31b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "google/gemma-4-26b-a4b-it:free",
]

API_KEY_NAME = "OPENROUTER_API_KEY"
_APP_REFERER = "https://github.com/"
_APP_TITLE = "EmoSense-ID"


def _secret(name, default=None):
    """Baca st.secrets[name] dengan aman (tak error bila secrets.toml tak ada)."""
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:
        return default


def get_api_key():
    """Ambil API key OpenRouter: sidebar → st.secrets → env var. None bila tak ada."""
    try:
        import streamlit as st

        ui_key = st.session_state.get("openrouter_api_key", "")
        if ui_key and ui_key.strip():
            return ui_key.strip()
    except Exception:
        pass

    secret_key = _secret(API_KEY_NAME)
    if secret_key and str(secret_key).strip():
        return str(secret_key).strip()

    env_key = os.environ.get(API_KEY_NAME, "")
    if env_key and env_key.strip():
        return env_key.strip()

    return None


SYSTEM_PROMPT = (
    "Anda adalah asisten analis ulasan produk e-commerce berbahasa Indonesia. "
    "Tugas Anda HANYA menafsirkan hasil analisis sebuah ulasan produk dan membantu "
    "penjual merespons. Jangan menjawab pertanyaan di luar konteks ulasan/produk.\n\n"
    "Anda menerima: teks ulasan asli, label Sentimen & Emosi dari model, serta "
    "rincian sentimen per-aspek (ABSA). Berdasarkan itu, hasilkan TEPAT dua bagian "
    "dengan format berikut (gunakan penanda persis, tanpa tambahan lain):\n\n"
    "RINGKASAN: <1-2 kalimat merangkum perasaan & poin utama pelanggan, "
    "sebutkan aspek positif/negatif bila ada>\n"
    "SARAN_BALASAN: <balasan sopan, empatik, dan solutif dari sudut pandang penjual, "
    "2-3 kalimat, sapa dengan 'Kak'>\n\n"
    "Gunakan Bahasa Indonesia yang natural. Jangan mengarang fakta di luar ulasan."
)


def _format_aspects(aspects: dict) -> str:
    """Ubah dict ABSA menjadi teks ringkas untuk prompt."""
    if not aspects:
        return "(tidak ada aspek spesifik terdeteksi)"
    lines = []
    for aspek, info in aspects.items():
        lines.append(f"- {aspek}: {info['sentiment']} ({info['confidence']:.0%})")
    return "\n".join(lines)


def _request_model(model, messages, headers, timeout):
    """Satu POST ke OpenRouter untuk satu model. Return (content|None, status, detail)."""
    payload = {"model": model, "messages": messages}
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", {}).get("message") or resp.text[:200]
        except Exception:
            detail = resp.text[:200]
        return None, resp.status_code, detail
    try:
        content = resp.json()["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, ValueError):
        content = ""
    if content.strip():
        return content, 200, None
    return None, 200, "balasan kosong"


def chat(messages, api_key, model=DEFAULT_MODEL, timeout=60):
    """Panggil OpenRouter dengan fallback antar-model. Return str balasan non-kosong.

    Raises RuntimeError (pesan ramah Bahasa Indonesia) bila semua model gagal.
    """
    if not api_key:
        raise RuntimeError("API key OpenRouter belum diisi.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": _APP_REFERER,
        "X-Title": _APP_TITLE,
    }
    candidates = [model] + [m for m in FALLBACK_MODELS if m != model]
    last_detail = ""
    for m in candidates:
        try:
            content, status, detail = _request_model(m, messages, headers, timeout)
        except requests.exceptions.Timeout:
            last_detail = f"{m}: timeout"
            continue
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Gagal menghubungi OpenRouter: {exc}")

        if content is not None:
            return content
        if status == 401:
            raise RuntimeError("API key OpenRouter tidak valid (401). Periksa key Anda.")
        last_detail = f"{m}: {status} {detail}"

    raise RuntimeError(
        "Semua model AI gratis sedang sibuk/limit (429). Coba lagi nanti, atau "
        f"tambahkan BYOK di openrouter.ai. (detail: {last_detail})"
    )


def _parse_response(text: str) -> dict:
    """Pisahkan keluaran LLM jadi {summary, suggested_reply} berdasarkan penanda."""
    summary, reply = text.strip(), ""
    upper = text.upper()
    if "SARAN_BALASAN:" in upper:
        idx = upper.index("SARAN_BALASAN:")
        summary = text[:idx]
        reply = text[idx + len("SARAN_BALASAN:"):].strip()
    # Bersihkan penanda RINGKASAN: dari bagian ringkasan.
    if "RINGKASAN:" in summary.upper():
        i = summary.upper().index("RINGKASAN:")
        summary = summary[i + len("RINGKASAN:"):]
    return {"summary": summary.strip(), "suggested_reply": reply.strip()}


def analyze(review: str, sentiment: str, emotion: str, aspects: dict,
            api_key: str | None = None, model: str = DEFAULT_MODEL) -> dict:
    """Hasilkan ringkasan natural + saran balasan dari hasil analisis model.

    Returns {"summary": str, "suggested_reply": str}.
    Raises RuntimeError bila API key tak ada / semua model gagal (ditangani pemanggil).
    """
    key = api_key or get_api_key()
    if not key:
        raise RuntimeError("API key OpenRouter belum diisi.")

    user_msg = (
        f"Ulasan: \"{review.strip()}\"\n\n"
        f"Hasil model:\n"
        f"- Sentimen keseluruhan: {sentiment}\n"
        f"- Emosi: {emotion}\n"
        f"- Sentimen per aspek:\n{_format_aspects(aspects)}\n\n"
        "Tolong buat RINGKASAN dan SARAN_BALASAN sesuai format."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    raw = chat(messages, key, model=model)
    return _parse_response(raw)


if __name__ == "__main__":
    # Demo manual (butuh OPENROUTER_API_KEY di environment).
    contoh_aspek = {
        "Pengiriman": {"sentiment": "Positive", "confidence": 0.93, "clause": "pengiriman cepat"},
        "Kualitas Produk": {"sentiment": "Negative", "confidence": 0.92, "clause": "barangnya jelek"},
    }
    try:
        hasil = analyze(
            "Pengiriman cepat banget tapi barangnya jelek dan harga kemahalan",
            "Negative", "Anger", contoh_aspek,
        )
        print("RINGKASAN  :", hasil["summary"])
        print("SARAN BALAS:", hasil["suggested_reply"])
    except RuntimeError as e:
        print("LLM tidak aktif:", e)
