# =============================================================================
# Dockerfile — aplikasi Streamlit "EmoSense-ID".
# -----------------------------------------------------------------------------
# Mengikuti pola Dockerfile proyek UAS Pembelajaran Mesin. `libgomp1` ditambahkan
# karena scikit-learn membutuhkannya saat runtime pada image slim.
# Streamlit Community Cloud TIDAK memakai file ini (ia langsung baca
# requirements.txt) — Dockerfile murni untuk Coolify/host container.
# =============================================================================

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependency dulu (layer terpisah → cache build lebih cepat).
# requirements.txt memuat `-r requirements-base.txt`, jadi keduanya disalin.
COPY requirements-base.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Salin sisa kode + model.
COPY . .

# Banyak host container memberi port lewat env $PORT; fallback 8501 untuk lokal.
EXPOSE 8501
CMD streamlit run app/streamlit_app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true
