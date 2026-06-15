"""EmoSense-ID — Aplikasi Streamlit (Tahap 2 UAS NLP).

Menerima teks ulasan dari pengguna, lalu menampilkan:
  - prediksi Sentimen (Positif/Negatif) + skor keyakinan
  - prediksi Emosi (Senang/Sedih/Marah/Takut/Cinta) + skor keyakinan
  - kata-kata paling berpengaruh terhadap prediksi (explainability)

PENTING: teks pengguna dibersihkan dengan fungsi clean_text() yang SAMA PERSIS
dengan yang dipakai saat pelatihan (diimpor dari src/preprocessing.py), lalu
diubah menjadi fitur dengan TF-IDF vectorizer yang sama. Inilah yang menjamin
input pengguna diperlakukan identik dengan data latih (syarat soal Tahap 2).

Jalankan: streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Tambahkan folder src/ ke path agar bisa mengimpor modul pra-pemrosesan.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import clean_text  # noqa: E402

MODELS_DIR = ROOT / "models"

# --- Pemetaan label ke teks Indonesia + emoji ------------------------------
SENTIMENT_LABELS = {
    "Positive": ("Positif", "😊"),
    "Negative": ("Negatif", "😞"),
}
EMOTION_LABELS = {
    "Happy": ("Senang", "😄"),
    "Sadness": ("Sedih", "😢"),
    "Anger": ("Marah", "😠"),
    "Fear": ("Takut", "😨"),
    "Love": ("Cinta", "🥰"),
}

CONTOH_ULASAN = {
    "Pilih contoh…": "",
    "Ulasan positif": "Barangnya bagus banget, pengiriman cepat dan penjual ramah. Sangat puas!",
    "Ulasan negatif": "Kecewa berat, barang datang rusak dan tidak sesuai deskripsi. Lama pula sampainya.",
    "Ulasan marah": "Parah! Sudah bayar mahal tapi pelayanannya buruk sekali, bikin emosi.",
    "Ulasan takut/ragu": "Agak khawatir barangnya ori atau tidak, takut tertipu seperti toko sebelah.",
}


@st.cache_resource
def load_artifacts():
    """Muat vectorizer + kedua model sekali saja (di-cache antar interaksi)."""
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    sentiment_model = joblib.load(MODELS_DIR / "classic_sentiment.joblib")
    emotion_model = joblib.load(MODELS_DIR / "classic_emotion.joblib")
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text())
    feature_names = vectorizer.get_feature_names_out()
    return vectorizer, sentiment_model, emotion_model, metadata, feature_names


def top_contributing_words(model, x_row, feature_names, predicted_class, n=8):
    """Kata paling mendorong prediksi ke `predicted_class` (kontribusi positif).

    Kontribusi tiap fitur = nilai TF-IDF kata × koefisien model untuk kelas
    terprediksi. Untuk kasus biner (sentimen), koefisien tersimpan untuk satu
    arah (kelas positif/classes_[1]); bila prediksi adalah kelas lain, arah
    koefisien dibalik.
    """
    x_dense = x_row.toarray().ravel()
    nonzero = x_dense.nonzero()[0]
    if len(nonzero) == 0:
        return []

    classes = list(model.classes_)
    coef = model.coef_
    if coef.shape[0] == 1:  # klasifikasi biner
        direction = 1.0 if predicted_class == classes[1] else -1.0
        coef_vec = coef[0] * direction
    else:  # multikelas (one-vs-rest)
        coef_vec = coef[classes.index(predicted_class)]

    contribs = [(feature_names[i], x_dense[i] * coef_vec[i]) for i in nonzero]
    contribs = [c for c in contribs if c[1] > 0]
    contribs.sort(key=lambda c: c[1], reverse=True)
    return contribs[:n]


def prob_dataframe(model, x_row, label_map):
    """DataFrame probabilitas tiap kelas (untuk bar chart), label diterjemahkan."""
    proba = model.predict_proba(x_row)[0]
    rows = []
    for cls, p in zip(model.classes_, proba):
        nama, emoji = label_map.get(cls, (cls, ""))
        rows.append({"Kelas": f"{emoji} {nama}", "Keyakinan": float(p)})
    return pd.DataFrame(rows).set_index("Kelas").sort_values("Keyakinan")


# --- Tata letak halaman -----------------------------------------------------
st.set_page_config(page_title="EmoSense-ID", page_icon="💬", layout="centered")

vectorizer, sentiment_model, emotion_model, metadata, feature_names = load_artifacts()

st.title("💬 EmoSense-ID")
st.caption(
    "Analisis **Sentimen + Emosi** ulasan produk berbahasa Indonesia "
    "(TF-IDF + Logistic Regression)."
)

with st.sidebar:
    st.header("ℹ️ Tentang Model")
    st.markdown(
        f"""
- **Dataset:** PRDECT-ID (±5.400 ulasan Tokopedia)
- **Representasi:** {metadata['vectorizer']}
- **Jumlah fitur:** {metadata['n_features']:,}
- **Model:** Logistic Regression
- **Akurasi uji Sentimen:** {metadata['tasks']['Sentimen']['test_accuracy']:.2%}
- **Akurasi uji Emosi:** {metadata['tasks']['Emosi']['test_accuracy']:.2%}
"""
    )
    st.markdown("---")
    st.markdown(
        "Teks Anda melewati pra-pemrosesan & TF-IDF yang **identik** dengan "
        "saat pelatihan, lalu diklasifikasikan oleh dua model terpisah."
    )

contoh_pilih = st.selectbox("Coba contoh ulasan:", list(CONTOH_ULASAN.keys()))
teks = st.text_area(
    "Tulis atau tempel ulasan produk di sini:",
    value=CONTOH_ULASAN[contoh_pilih],
    height=140,
    placeholder="Contoh: Barangnya bagus, pengiriman cepat, penjual ramah!",
)

if st.button("🔍 Analisis", type="primary", use_container_width=True):
    if not teks.strip():
        st.warning("Mohon masukkan teks ulasan terlebih dahulu.")
        st.stop()

    bersih = clean_text(teks)
    if not bersih:
        st.warning(
            "Setelah dibersihkan, tidak ada kata bermakna yang tersisa. "
            "Coba ulasan yang lebih panjang."
        )
        st.stop()

    x = vectorizer.transform([bersih])

    sent_pred = sentiment_model.predict(x)[0]
    emo_pred = emotion_model.predict(x)[0]
    sent_nama, sent_emoji = SENTIMENT_LABELS.get(sent_pred, (sent_pred, ""))
    emo_nama, emo_emoji = EMOTION_LABELS.get(emo_pred, (emo_pred, ""))
    sent_conf = sentiment_model.predict_proba(x)[0].max()
    emo_conf = emotion_model.predict_proba(x)[0].max()

    st.markdown("### Hasil")
    c1, c2 = st.columns(2)
    c1.metric("Sentimen", f"{sent_emoji} {sent_nama}", f"{sent_conf:.0%} yakin")
    c2.metric("Emosi", f"{emo_emoji} {emo_nama}", f"{emo_conf:.0%} yakin")

    st.markdown("#### Distribusi keyakinan")
    g1, g2 = st.columns(2)
    with g1:
        st.caption("Sentimen")
        st.bar_chart(prob_dataframe(sentiment_model, x, SENTIMENT_LABELS))
    with g2:
        st.caption("Emosi")
        st.bar_chart(prob_dataframe(emotion_model, x, EMOTION_LABELS))

    st.markdown("#### 🔎 Kenapa? Kata paling berpengaruh")
    e1, e2 = st.columns(2)
    with e1:
        st.caption(f"Mendorong sentimen **{sent_nama}**")
        kata_sent = top_contributing_words(sentiment_model, x, feature_names, sent_pred)
        if kata_sent:
            st.dataframe(
                pd.DataFrame(kata_sent, columns=["Kata", "Kontribusi"]),
                hide_index=True, use_container_width=True,
            )
        else:
            st.write("—")
    with e2:
        st.caption(f"Mendorong emosi **{emo_nama}**")
        kata_emo = top_contributing_words(emotion_model, x, feature_names, emo_pred)
        if kata_emo:
            st.dataframe(
                pd.DataFrame(kata_emo, columns=["Kata", "Kontribusi"]),
                hide_index=True, use_container_width=True,
            )
        else:
            st.write("—")

    with st.expander("Lihat teks setelah pra-pemrosesan"):
        st.code(bersih or "(kosong)", language=None)
