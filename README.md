# 💬 EmoSense-ID — Analisis Sentimen & Emosi Ulasan Produk Indonesia

Sistem NLP yang menerima teks ulasan produk berbahasa Indonesia, lalu memprediksi
**dua hal sekaligus**:

- **Sentimen** — Positif / Negatif
- **Emosi** — Senang / Sedih / Marah / Takut / Cinta

Model dilatih dengan **TF-IDF + Logistic Regression** (scikit-learn) di atas dataset
**PRDECT-ID**, lalu disajikan lewat aplikasi web **Streamlit** yang juga menjelaskan
*kata-kata paling berpengaruh* di balik tiap prediksi (explainability).

> Proyek UAS mata kuliah Natural Language Processing — Informatika, Primakara University.

![Demo aplikasi EmoSense-ID](reports/demo_app.png)

---

## ✨ Fitur Unggulan

1. **Dua tugas dalam satu sistem** — sentimen *dan* emosi dari satu ulasan.
2. **Explainability** — menampilkan kata (termasuk bigram seperti `tidak sesuai`)
   yang paling mendorong sebuah prediksi, dihitung dari koefisien model linear.
3. **Penanganan negasi** — kata seperti *tidak, bukan, jangan* sengaja dipertahankan
   saat pra-pemrosesan agar makna `tidak bagus` tidak terbalik menjadi `bagus`.
4. **Skor keyakinan** divisualkan sebagai bar chart per kelas.
5. **Konsistensi train/inference** — teks pengguna melewati pra-pemrosesan & TF-IDF
   yang identik dengan saat pelatihan (satu sumber kode: `src/preprocessing.py`).

---

## 📁 Struktur Proyek

```
uas-nlp/
├── data/
│   ├── prdect_id.csv               # dataset (unduh via src/download_data.py)
│   └── prdect_id_clean.parquet     # cache teks hasil pra-pemrosesan (auto)
├── src/
│   ├── download_data.py            # unduh dataset PRDECT-ID
│   ├── preprocessing.py            # clean_text() — dipakai training & app
│   ├── train.py                    # TF-IDF + bandingkan model + latih + simpan
│   └── evaluate.py                 # metrik + confusion matrix
├── models/                         # artefak hasil training (.joblib + metadata)
├── reports/                        # confusion matrix, ringkasan metrik, demo
├── notebooks/
│   └── 01_eksplorasi_data.ipynb    # EDA
├── app/
│   └── streamlit_app.py            # aplikasi web
├── requirements.txt
└── README.md
```

---

## 📂 Dataset

**PRDECT-ID** (*Indonesian Product Reviews Dataset for Emotions Classification Tasks*) —
±5.400 ulasan produk dari Tokopedia (29 kategori), dianotasi oleh ahli psikologi klinis.
Setiap ulasan memiliki label **Sentiment** (Positive/Negative) dan **Emotion**
(Anger, Fear, Happy, Love, Sadness).

- Sumber: <https://github.com/rhiosutoyo/PRDECT-ID-Indonesian-Product-Reviews-Dataset>
- Publikasi: Sutoyo dkk., *Data in Brief* (2022). DOI Mendeley: `10.17632/574v66hf2v.1`

**Distribusi label**

| Sentimen | Jumlah | | Emosi | Jumlah |
|---|---|---|---|---|
| Negative | 2.821 | | Happy | 1.770 |
| Positive | 2.579 | | Sadness | 1.202 |
| | | | Fear | 920 |
| | | | Love | 809 |
| | | | Anger | 699 |

Sentimen relatif seimbang; emosi tidak seimbang → evaluasi memakai **macro-F1** dan
pelatihan memakai `class_weight='balanced'`.

---

## 🚀 Cara Menjalankan

> Prasyarat: Python 3.11+ (diuji pada 3.13).

```bash
# 1. Buat virtual environment & install dependensi
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Unduh dataset ke data/prdect_id.csv
python src/download_data.py

# 3. Latih model (menghasilkan artefak di models/)
python src/train.py

# 4. Evaluasi (menghasilkan confusion matrix & ringkasan di reports/)
python src/evaluate.py

# 5. Jalankan aplikasi
streamlit run app/streamlit_app.py
```

Aplikasi akan terbuka di <http://localhost:8501>.

> Untuk membuka notebook EDA: `pip install jupyter` lalu
> `jupyter notebook notebooks/01_eksplorasi_data.ipynb`.

---

## 📊 Hasil Evaluasi (data uji, 1.080 ulasan)

| Tugas | Accuracy | Macro-F1 |
|---|---|---|
| **Sentimen** | 0.9315 | 0.9311 |
| **Emosi** | 0.6361 | 0.6086 |

Perbandingan kandidat model (5-fold CV, macro-F1) yang mendasari pemilihan
**Logistic Regression**:

| Model | Sentimen | Emosi |
|---|---|---|
| **Logistic Regression** | 0.9402 | **0.5870** |
| Linear SVM | 0.9433 | 0.5752 |
| Multinomial NB | 0.9404 | 0.4556 |

Logistic Regression menang pada tugas Emosi (yang lebih sulit) dan setara pada Sentimen,
**sekaligus** menyediakan `predict_proba` (skor keyakinan) dan `coef_` (explainability) —
karena itu dipilih sebagai model final.

Confusion matrix tersimpan di `reports/confusion_matrix_sentimen.png` dan
`reports/confusion_matrix_emosi.png`. Ringkasan lengkap: `reports/metrics_summary.md`.

---

## 🧠 Cara Kerja (Pipeline)

```
Teks ulasan
   │
   ▼
clean_text()  ── case folding → hapus URL/angka/simbol → hapus stopword
   │              (negasi dipertahankan) → stemming Sastrawi
   ▼
TfidfVectorizer  ── unigram + bigram, sublinear_tf, min_df=3  → vektor fitur
   │
   ├──► Logistic Regression (Sentimen)  → label + probabilitas
   └──► Logistic Regression (Emosi)     → label + probabilitas
                                            │
                                            ▼
                       Explainability: nilai TF-IDF × koefisien
                       → kata paling berpengaruh untuk kelas terprediksi
```

**Penting:** langkah `clean_text()` dan `TfidfVectorizer` yang sama persis dipakai
baik saat pelatihan maupun saat memproses input pengguna di aplikasi. Vectorizer
yang sudah di-`fit` disimpan (`models/tfidf_vectorizer.joblib`) dan dimuat ulang —
**tidak** dilatih ulang.

---

## 🖥️ Dokumentasi Antarmuka (Aplikasi Streamlit)

- **Input:** sebuah text area untuk mengetik/menempel ulasan, plus dropdown contoh siap-pakai.
- **Output:**
  - dua kartu metrik (Sentimen & Emosi) beserta % keyakinan;
  - dua bar chart distribusi probabilitas tiap kelas;
  - dua tabel "kata paling berpengaruh" terhadap prediksi;
  - expander untuk melihat teks setelah pra-pemrosesan.
- **Sidebar:** ringkasan informasi model (dataset, representasi, jumlah fitur, akurasi).

---

## ☁️ Deployment ke Streamlit Community Cloud

1. *Push* repository ini ke GitHub (sertakan folder `models/` — ukurannya kecil, ~300 KB).
2. Buka <https://share.streamlit.io> → **New app** → pilih repo & branch.
3. Set **Main file path** ke `app/streamlit_app.py`.
4. Streamlit Cloud otomatis membaca `requirements.txt`. Klik **Deploy**.

Karena model berbasis TF-IDF berukuran kecil, aplikasi berjalan ringan di *free tier*.

---

## 📝 Catatan Akademik

Setiap baris kode dapat dijelaskan: TF-IDF, perbedaan Logistic Regression / SVM / Naive
Bayes, metrik precision/recall/F1, pembacaan confusion matrix, serta alasan menjaga kata
negasi. Proyek ini menekankan **pemahaman**, bukan sekadar menjalankan kode.

---

## 📚 Referensi

- Sutoyo, R., dkk. (2022). *PRDECT-ID: Indonesian product reviews dataset for emotions
  classification tasks.* Data in Brief.
- Sastrawi — stemmer Bahasa Indonesia: <https://github.com/har07/PySastrawi>
- scikit-learn: <https://scikit-learn.org> · Streamlit: <https://streamlit.io>
