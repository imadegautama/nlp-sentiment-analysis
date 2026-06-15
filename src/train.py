"""Pelatihan model EmoSense-ID (Tahap 1 UAS NLP).

Alur:
  1. Muat dataset PRDECT-ID, bersihkan teks (preprocessing.clean_text).
  2. Bagi data latih/uji (stratified, reproducible).
  3. Representasi fitur dengan TF-IDF (uni+bigram).
  4. Bandingkan 3 kandidat model (LogReg, LinearSVC, MultinomialNB) via cross-
     validation macro-F1, lalu latih model final = TF-IDF + Logistic Regression
     untuk DUA tugas: Sentimen dan Emosi.
  5. Simpan vectorizer + kedua model ke models/ (joblib).

Kenapa Logistic Regression sebagai model final:
  - punya predict_proba  -> skor keyakinan untuk ditampilkan di aplikasi
  - punya coef_           -> dipakai untuk explainability "kata berpengaruh"
  - akurasinya kompetitif dengan SVM pada TF-IDF, tapi jauh lebih mudah
    dijelaskan saat ujian.

Jalankan: python src/train.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocessing import clean_text

# --- Lokasi file ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "prdect_id.csv"
CLEAN_CACHE = ROOT / "data" / "prdect_id_clean.parquet"  # cache teks bersih
MODELS_DIR = ROOT / "models"

TEXT_COL = "Customer Review"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_and_split():
    """Muat dataset, bersihkan teks, kembalikan (df_train, df_test).

    Hasil pembersihan di-cache ke parquet supaya train.py & evaluate.py tidak
    perlu menjalankan stemming yang lambat berulang kali. Pembagian latih/uji
    memakai random_state tetap + stratifikasi Emosi agar bisa direproduksi
    dan distribusi kelas (yang tak seimbang) terjaga di kedua subset.
    """
    if CLEAN_CACHE.exists():
        df = pd.read_parquet(CLEAN_CACHE)
    else:
        df = pd.read_csv(DATA_CSV)
        print(f"Membersihkan {len(df)} ulasan (stemming Sastrawi)...")
        df["clean"] = df[TEXT_COL].apply(clean_text)
        df = df[df["clean"].str.len() > 0].reset_index(drop=True)
        df[["clean", "Sentiment", "Emotion"]].to_parquet(CLEAN_CACHE, index=False)
        print(f"Teks bersih disimpan ke {CLEAN_CACHE}")

    df_train, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["Emotion"],
    )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def build_vectorizer() -> TfidfVectorizer:
    """TF-IDF: unigram + bigram (bigram menangkap pola seperti 'tidak bagus')."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,            # buang istilah yang muncul di <3 dokumen (noise)
        max_features=20_000,
        sublinear_tf=True,   # 1+log(tf), meredam dominasi kata yang sering muncul
    )


def compare_models(X_train, y_train, task_name: str) -> None:
    """Bandingkan 3 kandidat model dengan 5-fold CV (macro-F1) dan cetak tabel."""
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Linear SVM": LinearSVC(class_weight="balanced"),
        "Multinomial NB": MultinomialNB(),
    }
    print(f"\n=== Perbandingan model (5-fold CV, macro-F1) — tugas: {task_name} ===")
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")
        print(f"  {name:<22}: {scores.mean():.4f} (+/- {scores.std():.4f})")


def train_final(X_train, y_train) -> LogisticRegression:
    """Latih model final (Logistic Regression) pada seluruh data latih."""
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    df_train, df_test = load_and_split()
    print(f"Data latih: {len(df_train)} | Data uji: {len(df_test)}")

    # Representasi fitur: fit HANYA pada data latih, lalu transform keduanya.
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(df_train["clean"])
    X_test = vectorizer.transform(df_test["clean"])
    print(f"Dimensi fitur TF-IDF: {X_train.shape[1]} fitur")

    metadata = {
        "n_features": int(X_train.shape[1]),
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "vectorizer": "TF-IDF (1,2)-gram, sublinear_tf, min_df=3",
        "final_model": "LogisticRegression(class_weight='balanced')",
        "tasks": {},
    }

    for task, col, fname in [
        ("Sentimen", "Sentiment", "classic_sentiment.joblib"),
        ("Emosi", "Emotion", "classic_emotion.joblib"),
    ]:
        y_train, y_test = df_train[col], df_test[col]
        compare_models(X_train, y_train, task)

        model = train_final(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"  -> Model final {task}: akurasi data uji = {test_acc:.4f}")

        joblib.dump(model, MODELS_DIR / fname)
        metadata["tasks"][task] = {
            "label_col": col,
            "classes": list(model.classes_),
            "test_accuracy": round(float(test_acc), 4),
            "model_file": fname,
        }

    # Simpan vectorizer (komponen pra-pemrosesan) & metadata.
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )

    print(f"\nSelesai. Artefak tersimpan di {MODELS_DIR}/")
    print("  - tfidf_vectorizer.joblib")
    print("  - classic_sentiment.joblib")
    print("  - classic_emotion.joblib")
    print("  - metadata.json")


if __name__ == "__main__":
    main()
