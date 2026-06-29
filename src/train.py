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

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
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


def train_final(X_train, y_train, C: float = 1.0) -> LogisticRegression:
    """Latih model final (Logistic Regression) pada seluruh data latih."""
    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=C)
    model.fit(X_train, y_train)
    return model


# Grid hyperparameter untuk tuning (TF-IDF + Logistic Regression).
_PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [2, 3, 5],
    "tfidf__max_features": [10_000, 20_000, None],
    "clf__C": [0.1, 1, 3, 10],
}


def tune_models(df_train):
    """Hyperparameter tuning via GridSearchCV (scoring=macro-F1, cv=5).

    Strategi (mempertahankan arsitektur 1 vectorizer + 2 model):
      1. Tuning Pipeline(TF-IDF → LogReg) penuh pada tugas **Emosi** (tugas tersulit
         & paling diuntungkan) → menentukan hyperparameter TF-IDF bersama + C emosi.
      2. Bangun vectorizer bersama dengan param TF-IDF terbaik, fit di data latih.
      3. Tuning `C` untuk Sentimen pada fitur tersebut.
    Mengembalikan (vectorizer, model_sentimen, model_emosi, info_tuning).
    """
    base = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipe = Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True)), ("clf", base)])

    print("\n=== Tuning (GridSearchCV, macro-F1) — Emosi ===")
    print(f"Mencoba {2*3*3*4} kombinasi × 5 fold ...")
    gs_emo = GridSearchCV(pipe, _PARAM_GRID, scoring="f1_macro", cv=5, n_jobs=-1)
    gs_emo.fit(df_train["clean"], df_train["Emotion"])
    best = gs_emo.best_params_
    print(f"  Best params : {best}")
    print(f"  CV macro-F1 : {gs_emo.best_score_:.4f}")

    tfidf_params = dict(
        ngram_range=best["tfidf__ngram_range"],
        min_df=best["tfidf__min_df"],
        max_features=best["tfidf__max_features"],
        sublinear_tf=True,
    )
    c_emo = best["clf__C"]

    # Vectorizer bersama dengan param terbaik.
    vectorizer = TfidfVectorizer(**tfidf_params)
    X_train = vectorizer.fit_transform(df_train["clean"])

    # Tuning C untuk Sentimen pada fitur yang sudah ditentukan.
    print("\n=== Tuning (GridSearchCV, macro-F1) — Sentimen (parameter C) ===")
    gs_sent = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"C": [0.1, 1, 3, 10]}, scoring="f1_macro", cv=5, n_jobs=-1,
    )
    gs_sent.fit(X_train, df_train["Sentiment"])
    c_sent = gs_sent.best_params_["C"]
    print(f"  Best C      : {c_sent}")
    print(f"  CV macro-F1 : {gs_sent.best_score_:.4f}")

    model_sent = train_final(X_train, df_train["Sentiment"], C=c_sent)
    model_emo = train_final(X_train, df_train["Emotion"], C=c_emo)

    info = {
        "tuned": True,
        "method": "GridSearchCV (cv=5, scoring=f1_macro)",
        "tfidf_params": {k: str(v) for k, v in tfidf_params.items()},
        "C": {"Sentimen": c_sent, "Emosi": c_emo},
        "cv_f1_macro": {
            "Sentimen": round(float(gs_sent.best_score_), 4),
            "Emosi": round(float(gs_emo.best_score_), 4),
        },
    }
    return vectorizer, model_sent, model_emo, info


def main(tune: bool = False) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    df_train, df_test = load_and_split()
    print(f"Data latih: {len(df_train)} | Data uji: {len(df_test)}")

    tuning_info = {"tuned": False}
    models = {}  # task -> (model, fname)

    if tune:
        # Jalur tuning: GridSearchCV menentukan hyperparameter terbaik.
        vectorizer, model_sent, model_emo, tuning_info = tune_models(df_train)
        X_test = vectorizer.transform(df_test["clean"])
        models = {
            "Sentimen": (model_sent, "Sentiment", "classic_sentiment.joblib"),
            "Emosi": (model_emo, "Emotion", "classic_emotion.joblib"),
        }
        vec_desc = ", ".join(f"{k}={v}" for k, v in tuning_info["tfidf_params"].items())
    else:
        # Jalur default: parameter manual + perbandingan model.
        vectorizer = build_vectorizer()
        X_train = vectorizer.fit_transform(df_train["clean"])
        for task, col, fname in [
            ("Sentimen", "Sentiment", "classic_sentiment.joblib"),
            ("Emosi", "Emotion", "classic_emotion.joblib"),
        ]:
            compare_models(X_train, df_train[col], task)
            models[task] = (train_final(X_train, df_train[col]), col, fname)
        X_test = vectorizer.transform(df_test["clean"])
        vec_desc = "TF-IDF (1,2)-gram, sublinear_tf, min_df=3"

    n_features = len(vectorizer.get_feature_names_out())
    print(f"\nDimensi fitur TF-IDF: {n_features} fitur")

    metadata = {
        "n_features": int(n_features),
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "vectorizer": vec_desc,
        "final_model": "LogisticRegression(class_weight='balanced')",
        "tuning": tuning_info,
        "tasks": {},
    }

    # Evaluasi & simpan tiap model.
    for task, (model, col, fname) in models.items():
        test_acc = model.score(X_test, df_test[col])
        print(f"  -> Model final {task}: akurasi data uji = {test_acc:.4f} (C={model.C})")
        joblib.dump(model, MODELS_DIR / fname)
        metadata["tasks"][task] = {
            "label_col": col,
            "classes": list(model.classes_),
            "test_accuracy": round(float(test_acc), 4),
            "C": float(model.C),
            "model_file": fname,
        }

    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )

    print(f"\nSelesai ({'TUNED' if tune else 'default'}). Artefak tersimpan di {MODELS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latih model EmoSense-ID.")
    parser.add_argument(
        "--tune", action="store_true",
        help="Jalankan hyperparameter tuning (GridSearchCV) sebelum melatih model final.",
    )
    args = parser.parse_args()
    main(tune=args.tune)
