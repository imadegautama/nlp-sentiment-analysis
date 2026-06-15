"""Evaluasi model EmoSense-ID (Tahap 1 UAS NLP).

Memuat model & vectorizer hasil train.py, mengevaluasi pada data uji yang sama,
lalu menyimpan:
  - classification_report (precision/recall/F1 per kelas) ke terminal & file
  - confusion matrix sebagai gambar PNG di reports/
  - ringkasan metrik gabungan reports/metrics_summary.md

Jalankan SETELAH train.py:  python src/evaluate.py
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # backend non-interaktif (tak butuh layar)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

from train import ROOT, load_and_split

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TASKS = [
    ("Sentimen", "Sentiment", "classic_sentiment.joblib"),
    ("Emosi", "Emotion", "classic_emotion.joblib"),
]


def save_confusion_matrix(model, X_test, y_test, task: str) -> Path:
    """Simpan confusion matrix sebagai PNG, kembalikan path file."""
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, labels=model.classes_, cmap="Blues", ax=ax,
        xticks_rotation=45, colorbar=False,
    )
    ax.set_title(f"Confusion Matrix — {task}")
    fig.tight_layout()
    out = REPORTS_DIR / f"confusion_matrix_{task.lower()}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    _, df_test = load_and_split()
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    X_test = vectorizer.transform(df_test["clean"])

    summary_lines = ["# Ringkasan Evaluasi EmoSense-ID\n"]

    for task, col, fname in TASKS:
        y_test = df_test[col]
        model = joblib.load(MODELS_DIR / fname)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred, digits=4)

        print(f"\n{'=' * 60}\nTUGAS: {task}\n{'=' * 60}")
        print(f"Accuracy : {acc:.4f}")
        print(f"Macro-F1 : {f1m:.4f}\n")
        print(report)

        cm_path = save_confusion_matrix(model, X_test, y_test, task)
        print(f"Confusion matrix disimpan: {cm_path}")

        summary_lines += [
            f"## Tugas: {task}\n",
            f"- **Accuracy**: {acc:.4f}",
            f"- **Macro-F1**: {f1m:.4f}\n",
            "```",
            report.rstrip(),
            "```",
            f"\n![Confusion Matrix {task}]({cm_path.name})\n",
        ]

    (REPORTS_DIR / "metrics_summary.md").write_text("\n".join(summary_lines))
    print(f"\nRingkasan metrik disimpan: {REPORTS_DIR / 'metrics_summary.md'}")


if __name__ == "__main__":
    main()
