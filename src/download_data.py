"""Unduh dataset PRDECT-ID ke folder data/.

PRDECT-ID: Indonesian Product Reviews Dataset for Emotions Classification Tasks
(Sutoyo dkk., Data in Brief, 2022). ~5.400 ulasan produk Tokopedia, berlabel
Sentiment (Positive/Negative) dan Emotion (Anger, Fear, Happy, Love, Sadness).

Sumber: https://github.com/rhiosutoyo/PRDECT-ID-Indonesian-Product-Reviews-Dataset

Jalankan: python src/download_data.py
"""

from pathlib import Path
from urllib.request import urlopen

# URL file CSV mentah di GitHub (branch main).
DATA_URL = (
    "https://raw.githubusercontent.com/rhiosutoyo/"
    "PRDECT-ID-Indonesian-Product-Reviews-Dataset/main/"
    "Dataset/PRDECT-ID%20Dataset.csv"
)

# Simpan ke data/prdect_id.csv relatif terhadap root proyek.
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "prdect_id.csv"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Mengunduh dataset dari:\n  {DATA_URL}")
    with urlopen(DATA_URL) as resp:  # noqa: S310 (URL tepercaya & statis)
        content = resp.read()
    OUTPUT_PATH.write_bytes(content)
    size_kb = len(content) / 1024
    print(f"Tersimpan ke {OUTPUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
