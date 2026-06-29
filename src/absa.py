"""Aspect-Based Sentiment Analysis (ABSA) ringan untuk EmoSense-ID Pro.

Membangun ABSA DI ATAS model sentimen TF-IDF yang sudah dilatih — tanpa data
beranotasi aspek dan tanpa LLM. Idenya sederhana dan sepenuhnya dapat dijelaskan:

  1. Pecah ulasan menjadi klausa (dipotong di kata sambung kontras seperti
     'tapi', 'namun', dan tanda baca).
  2. Untuk tiap klausa, deteksi aspek yang disinggung memakai leksikon kata kunci
     (pengiriman, kualitas, harga, pelayanan, kemasan).
  3. Jalankan model sentimen terlatih pada klausa tersebut → sentimen per aspek.

Dengan begitu kalimat seperti "barangnya bagus tapi pengirimannya lama" bisa
menghasilkan: Kualitas = Positif, Pengiriman = Negatif. Semua langkah memakai
komponen yang sama dengan training (clean_text + vectorizer + model).
"""

import re

from preprocessing import clean_text

# Leksikon aspek: kata kunci (termasuk bentuk berimbuhan) penanda tiap aspek.
ASPECT_LEXICON = {
    "Pengiriman": [
        "kirim", "pengiriman", "dikirim", "sampai", "sampe", "kurir", "paket",
        "datang", "ongkir", "resi", "jne", "jnt", "sicepat", "gosend", "antar",
    ],
    "Kualitas Produk": [
        "bagus", "jelek", "rusak", "awet", "kualitas", "barang", "produk",
        "palsu", "ori", "original", "cacat", "mulus", "kokoh", "bahan", "kuat",
        "berfungsi", "sesuai", "mantap", "berkualitas",
    ],
    "Harga": [
        "harga", "mahal", "murah", "worth", "terjangkau", "sebanding", "diskon",
        "promo", "hemat",
    ],
    "Pelayanan": [
        "respon", "ramah", "admin", "penjual", "seller", "pelayanan", "layanan",
        "balas", "jutek", "komunikasi", "amanah", "fast", "cs", "sopan", "judes",
    ],
    "Kemasan": [
        "packing", "kemasan", "bubble", "wrap", "rapi", "aman", "kardus",
        "bungkus", "segel", "dus",
    ],
}

# Emoji pendamping tiap aspek (untuk tampilan).
ASPECT_EMOJI = {
    "Pengiriman": "🚚",
    "Kualitas Produk": "📦",
    "Harga": "💰",
    "Pelayanan": "🙋",
    "Kemasan": "🎁",
}

# Pisahkan klausa di kata sambung kontras/penghubung dan tanda baca.
# Menyertakan 'dan/serta/juga' penting agar dua aspek berbeda yang digabung
# (mis. "pengiriman cepat dan barangnya jelek") dipisah & dinilai sendiri-sendiri.
_CLAUSE_SPLIT = re.compile(
    r"\s+(?:tapi|tetapi|namun|sedangkan|walaupun|walau|meskipun|meski|"
    r"cuma|hanya|sayangnya|tp|kecuali|dan|serta|juga|lalu|kemudian)\s+"
    r"|[.,;!?\n]+"
)


def split_clauses(text: str) -> list[str]:
    """Pecah ulasan menjadi klausa-klausa pendek."""
    parts = _CLAUSE_SPLIT.split(text.lower())
    return [p.strip() for p in parts if len(p.strip()) >= 3]


def detect_aspects(clause: str) -> list[str]:
    """Aspek apa saja yang disinggung sebuah klausa (berdasarkan leksikon)."""
    # Cocokkan pada teks mentah + teks ter-stem agar tahan terhadap imbuhan.
    haystack = clause + " " + clean_text(clause)
    found = []
    for aspect, keywords in ASPECT_LEXICON.items():
        if any(kw in haystack for kw in keywords):
            found.append(aspect)
    return found


def analyze_aspects(text: str, vectorizer, sentiment_model) -> dict:
    """Kembalikan sentimen per aspek untuk sebuah ulasan.

    Hasil: {aspek: {"sentiment": str, "confidence": float, "clause": str}}.
    Tiap aspek mengambil klausa dengan keyakinan tertinggi bila disinggung
    di beberapa klausa.
    """
    results: dict = {}
    for clause in split_clauses(text):
        aspects = detect_aspects(clause)
        if not aspects:
            continue
        cleaned = clean_text(clause)
        if not cleaned:
            continue
        x = vectorizer.transform([cleaned])
        pred = sentiment_model.predict(x)[0]
        conf = float(sentiment_model.predict_proba(x)[0].max())
        for aspect in aspects:
            prev = results.get(aspect)
            if prev is None or conf > prev["confidence"]:
                results[aspect] = {
                    "sentiment": pred,
                    "confidence": conf,
                    "clause": clause,
                }
    return results


if __name__ == "__main__":
    # Demo: muat artefak lalu jalankan ABSA pada beberapa contoh.
    from pathlib import Path

    import joblib

    models_dir = Path(__file__).resolve().parent.parent / "models"
    vec = joblib.load(models_dir / "tfidf_vectorizer.joblib")
    sent = joblib.load(models_dir / "classic_sentiment.joblib")

    contoh = [
        "Barangnya bagus dan awet, tapi pengirimannya lama banget dan kurirnya jutek",
        "Harga murah, kualitas oke, packing rapi dan aman. Pelayanan ramah!",
        "Kecewa, barang rusak dan tidak sesuai. Untung pengiriman cepat sih.",
        "Pengiriman cepat banget dan kurir ramah, tapi barangnya jelek dan harga kemahalan",
    ]
    for c in contoh:
        print(f"\nULASAN: {c}")
        hasil = analyze_aspects(c, vec, sent)
        if not hasil:
            print("  (tidak ada aspek terdeteksi)")
        for aspek, info in hasil.items():
            emoji = ASPECT_EMOJI.get(aspek, "")
            print(
                f"  {emoji} {aspek:<16} → {info['sentiment']:<8} "
                f"({info['confidence']:.0%})  «{info['clause']}»"
            )
