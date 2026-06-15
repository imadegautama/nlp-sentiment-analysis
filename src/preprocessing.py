"""Pra-pemrosesan teks Bahasa Indonesia untuk EmoSense-ID.

File ini adalah SATU-SATUNYA sumber logika pembersihan teks. Di-import oleh
skrip pelatihan (train.py) DAN oleh aplikasi Streamlit (app/streamlit_app.py),
sehingga teks dari pengguna dijamin melewati langkah yang IDENTIK dengan saat
model dilatih. Konsistensi ini adalah syarat eksplisit pada soal UAS (Tahap 2).

Tahapan clean_text():
  1. Case folding         -> semua huruf jadi kecil
  2. Hapus URL & mention  -> token tak bermakna untuk klasifikasi
  3. Hapus angka/simbol   -> sisakan huruf a-z dan spasi saja
  4. Tokenisasi sederhana -> pisah berdasarkan spasi
  5. Hapus stopword       -> kata umum tak informatif (pakai daftar Sastrawi),
                             TAPI kata negasi sengaja DIPERTAHANKAN (lihat bawah)
  6. Stemming (Sastrawi)  -> kembalikan kata ke bentuk dasar ("membelikan"->"beli")

Catatan penting (kata negasi):
  Daftar stopword bawaan Sastrawi memuat kata seperti "tidak", "bukan", "jangan".
  Untuk analisis sentimen, menghapusnya berbahaya: "tidak bagus" akan menyusut
  jadi "bagus" dan maknanya terbalik. Karena itu kata-kata negasi dikeluarkan
  dari daftar stopword agar tetap dipertahankan. Dipadukan dengan fitur bigram
  pada TF-IDF, pola seperti "tidak_bagus" tetap tertangkap model.
"""

import re
from functools import lru_cache

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- Singleton objek Sastrawi (dibuat sekali, dipakai ulang) ---------------
# Membuat stemmer cukup mahal, jadi diinisialisasi sekali di level modul.
_stemmer = StemmerFactory().create_stemmer()

# Kata negasi & penanda sentimen yang WAJIB dipertahankan (jangan dihapus).
NEGATION_WORDS = {
    "tidak", "tak", "tdk", "bukan", "jangan", "belum", "kurang",
    "gak", "ga", "nggak", "ngga", "enggak", "engga", "tanpa", "jgn",
}

# Daftar stopword final = stopword Sastrawi DIKURANGI kata negasi di atas.
_STOPWORDS = set(StopWordRemoverFactory().get_stop_words()) - NEGATION_WORDS

# Pola regex dikompilasi sekali untuk efisiensi.
_RE_URL = re.compile(r"http\S+|www\.\S+")
_RE_MENTION = re.compile(r"[@#]\w+")
_RE_NON_ALPHA = re.compile(r"[^a-z\s]")  # sisakan huruf kecil & spasi
_RE_MULTISPACE = re.compile(r"\s+")


@lru_cache(maxsize=100_000)
def _stem_word(word: str) -> str:
    """Stem satu kata dengan cache.

    Stemming Sastrawi relatif lambat; meng-cache per-kata membuat pemrosesan
    ribuan ulasan jauh lebih cepat karena kata yang sama tak distem berulang.
    """
    return _stemmer.stem(word)


def clean_text(text: str) -> str:
    """Bersihkan & normalkan satu string ulasan menjadi teks siap-fitur.

    Mengembalikan string kosong bila input bukan teks (mis. NaN dari pandas).
    """
    if not isinstance(text, str):
        return ""

    # 1. Case folding
    text = text.lower()

    # 2. Hapus URL dan mention/hashtag
    text = _RE_URL.sub(" ", text)
    text = _RE_MENTION.sub(" ", text)

    # 3. Hapus angka, tanda baca, emoji -> hanya huruf & spasi yang tersisa
    text = _RE_NON_ALPHA.sub(" ", text)

    # 4. Tokenisasi sederhana berbasis spasi
    tokens = _RE_MULTISPACE.sub(" ", text).strip().split()

    # 5. Hapus stopword (negasi dipertahankan) & token sangat pendek
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    # 6. Stemming ke bentuk dasar
    tokens = [_stem_word(t) for t in tokens]

    return " ".join(tokens)


if __name__ == "__main__":
    # Demo cepat untuk verifikasi manual.
    contoh = [
        "Barangnya BAGUS banget!!! Pengiriman super cepat 😍 http://toko.id/x",
        "Produk ini TIDAK bagus, mengecewakan sekali :(",
        "Lumayan lah, sesuai harga. Tidak istimewa tapi tidak jelek juga.",
    ]
    for c in contoh:
        print(f"ASLI  : {c}")
        print(f"BERSIH: {clean_text(c)}")
        print("-" * 60)
