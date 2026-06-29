# 🚀 Deploy ke Coolify (GitHub Actions → Docker Hub → Coolify)

Pipeline CI/CD: **push ke `main`** → GitHub Actions **build 2 image** (API + Streamlit) →
push ke **Docker Hub** → **trigger redeploy** 2 resource Coolify lewat webhook.

```
git push main ─▶ GitHub Actions ─┬─ build Dockerfile.api ─▶ Docker Hub: emosense-api
                                 └─ build Dockerfile     ─▶ Docker Hub: emosense-streamlit
                                          │
                                          └─ curl webhook ─▶ Coolify pull image + redeploy
```

Target: 3 service di satu Coolify → **API** (8800), **Streamlit** (8501), **n8n** (5678).

> EmoSense pakai **scikit-learn** (bukan TensorFlow) → image jalan di x86_64 **maupun** arm64,
> bebas isu AVX. Anda bahkan bisa build & uji image ini di Mac lokal.

---

## Langkah 0 — Push repo ke GitHub
Coolify & Actions menarik dari GitHub, jadi repo harus ada di sana dulu:
```bash
git remote add origin https://github.com/<username>/emosense-id.git
git push -u origin main
```

## Langkah 1 — Docker Hub
1. Punya akun Docker Hub.
2. **Account Settings → Security → New Access Token** → simpan (jadi `DOCKERHUB_TOKEN`).
3. Repo image dibuat otomatis saat push pertama: `emosense-api` & `emosense-streamlit`.

## Langkah 2 — GitHub Secrets
Repo GitHub → **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Isi |
|--------|-----|
| `DOCKERHUB_USERNAME` | username Docker Hub |
| `DOCKERHUB_TOKEN` | access token Docker Hub (Langkah 1) |
| `COOLIFY_TOKEN` | API token Coolify (Langkah 3) |
| `COOLIFY_WEBHOOK_API` | deploy-webhook resource **API** (Langkah 4) |
| `COOLIFY_WEBHOOK_STREAMLIT` | deploy-webhook resource **Streamlit** (Langkah 4) |

## Langkah 3 — Token Coolify
Coolify → **Keys & Tokens → API tokens → Create** → simpan (jadi `COOLIFY_TOKEN`).

## Langkah 4 — Buat resource di Coolify

### a) App API
- **+ New → Resource → Docker Image** → image: `<DOCKERHUB_USERNAME>/emosense-api:latest`.
- **Ports Exposes:** `8800`.
- **Environment Variables:** `OPENROUTER_API_KEY=sk-or-...` (untuk fitur AI di `/analyze`).
- Set **Domain** (mis. `https://nlp-api.domain.com`) → Coolify urus HTTPS otomatis.
- Tab **Webhooks / Deploy** → salin **Deploy Webhook URL** → GitHub secret `COOLIFY_WEBHOOK_API`.

### b) App Streamlit
- **Docker Image** → `<DOCKERHUB_USERNAME>/emosense-streamlit:latest`.
- **Ports Exposes:** `8501`.
- **Environment Variables:** `OPENROUTER_API_KEY=sk-or-...` (opsional, untuk seksi Analisis AI).
- Set **Domain** (mis. `https://emosense.domain.com`).
- Salin **Deploy Webhook URL** → GitHub secret `COOLIFY_WEBHOOK_STREAMLIT`.
- Jika UI macet di balik proxy (jarang), tambahkan **Custom Start Command**:
  `streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false`

### c) n8n
- **+ New → Resource → Service → n8n** (one-click).
- Pasang **persistent volume** (agar workflow & kredensial tak hilang saat redeploy).
- Env: `WEBHOOK_URL=https://<domain-n8n>` (wajib, agar webhook Telegram HTTPS).
- Set **Domain** untuk n8n.

## Langkah 5 — Deploy pertama
```bash
git add Dockerfile Dockerfile.api .dockerignore .github/workflows/deploy.yml \
  docs/COOLIFY_DEPLOY.md requirements-base.txt requirements.txt requirements-api.txt
git commit -m "Tambah CI/CD deploy ke Coolify (API + Streamlit)"
git push origin main
```
- Tab **Actions** GitHub: job **build-and-push** (2 image) harus hijau → cek image di Docker Hub
  → job **deploy** memicu Coolify menarik image & redeploy.

## Langkah 6 — Hubungkan bot Telegram ke API di Coolify
Di **n8n (Coolify)**: import `n8n/emosense-bot.workflow.json`, set kredensial Telegram, lalu
**ubah URL** node **Analisis**:
- dari `http://host.docker.internal:8800/analyze`
- menjadi **domain publik API**: `https://nlp-api.domain.com/analyze`
  (alternatif lebih cepat/privat: hostname internal service API di jaringan Coolify).

Lalu **Activate** workflow → kirim teks ulasan ke bot.

---

## Verifikasi
```bash
curl https://<domain-api>/health
# → {"status":"ok","sentiment_classes":[...],"emotion_classes":[...],"ai_enabled":true}

curl -X POST https://<domain-api>/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"pengiriman cepat tapi barangnya jelek dan harga kemahalan"}'
# → JSON: sentiment + emotion + aspects[] (+ ai bila OPENROUTER_API_KEY diset)
```
- Buka `https://<domain-streamlit>` → analisis + seksi AI jalan.
- Kirim ulasan ke bot Telegram → Sentimen + Emosi + Aspek (+ Ringkasan & Saran balasan).

## Catatan
- `OPENROUTER_API_KEY` di-set sebagai **env Coolify per-resource**, BUKAN di-bake ke image.
- Tag image yang dipush: `latest` (main), `sha-<commit>`, nama branch, dan `vX.Y.Z` (saat tag).
- Update berikutnya: cukup `git push` → Actions build & trigger redeploy otomatis.
- Uji image lokal (opsional, Docker Desktop nyala):
  `docker build -f Dockerfile.api -t emosense-api . && docker run -p 8800:8800 emosense-api`.
