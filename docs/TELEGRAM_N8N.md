# Bot Telegram EmoSense-ID via n8n

Panduan menghubungkan **bot Telegram** ke REST API EmoSense-ID (`api.py`) memakai
**n8n**. Alur: user kirim teks ulasan → n8n → `POST /analyze` → format → balas.

```
Telegram ──► n8n (Telegram Trigger) ──► HTTP POST /analyze ──► Format ──► sendMessage ──► Telegram
```

## Prasyarat
- Python env API sudah jalan (lihat README). Model sudah dilatih (`models/` terisi).
- **n8n** (Docker atau `npx n8n`).
- **Token bot Telegram** dari [@BotFather](https://t.me/BotFather).

## 1. Buat bot Telegram
1. Chat **@BotFather** → `/newbot` → ikuti instruksi → salin **token** (mis. `123456:ABC-...`).

## 2. Jalankan REST API
Jalankan di mesin yang sama (native, bukan dalam container yang sama dengan n8n):
```bash
# tanpa AI
.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8800
# dengan AI (ringkasan + saran balasan)
OPENROUTER_API_KEY="sk-or-..." .venv/bin/uvicorn api:app --host 0.0.0.0 --port 8800
```
Cek: `curl http://localhost:8800/health` → `{"status":"ok",...}`.

## 3. Jalankan n8n + import workflow
- **Docker:**
  ```bash
  docker run -it --rm -p 5678:5678 -e WEBHOOK_URL=https://<subdomain>.ngrok-free.app \
    -v n8n_data:/home/node/.n8n n8nio/n8n
  ```
  > n8n di Docker menghubungi API di host lewat `http://host.docker.internal:8800`
  > (sudah dipakai di workflow). Jika n8n jalan **native** (`npx n8n`), ubah URL node
  > **Analisis** menjadi `http://localhost:8800/analyze`.
- Buka `http://localhost:5678` → menu **⋯ → Import from File** → pilih
  `n8n/emosense-bot.workflow.json`.

## 4. Set kredensial Telegram
Pada node **Telegram Trigger**, **Kirim Balasan**, dan **Kirim Sambutan**:
- Buka node → Credentials → **Create New** → tempel **token** bot → simpan.
- (Placeholder `REPLACE_ME` akan tergantikan otomatis setelah kredensial dipilih.)

## 5. Webhook (agar Telegram bisa menjangkau n8n)
Telegram perlu URL publik. Untuk demo lokal, gunakan salah satu:
- **n8n tunnel:** jalankan `npx n8n start --tunnel` (memberi URL publik sementara), atau
- **ngrok:** `ngrok http 5678`, lalu set `WEBHOOK_URL` ke URL ngrok saat menjalankan n8n.

## 6. Aktifkan & uji
1. Toggle workflow ke **Active** (kanan atas).
2. Buka bot Anda di Telegram → kirim `/start` → muncul sambutan.
3. Kirim contoh: *"pengiriman cepat tapi barangnya jelek dan harga kemahalan"*.
4. Bot membalas: Sentimen + Emosi + Analisis Aspek (+ Ringkasan & Saran balasan bila AI aktif).

## Produksi (opsional)
Bila API di-deploy (mis. Coolify, subdomain `nlp-api.contoh.com`), ubah URL node
**Analisis** ke `https://nlp-api.contoh.com/analyze`. n8n cloud/self-host yang sudah punya
domain tidak perlu tunnel.

## Troubleshooting
- **Bot tak membalas:** pastikan workflow Active, webhook/tunnel hidup, dan token benar.
- **HTTP node error (ECONNREFUSED):** API tidak terjangkau — cek port 8800 & `host.docker.internal`
  (Docker) vs `localhost` (native).
- **"Bad Request" saat sendMessage:** biasanya teks kosong — pastikan API mengembalikan field
  `reply`/JSON valid (cek `/health`).
