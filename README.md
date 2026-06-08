# UPI Guardian

AI-powered UPI fraud detection using NVIDIA NIM vision models. Upload a UPI screenshot or paste transaction text to get an instant risk verdict with explainability breakdown, bilingual output, and one-tap helpline reporting.

## Features

- **Live Analyzer** — Upload a UPI screenshot or enter transaction text; get risk level, confidence score, extracted fields, and bullet reasons
- **Explainability Panel** — Expandable "Why did we flag this?" section with signal cards (UPI ID pattern, merchant legitimacy, amount anomaly, urgency, etc.), each with a severity bar
- **Bilingual Verdict** — Results in English and Hindi with a Hindi summary
- **Follow-up Analysis** — Re-analyze with context (WhatsApp forward, Unknown caller, Merchant QR)
- **One-tap Helpline Reporting** — Report HIGH_RISK findings to NPCI; tap-to-call buttons for National Cyber Crime Helpline (1930) and NPCI UPI Fraud Helpline (1800-120-1740)
- **Live Dashboard** — Session log in localStorage, Chart.js risk history, server-side stats
- **Image Compression** — Client-side resize to 1200px/80% JPEG before upload
- **Share Results** — Download verdict as PNG via html2canvas

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla HTML/CSS/JS, Chart.js, html2canvas |
| Backend | Python (FastAPI / http.server) |
| AI Model | NVIDIA NIM — `meta/llama-3.2-90b-vision-instruct` |
| Deployment | Vercel (serverless) or local Python server |

## Getting Started

### Prerequisites

- Python 3.9+
- A [NVIDIA NIM API key](https://build.nvidia.com/explore/discover)

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/upi-guardian.git
cd upi-guardian

# 2. Set your NVIDIA API key
export NVIDIA_API_KEY="nvapi-..."

# 3. Run the local server
python3 server.py
```

Open **http://127.0.0.1:8000** in your browser.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_API_KEY` | — | **Required.** Your NVIDIA NIM API key |
| `NVIDIA_MODEL` | `meta/llama-3.2-90b-vision-instruct` | NVIDIA NIM model name |
| `NVIDIA_TIMEOUT_SECONDS` | `240` | Timeout for each NVIDIA API call |
| `NVIDIA_MAX_TOKENS` | `800` | Max response tokens from the model |
| `PORT` | `8000` | Local server port |

### Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)

1. Push this repo to GitHub
2. Import the project in Vercel
3. Add the `NVIDIA_API_KEY` environment variable in Vercel project settings
4. Deploy

The Vercel deployment uses `api/index.py` (FastAPI) for API routes and serves `index.html` as a static file.

```bash
# Or deploy via CLI
npm i -g vercel
vercel --prod
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/analyze` | Analyze text/image for fraud risk |
| `POST` | `/api/reanalyze` | Re-evaluate with "how received" context |
| `POST` | `/api/report-scam` | Log a scam report |
| `GET` | `/api/stats` | Get session statistics and helpline numbers |

### `/api/analyze` request body

```json
{
  "text": "Payment request from unknown number for ₹5000",
  "image": "data:image/jpeg;base64,..."
}
```

Both `text` and `image` are optional — at least one is required.

### `/api/analyze` response

```json
{
  "risk_level": "HIGH_RISK",
  "confidence_score": 80,
  "upi_id": "",
  "merchant_name": "",
  "amount": "5000",
  "transaction_type": "request",
  "reasons": ["Unknown sender", "Urgent payment pressure"],
  "hindi_summary": "अनजान नंबर से ₹5000 का भुगतान अनुरोध उच्च जोखिम वाला है",
  "suggested_action": "Block transaction and report to NPCI",
  "verdict_en": "HIGH RISK — This UPI transaction shows strong fraud indicators...",
  "verdict_hi": "उच्च जोखिम — इस UPI लेन-देन में धोखाधड़ी के मजबूत संकेत हैं...",
  "analysis_id": 1
}
```

## Project Structure

```
upi-guardian/
├── index.html          # Frontend (single-page app)
├── server.py           # Local Python server (http.server)
├── api/
│   ├── __init__.py
│   └── index.py        # Vercel serverless entry (FastAPI)
├── vercel.json         # Vercel deployment config
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT License
└── README.md
```

## License

MIT
