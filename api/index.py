import json
import os
import traceback
from collections import Counter
from datetime import datetime
from http import HTTPStatus
from threading import Lock

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="UPI Guardian")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ──────────────────────────────────────────────────
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.2-90b-vision-instruct")
NVIDIA_TIMEOUT = int(os.environ.get("NVIDIA_TIMEOUT_SECONDS", "240"))
NVIDIA_MAX_TOKENS = int(os.environ.get("NVIDIA_MAX_TOKENS", "800"))

FALLBACK_RESULT = {
    "risk_level": "MEDIUM_RISK",
    "confidence_score": 40,
    "upi_id": "",
    "merchant_name": "",
    "amount": "",
    "transaction_type": "unknown",
    "reasons": [
        "Could not parse the image. Try a clearer screenshot.",
        "If this persists, try uploading only the UPI section.",
    ],
    "hindi_summary": "छवि स्पष्ट नहीं है। कृपया सिर्फ UPI स्क्रीनशॉट अपलोड करें।",
    "suggested_action": "Try uploading a clearer screenshot",
    "verdict_en": "MEDIUM RISK — Could not fully analyze the image. Try a clearer screenshot.",
    "verdict_hi": "मध्यम जोखिम — छवि का विश्लेषण नहीं हो सका। स्पष्ट स्क्रीनशॉट अपलोड करें।",
}

SYSTEM_PROMPT = (
    "You are an AI-powered Financial Fraud Detection Engine specialized in analyzing UPI "
    "payment screenshots, transaction messages, and user descriptions.\n\n"
    "Your job is to detect whether a transaction is SAFE, MEDIUM RISK, or HIGH RISK based "
    "on visual and textual evidence.\n\n"
    "You will receive inputs in two possible forms:\n"
    "1. IMAGE (UPI screenshot, QR payment screen, transaction proof)\n"
    "2. TEXT (user message describing transaction, payment request, or suspicious activity)\n"
    "3. BOTH IMAGE + TEXT\n\n"
    "You MUST analyze all available inputs jointly before making a decision.\n\n"
    "OBJECTIVE\n"
    "Extract financial and transactional details and evaluate fraud risk based on:\n"
    "- UPI ID patterns\n"
    "- Merchant legitimacy\n"
    "- Payment context\n"
    "- Urgency or manipulation language\n"
    "- Mismatch between sender/receiver\n"
    "- Suspicious QR or payment instructions\n"
    "- Known scam indicators\n"
    "- Social engineering patterns\n\n"
    "REQUIRED EXTRACTION FIELDS\n"
    "Always extract the following if present:\n"
    "- upi_id (string or null)\n"
    "- merchant_name (string or null)\n"
    "- amount (string or number or null)\n"
    "- transaction_type (e.g., request, payment, QR scan, unknown)\n\n"
    "RISK CLASSIFICATION RULES\n"
    "HIGH_RISK — Fake merchant, random UPI IDs, urgent pressure, impersonation.\n"
    "MEDIUM_RISK — Partially unclear merchant, missing verification, ambiguous context.\n"
    "SAFE — Verified merchant, normal behavior, consistent context.\n\n"
    "CONFIDENCE SCORE\n"
    "Return 0-100 based on clarity and strength of signals.\n\n"
    "REASONING REQUIREMENTS\n"
    "Provide 2-4 specific bullet reasons referencing extracted evidence.\n\n"
    "MULTILINGUAL OUTPUT\n"
    "Also provide hindi_summary in 1-2 lines explaining risk in Hindi.\n\n"
    "SUGGESTED ACTION LOGIC\n"
    "- HIGH_RISK -> Block transaction and report to NPCI\n"
    "- MEDIUM_RISK -> Verify details before proceeding\n"
    "- SAFE -> Transaction appears safe\n\n"
    "FINAL RULE\n"
    "Output must always be valid JSON and must not include explanations outside JSON.\n\n"
    "Return ONLY this JSON shape:\n"
    '{"risk_level":"HIGH_RISK|MEDIUM_RISK|SAFE","confidence_score":0,'
    '"upi_id":"","merchant_name":"","amount":"","transaction_type":"",'
    '"reasons":["",""],"hindi_summary":"","suggested_action":""}'
)

FOLLOWUP_PROMPT_TPL = (
    "You previously analyzed a UPI transaction and returned a risk verdict.\n"
    "Now the user has provided additional context about HOW they received this payment request.\n\n"
    "Re-evaluate the transaction risk considering this new how_received context.\n"
    "The user received this via: {how_received}\n\n"
    'How how_received should influence risk:\n'
    '- WhatsApp forward — slightly elevated risk\n'
    '- Unknown caller — HIGH risk signal\n'
    '- Merchant QR — generally lower risk IF merchant appears legitimate\n\n'
    "Return ONLY this JSON shape:\n"
    '{{"risk_level":"HIGH_RISK|MEDIUM_RISK|SAFE","confidence_score":0,'
    '"upi_id":"","merchant_name":"","amount":"","transaction_type":"",'
    '"reasons":["",""],"hindi_summary":"","suggested_action":""}}'
)

# ── In-memory store ──────────────────────────────────────────
_lock = Lock()
_analyses: dict[int, dict] = {}
_scam_reports: list[dict] = []
_next_id = 1


# ── Helpers ──────────────────────────────────────────────────
def normalize_result(raw_text: str) -> dict:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(raw_text[start : end + 1])

    risk_level = (data.get("risk_level") or "MEDIUM_RISK").replace("MEDIUM RISK", "MEDIUM_RISK")
    if risk_level not in {"HIGH_RISK", "MEDIUM_RISK", "SAFE"}:
        risk_level = "MEDIUM_RISK"

    try:
        confidence = int(data.get("confidence_score", 0))
    except (TypeError, ValueError):
        confidence = 0

    action_map = {
        "HIGH_RISK": "Block transaction and report to NPCI",
        "MEDIUM_RISK": "Verify details before proceeding",
        "SAFE": "Transaction appears safe",
    }
    action_text = data.get("suggested_action") or action_map[risk_level]
    reasons = data.get("reasons") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]

    return {
        "risk_level": risk_level,
        "confidence_score": max(0, min(100, confidence)),
        "upi_id": data.get("upi_id") or "",
        "merchant_name": data.get("merchant_name") or "",
        "amount": str(data.get("amount") or ""),
        "transaction_type": data.get("transaction_type") or "unknown",
        "reasons": [str(r) for r in reasons[:4]],
        "hindi_summary": data.get("hindi_summary") or "",
        "suggested_action": action_text,
        "verdict_en": data.get("verdict_en")
        or (
            f"HIGH RISK — This UPI transaction shows strong fraud indicators. {action_text}."
            if risk_level == "HIGH_RISK"
            else (
                f"MEDIUM RISK — Some suspicious signals detected. {action_text}."
                if risk_level == "MEDIUM_RISK"
                else f"SAFE — No significant fraud indicators found. {action_text}."
            )
        ),
        "verdict_hi": data.get("verdict_hi")
        or (
            "उच्च जोखिम — इस UPI लेन-देन में धोखाधड़ी के मजबूत संकेत हैं। लेन-देन रोकें और NPCI को रिपोर्ट करें।"
            if risk_level == "HIGH_RISK"
            else (
                "मध्यम जोखिम — कुछ संदिग्ध संकेत मिले हैं। आगे बढ़ने से पहले विवरण सत्यापित करें।"
                if risk_level == "MEDIUM_RISK"
                else "सुरक्षित — कोई महत्वपूर्ण धोखाधड़ी संकेत नहीं मिला। लेन-देन सुरक्षित प्रतीत होता है।"
            )
        ),
    }


async def call_nvidia(messages: list) -> dict:
    api_key = (os.environ.get("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        return FALLBACK_RESULT

    payload = {
        "model": NVIDIA_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": NVIDIA_MAX_TOKENS,
    }

    async with httpx.AsyncClient(timeout=NVIDIA_TIMEOUT) as client:
        try:
            resp = await client.post(
                NVIDIA_API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            raw = resp.text
            try:
                api_resp = resp.json()
            except json.JSONDecodeError:
                print("WARN: Non-JSON response from NVIDIA:", raw[:500], flush=True)
                return FALLBACK_RESULT
        except httpx.HTTPStatusError as e:
            detail = e.response.text[:500]
            print("NVIDIA HTTPError:", detail, flush=True)
            raise HTTPException(status_code=e.response.status_code, detail=detail)
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail=f"NVIDIA NIM request timed out after {NVIDIA_TIMEOUT} seconds.",
            )
        except httpx.RequestError as e:
            traceback.print_exc()
            raise HTTPException(status_code=502, detail=str(e))

    try:
        raw_content = api_resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        print("WARN: Unexpected NVIDIA response:", raw[:500], flush=True)
        return FALLBACK_RESULT

    if not raw_content or not raw_content.strip():
        print("WARN: Empty content from NVIDIA:", raw[:500], flush=True)
        return FALLBACK_RESULT

    try:
        return normalize_result(raw_content)
    except json.JSONDecodeError:
        print("WARN: Non-JSON from NVIDIA:", raw_content[:500], flush=True)
        return FALLBACK_RESULT


def store_analysis(result: dict, user_text: str = "", how_received: str = "") -> int:
    global _next_id
    with _lock:
        aid = _next_id
        _next_id += 1
        _analyses[aid] = {
            "id": aid,
            "timestamp": datetime.utcnow().isoformat(),
            "result": result,
            "user_text": user_text,
            "how_received": how_received,
        }
        return aid


def helplines() -> list[dict]:
    return [
        {"name": "National Cyber Crime Helpline", "number": "1930", "managed_by": "Ministry of Home Affairs"},
        {"name": "NPCI UPI Fraud Helpline", "number": "1800-120-1740", "managed_by": "National Payments Corporation of India"},
    ]


# ── Routes ───────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    with _lock:
        total = len(_analyses)
        reports = len(_scam_reports)
        high_risk = sum(1 for a in _analyses.values() if a["result"]["risk_level"] == "HIGH_RISK")
        upi_ids = [a["result"]["upi_id"] for a in _analyses.values() if a["result"].get("upi_id")]
        merchant_names = [a["result"]["merchant_name"] for a in _analyses.values() if a["result"].get("merchant_name")]
        reported_upi = [r["upi_id"] for r in _scam_reports if r.get("upi_id")]

    return {
        "total_analyses": total,
        "total_scams_reported": reports,
        "high_risk_count": high_risk,
        "top_upi_patterns": [item for item, _ in Counter(upi_ids).most_common(5)],
        "top_merchant_names": [item for item, _ in Counter(merchant_names).most_common(5)],
        "top_reported_upi": [item for item, _ in Counter(reported_upi).most_common(5)],
        "helplines": helplines(),
    }


@app.post("/api/analyze")
async def analyze(request: Request):
    api_key = (os.environ.get("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="NVIDIA_API_KEY is not set.")

    body = await request.json()
    user_text = (body.get("text") or "").strip()
    image_data_url = (body.get("image") or "").strip()

    if not user_text and not image_data_url:
        raise HTTPException(status_code=400, detail="Add a screenshot, transaction message, or both.")

    content = [
        {
            "type": "text",
            "text": f"Analyze the provided UPI evidence and return strict JSON only. User text/context: {user_text or 'No text provided.'}",
        }
    ]

    if image_data_url:
        if not image_data_url.startswith("data:image/"):
            raise HTTPException(status_code=400, detail="Image must be sent as a data URL.")
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    result = await call_nvidia(messages)
    analysis_id = store_analysis(result, user_text=user_text)
    return {**result, "analysis_id": analysis_id}


@app.post("/api/reanalyze")
async def reanalyze(request: Request):
    api_key = (os.environ.get("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="NVIDIA_API_KEY is not set.")

    body = await request.json()
    analysis_id = body.get("analysis_id")
    how_received = (body.get("how_received") or "").strip()

    with _lock:
        if not analysis_id or analysis_id not in _analyses:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        prev = _analyses[analysis_id]

    if how_received not in ("WhatsApp forward", "Unknown caller", "Merchant QR"):
        raise HTTPException(
            status_code=400,
            detail="how_received must be one of: WhatsApp forward, Unknown caller, Merchant QR",
        )

    prev_result = prev["result"]
    prompt_text = FOLLOWUP_PROMPT_TPL.format(how_received=how_received)
    context = (
        f"Previous extraction — UPI ID: {prev_result['upi_id']}, "
        f"Merchant: {prev_result['merchant_name']}, "
        f"Amount: {prev_result['amount']}, "
        f"Previous verdict: {prev_result['risk_level']} (confidence: {prev_result['confidence_score']}%), "
        f"Previous reasons: {'; '.join(prev_result['reasons'])}"
    )

    messages = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": context},
    ]

    result = await call_nvidia(messages)
    new_id = store_analysis(result, user_text=prev.get("user_text", ""), how_received=how_received)
    return {**result, "analysis_id": new_id, "reanalyzed_from": analysis_id}


@app.post("/api/report-scam")
async def report_scam(request: Request):
    body = await request.json()
    analysis_id = body.get("analysis_id")

    with _lock:
        if not analysis_id or analysis_id not in _analyses:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        analysis = _analyses[analysis_id]

    result = analysis["result"]
    report = {
        "id": analysis_id,
        "timestamp": datetime.utcnow().isoformat(),
        "upi_id": result["upi_id"],
        "merchant_name": result["merchant_name"],
        "amount": result["amount"],
        "risk_level": result["risk_level"],
        "confidence_score": result["confidence_score"],
        "how_received": analysis.get("how_received", ""),
    }

    with _lock:
        _scam_reports.append(report)

    return {
        "status": "reported",
        "helplines": helplines(),
        "message": "Scam report logged. Contact helpline immediately.",
        "report": report,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )
