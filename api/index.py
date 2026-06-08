import json
import os
import traceback
from collections import Counter
from datetime import datetime
from threading import Lock

import httpx
from flask import Flask, jsonify, request

app = Flask(__name__)

# ── Config ──
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
    "How how_received should influence risk:\n"
    "- WhatsApp forward — slightly elevated risk\n"
    "- Unknown caller — HIGH risk signal\n"
    "- Merchant QR — generally lower risk IF merchant appears legitimate\n\n"
    "Return ONLY this JSON shape:\n"
    '{{"risk_level":"HIGH_RISK|MEDIUM_RISK|SAFE","confidence_score":0,'
    '"upi_id":"","merchant_name":"","amount":"","transaction_type":"",'
    '"reasons":["",""],"hindi_summary":"","suggested_action":""}}'
)

# ── In-memory store ──
_lock = Lock()
_analyses: dict[int, dict] = {}
_scam_reports: list[dict] = []
_next_id = 1


# ── Helpers ──
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

    verdict_en = data.get("verdict_en") or {
        "HIGH_RISK": f"HIGH RISK — This UPI transaction shows strong fraud indicators. {action_text}.",
        "MEDIUM_RISK": f"MEDIUM RISK — Some suspicious signals detected. {action_text}.",
        "SAFE": f"SAFE — No significant fraud indicators found. {action_text}.",
    }[risk_level]

    verdict_hi = data.get("verdict_hi") or {
        "HIGH_RISK": "उच्च जोखिम — इस UPI लेन-देन में धोखाधड़ी के मजबूत संकेत हैं। लेन-देन रोकें और NPCI को रिपोर्ट करें।",
        "MEDIUM_RISK": "मध्यम जोखिम — कुछ संदिग्ध संकेत मिले हैं। आगे बढ़ने से पहले विवरण सत्यापित करें।",
        "SAFE": "सुरक्षित — कोई महत्वपूर्ण धोखाधड़ी संकेत नहीं मिला। लेन-देन सुरक्षित प्रतीत होता है।",
    }[risk_level]

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
        "verdict_en": verdict_en,
        "verdict_hi": verdict_hi,
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
            return {"error": "NVIDIA NIM request failed.", "detail": detail, "_status": e.response.status_code}
        except httpx.TimeoutException:
            return {"error": "NVIDIA NIM request timed out.", "_status": 504}
        except httpx.RequestError as e:
            traceback.print_exc()
            return {"error": "Could not reach NVIDIA NIM.", "detail": str(e), "_status": 502}

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


def json_error(msg: str, status: int):
    return jsonify({"error": msg}), status


# ── Routes ──

@app.route("/api/stats")
def handle_stats():
    with _lock:
        total = len(_analyses)
        reports = len(_scam_reports)
        high_risk = sum(1 for a in _analyses.values() if a["result"]["risk_level"] == "HIGH_RISK")
        upi_ids = [a["result"]["upi_id"] for a in _analyses.values() if a["result"].get("upi_id")]
        merchant_names = [a["result"]["merchant_name"] for a in _analyses.values() if a["result"].get("merchant_name")]
        reported_upi = [r["upi_id"] for r in _scam_reports if r.get("upi_id")]

    return jsonify({
        "total_analyses": total,
        "total_scams_reported": reports,
        "high_risk_count": high_risk,
        "top_upi_patterns": [item for item, _ in Counter(upi_ids).most_common(5)],
        "top_merchant_names": [item for item, _ in Counter(merchant_names).most_common(5)],
        "top_reported_upi": [item for item, _ in Counter(reported_upi).most_common(5)],
        "helplines": helplines(),
    })


@app.route("/api/analyze", methods=["POST"])
async def handle_analyze():
    api_key = (os.environ.get("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        return json_error("NVIDIA_API_KEY is not set.", 500)

    body = request.get_json(silent=True)
    if not body:
        return json_error("Invalid JSON request.", 400)

    user_text = (body.get("text") or "").strip()
    image_data_url = (body.get("image") or "").strip()

    if not user_text and not image_data_url:
        return json_error("Add a screenshot, transaction message, or both.", 400)

    content = [
        {
            "type": "text",
            "text": f"Analyze the provided UPI evidence and return strict JSON only. User text/context: {user_text or 'No text provided.'}",
        }
    ]

    if image_data_url:
        if not image_data_url.startswith("data:image/"):
            return json_error("Image must be sent as a data URL.", 400)
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    result = await call_nvidia(messages)
    if "_status" in result:
        return jsonify(result), result.pop("_status")

    analysis_id = store_analysis(result, user_text=user_text)
    return jsonify({**result, "analysis_id": analysis_id})


@app.route("/api/reanalyze", methods=["POST"])
async def handle_reanalyze():
    api_key = (os.environ.get("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        return json_error("NVIDIA_API_KEY is not set.", 500)

    body = request.get_json(silent=True)
    if not body:
        return json_error("Invalid JSON request.", 400)

    analysis_id = body.get("analysis_id")
    how_received = (body.get("how_received") or "").strip()

    with _lock:
        if not analysis_id or analysis_id not in _analyses:
            return json_error("Analysis not found. Run an analysis first.", 404)
        prev = _analyses[analysis_id]

    if how_received not in ("WhatsApp forward", "Unknown caller", "Merchant QR"):
        return json_error("how_received must be one of: WhatsApp forward, Unknown caller, Merchant QR", 400)

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
    if "_status" in result:
        return jsonify(result), result.pop("_status")

    new_id = store_analysis(result, user_text=prev.get("user_text", ""), how_received=how_received)
    return jsonify({**result, "analysis_id": new_id, "reanalyzed_from": analysis_id})


@app.route("/api/report-scam", methods=["POST"])
def handle_report_scam():
    body = request.get_json(silent=True)
    if not body:
        return json_error("Invalid JSON request.", 400)

    analysis_id = body.get("analysis_id")

    with _lock:
        if not analysis_id or analysis_id not in _analyses:
            return json_error("Analysis not found.", 404)
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

    return jsonify({
        "status": "reported",
        "helplines": helplines(),
        "message": "Scam report logged. Contact helpline immediately.",
        "report": report,
    })


@app.errorhandler(Exception)
def handle_exception(e):
    traceback.print_exc()
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500
