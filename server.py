#!/usr/bin/env python3
import json
import os
import socket
import sys
import threading
import traceback
from collections import Counter
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "meta/llama-3.2-90b-vision-instruct"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
NVIDIA_TIMEOUT_SECONDS = int(os.environ.get("NVIDIA_TIMEOUT_SECONDS", "180"))
NVIDIA_MAX_TOKENS = int(os.environ.get("NVIDIA_MAX_TOKENS", "800"))

FALLBACK_RESULT = {
    "risk_level": "MEDIUM_RISK",
    "confidence_score": 40,
    "upi_id": "",
    "merchant_name": "",
    "amount": "",
    "transaction_type": "unknown",
    "reasons": ["Could not parse the image. Try a clearer screenshot.", "If this persists, try uploading only the UPI section."],
    "hindi_summary": "छवि स्पष्ट नहीं है। कृपया सिर्फ UPI स्क्रीनशॉट अपलोड करें।",
    "suggested_action": "Try uploading a clearer screenshot",
    "verdict_en": "MEDIUM RISK — Could not fully analyze the image. Try a clearer screenshot.",
    "verdict_hi": "मध्यम जोखिम — छवि का विश्लेषण नहीं हो सका। स्पष्ट स्क्रीनशॉट अपलोड करें।",
}

SYSTEM_PROMPT = """You are an AI-powered Financial Fraud Detection Engine specialized in analyzing UPI payment screenshots, transaction messages, and user descriptions.

Your job is to detect whether a transaction is SAFE, MEDIUM RISK, or HIGH RISK based on visual and textual evidence.

You will receive inputs in two possible forms:
1. IMAGE (UPI screenshot, QR payment screen, transaction proof)
2. TEXT (user message describing transaction, payment request, or suspicious activity)
3. BOTH IMAGE + TEXT

You MUST analyze all available inputs jointly before making a decision.

OBJECTIVE
Extract financial and transactional details and evaluate fraud risk based on:
- UPI ID patterns
- Merchant legitimacy
- Payment context
- Urgency or manipulation language
- Mismatch between sender/receiver
- Suspicious QR or payment instructions
- Known scam indicators
- Social engineering patterns

REQUIRED EXTRACTION FIELDS
Always extract the following if present:
- upi_id (string or null)
- merchant_name (string or null)
- amount (string or number or null)
- transaction_type (e.g., "request", "payment", "QR scan", "unknown")
- sender_context (if inferable)
- receiver_context (if inferable)

RISK CLASSIFICATION RULES
Classify into one of:

HIGH_RISK
- Fake or suspicious merchant
- Unknown or randomly generated UPI IDs
- Urgent payment pressure or scam language
- QR/payment mismatch signals
- Impersonation or phishing indicators

MEDIUM_RISK
- Partially unclear merchant
- Missing verification signals
- Ambiguous transaction context
- Requires user confirmation

SAFE
- Verified merchant patterns
- Normal transaction behavior
- Consistent payment context
- No scam indicators

CONFIDENCE SCORE
Return a confidence score from 0 to 100 based on clarity of image/text, number of extracted fields, and strength of fraud signals.

REASONING REQUIREMENTS
Provide 2-4 clear bullet reasons. Each reason must be specific, reference extracted evidence, and avoid generic statements.

MULTILINGUAL OUTPUT
Also provide hindi_summary in 1-2 lines explaining risk in Hindi.

SUGGESTED ACTION LOGIC
- HIGH_RISK -> "Block transaction and report to NPCI"
- MEDIUM_RISK -> "Verify details before proceeding"
- SAFE -> "Transaction appears safe"

IMPORTANT BEHAVIOR RULES
- If image is unclear, rely more on text
- If text is missing, rely on image
- If both are present, cross-validate them
- Never guess UPI ID or amount if not visible
- Never hallucinate merchant names
- Be conservative in fraud detection (avoid false positives unless strong signals exist)

FINAL RULE
Output must always be valid JSON and must not include explanations outside JSON.

Return ONLY this JSON shape:
{
  "risk_level": "HIGH_RISK | MEDIUM_RISK | SAFE",
  "confidence_score": 0,
  "upi_id": "",
  "merchant_name": "",
  "amount": "",
  "transaction_type": "",
  "reasons": ["", ""],
  "hindi_summary": "",
  "suggested_action": ""
}"""

FOLLOWUP_PROMPT_TPL = """You previously analyzed a UPI transaction and returned a risk verdict. Now the user has provided additional context about HOW they received this payment request.

Re-evaluate the transaction risk considering this new "how_received" context. Your previous analysis may need to be updated.

The user received this via: {how_received}

How "how_received" should influence risk:
- "WhatsApp forward" — slightly elevated risk; forwards from unknown numbers are common scam vectors
- "Unknown caller" — HIGH risk signal; unsolicited calls asking for payment are a top fraud pattern
- "Merchant QR" — generally lower risk IF the merchant appears legitimate; verify merchant name match

Use your previous extracted data AND this new context to produce an updated verdict.

Return ONLY this JSON shape (same format as before):
{{
  "risk_level": "HIGH_RISK | MEDIUM_RISK | SAFE",
  "confidence_score": 0,
  "upi_id": "",
  "merchant_name": "",
  "amount": "",
  "transaction_type": "",
  "reasons": ["", ""],
  "hindi_summary": "",
  "suggested_action": ""
}}"""


# ── In-memory store ──────────────────────────────────────────
_store_lock = threading.Lock()
_analyses = {}
_scam_reports = []
_next_id = 1


def normalize_result(raw_text):
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(raw_text[start : end + 1])

    risk_level = data.get("risk_level") or "MEDIUM_RISK"
    if risk_level == "MEDIUM RISK":
        risk_level = "MEDIUM_RISK"
    if risk_level not in {"HIGH_RISK", "MEDIUM_RISK", "SAFE"}:
        risk_level = "MEDIUM_RISK"

    try:
        confidence = int(data.get("confidence_score", 0))
    except (TypeError, ValueError):
        confidence = 0

    action_by_risk = {
        "HIGH_RISK": "Block transaction and report to NPCI",
        "MEDIUM_RISK": "Verify details before proceeding",
        "SAFE": "Transaction appears safe",
    }

    reasons = data.get("reasons") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]

    action_text = data.get("suggested_action") or action_by_risk[risk_level]

    verdict_en_map = {
        "HIGH_RISK": f"HIGH RISK — This UPI transaction shows strong fraud indicators. {action_text}.",
        "MEDIUM_RISK": f"MEDIUM RISK — Some suspicious signals detected. {action_text}.",
        "SAFE": f"SAFE — No significant fraud indicators found. {action_text}.",
    }
    verdict_hi_map = {
        "HIGH_RISK": "उच्च जोखिम — इस UPI लेन-देन में धोखाधड़ी के मजबूत संकेत हैं। लेन-देन रोकें और NPCI को रिपोर्ट करें।",
        "MEDIUM_RISK": "मध्यम जोखिम — कुछ संदिग्ध संकेत मिले हैं। आगे बढ़ने से पहले विवरण सत्यापित करें।",
        "SAFE": "सुरक्षित — कोई महत्वपूर्ण धोखाधड़ी संकेत नहीं मिला। लेन-देन सुरक्षित प्रतीत होता है।",
    }

    return {
        "risk_level": risk_level,
        "confidence_score": max(0, min(100, confidence)),
        "upi_id": data.get("upi_id") or "",
        "merchant_name": data.get("merchant_name") or "",
        "amount": str(data.get("amount") or ""),
        "transaction_type": data.get("transaction_type") or "unknown",
        "reasons": [str(reason) for reason in reasons[:4]],
        "hindi_summary": data.get("hindi_summary") or "",
        "suggested_action": action_text,
        "verdict_en": data.get("verdict_en") or verdict_en_map[risk_level],
        "verdict_hi": data.get("verdict_hi") or verdict_hi_map[risk_level],
    }


def call_nvidia(messages):
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    payload = {
        "model": NVIDIA_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": NVIDIA_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    request = Request(
        NVIDIA_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=NVIDIA_TIMEOUT_SECONDS) as response:
        raw = response.read().decode("utf-8")
        api_response = json.loads(raw)
    try:
        raw_content = api_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        print("WARN: Unexpected NVIDIA response structure:", raw[:500], flush=True)

        if not raw_content or not raw_content.strip():
            print("WARN: NVIDIA returned empty/whitespace content. Full response:", raw[:500], flush=True)
        return FALLBACK_RESULT
    try:
        return normalize_result(raw_content)
    except json.JSONDecodeError:
        print("WARN: NVIDIA returned non-JSON content:", raw_content[:500], flush=True)
        return FALLBACK_RESULT


def store_analysis(result, user_text="", how_received=""):
    global _next_id
    with _store_lock:
        analysis_id = _next_id
        _next_id += 1
        entry = {
            "id": analysis_id,
            "timestamp": datetime.utcnow().isoformat(),
            "result": result,
            "user_text": user_text,
            "how_received": how_received,
        }
        _analyses[analysis_id] = entry
        return analysis_id


class FraudHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path == "/api/stats":
            self.handle_stats()
        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/analyze":
            self.handle_analyze()
        elif self.path == "/api/reanalyze":
            self.handle_reanalyze()
        elif self.path == "/api/report-scam":
            self.handle_report_scam()
        else:
            self.send_error(404, "Not found")

    # ── Helpers ──────────────────────────────────────────────

    def read_body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None, (400, {"error": "Invalid request size."})
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            return None, (413, {"error": "Request is empty or too large."})
        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return None, (400, {"error": "Invalid JSON request."})
        return body, None

    def check_api_key(self):
        api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if not api_key:
            return False, (500, {"error": "NVIDIA_API_KEY is not set."})
        return True, None

    def write_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ── POST /api/analyze ────────────────────────────────────

    def handle_analyze(self):
        ok, err = self.check_api_key()
        if not ok:
            self.write_json(*err)
            return

        body, err = self.read_body()
        if err:
            self.write_json(*err)
            return

        user_text = (body.get("text") or "").strip()
        image_data_url = (body.get("image") or "").strip()
        if not user_text and not image_data_url:
            self.write_json(400, {"error": "Add a screenshot, transaction message, or both."})
            return

        content = [
            {
                "type": "text",
                "text": (
                    "Analyze the provided UPI evidence and return strict JSON only. "
                    f"User text/context: {user_text or 'No text provided.'}"
                ),
            }
        ]

        if image_data_url:
            if not image_data_url.startswith("data:image/"):
                self.write_json(400, {"error": "Image must be sent as a data URL."})
                return
            content.append({"type": "image_url", "image_url": {"url": image_data_url}})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        try:
            result = call_nvidia(messages)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            print("NVIDIA HTTPError:", detail[:500])
            self.write_json(error.code, {"error": "NVIDIA NIM request failed.", "detail": detail})
            return
        except URLError as error:
            traceback.print_exc()
            self.write_json(502, {"error": "Could not reach NVIDIA NIM.", "detail": str(error.reason)})
            return
        except (TimeoutError, socket.timeout):
            self.write_json(504, {"error": f"NVIDIA NIM request timed out after {NVIDIA_TIMEOUT_SECONDS} seconds."})
            return
        except Exception as error:
            traceback.print_exc()
            self.write_json(502, {"error": "Unexpected NVIDIA NIM response.", "detail": str(error)})
            return

        analysis_id = store_analysis(result, user_text=user_text)

        self.write_json(200, {**result, "analysis_id": analysis_id})

    # ── POST /api/reanalyze ──────────────────────────────────

    def handle_reanalyze(self):
        ok, err = self.check_api_key()
        if not ok:
            self.write_json(*err)
            return

        body, err = self.read_body()
        if err:
            self.write_json(*err)
            return

        analysis_id = body.get("analysis_id")
        how_received = (body.get("how_received") or "").strip()

        if not analysis_id or analysis_id not in _analyses:
            self.write_json(404, {"error": "Analysis not found. Run an analysis first."})
            return

        if how_received not in ("WhatsApp forward", "Unknown caller", "Merchant QR"):
            self.write_json(400, {"error": "how_received must be one of: WhatsApp forward, Unknown caller, Merchant QR"})
            return

        prev = _analyses[analysis_id]
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

        try:
            result = call_nvidia(messages)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            print("NVIDIA HTTPError (reanalyze):", detail[:500])
            self.write_json(error.code, {"error": "NVIDIA NIM request failed.", "detail": detail})
            return
        except URLError as error:
            traceback.print_exc()
            self.write_json(502, {"error": "Could not reach NVIDIA NIM.", "detail": str(error.reason)})
            return
        except (TimeoutError, socket.timeout):
            self.write_json(504, {"error": f"NVIDIA NIM request timed out after {NVIDIA_TIMEOUT_SECONDS} seconds."})
            return
        except Exception as error:
            traceback.print_exc()
            self.write_json(502, {"error": "Unexpected NVIDIA NIM response.", "detail": str(error)})
            return

        new_id = store_analysis(result, user_text=prev.get("user_text", ""), how_received=how_received)

        self.write_json(200, {**result, "analysis_id": new_id, "reanalyzed_from": analysis_id})

    # ── POST /api/report-scam ────────────────────────────────

    def handle_report_scam(self):
        body, err = self.read_body()
        if err:
            self.write_json(*err)
            return

        analysis_id = body.get("analysis_id")
        if not analysis_id or analysis_id not in _analyses:
            self.write_json(404, {"error": "Analysis not found."})
            return

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

        with _store_lock:
            _scam_reports.append(report)

        self.write_json(200, {
            "status": "reported",
            "helplines": [
                {"name": "National Cyber Crime Helpline", "number": "1930", "managed_by": "Ministry of Home Affairs"},
                {"name": "NPCI UPI Fraud Helpline", "number": "1800-120-1740", "managed_by": "National Payments Corporation of India"},
            ],
            "message": "Scam report logged. Contact helpline immediately.",
            "report": report,
        })

    # ── GET /api/stats ───────────────────────────────────────

    def handle_stats(self):
        with _store_lock:
            total_analyses = len(_analyses)
            total_reports = len(_scam_reports)
            high_risk_count = sum(1 for a in _analyses.values() if a["result"]["risk_level"] == "HIGH_RISK")

            upi_ids = [a["result"]["upi_id"] for a in _analyses.values() if a["result"].get("upi_id")]
            merchant_names = [a["result"]["merchant_name"] for a in _analyses.values() if a["result"].get("merchant_name")]

            top_upi = [item for item, _ in Counter(upi_ids).most_common(5)]
            top_merchants = [item for item, _ in Counter(merchant_names).most_common(5)]

            reported_upi = [r["upi_id"] for r in _scam_reports if r.get("upi_id")]
            top_reported_upi = [item for item, _ in Counter(reported_upi).most_common(5)]

        self.write_json(200, {
            "total_analyses": total_analyses,
            "total_scams_reported": total_reports,
            "high_risk_count": high_risk_count,
            "top_upi_patterns": top_upi,
            "top_merchant_names": top_merchants,
            "top_reported_upi": top_reported_upi,
            "helplines": [
                {"name": "National Cyber Crime Helpline", "number": "1930", "managed_by": "Ministry of Home Affairs"},
                {"name": "NPCI UPI Fraud Helpline", "number": "1800-120-1740", "managed_by": "National Payments Corporation of India"},
            ],
        })


def main():
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), FraudHandler)
    print(f"UPI Guardian running at http://127.0.0.1:{port}")
    print(f"Endpoints: POST /api/analyze, POST /api/reanalyze, POST /api/report-scam, GET /api/stats")
    server.serve_forever()


if __name__ == "__main__":
    main()
