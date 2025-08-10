import os
import re
import requests
from bs4 import BeautifulSoup
from flask import jsonify
from typing import List
from concurrent.futures import ThreadPoolExecutor


_secret_token_re = re.compile(r"\b[0-9a-fA-F]{64}\b")


def is_plain_web_url(url: str) -> bool:
    """Check if URL points to a web page rather than a document file."""
    lowered = url.lower().split("?")[0]
    doc_exts = (".pdf", ".docx", ".doc", ".txt", ".csv", ".xlsx", ".xls", ".pptx", ".ppt")
    if lowered.endswith((".html", ".htm")):
        return True
    return not lowered.endswith(doc_exts)


def scrape_text_from_url(url: str) -> str:
    """Download and extract text content from a web page."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "html" in content_type or url.lower().endswith((".html", ".htm")):
        soup = BeautifulSoup(resp.text, "lxml")
        text = soup.get_text(separator="\n")
    else:
        # Fallback to raw text
        text = resp.text
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_secret_token(url: str) -> dict:
    """Search for a 64-character hex token in the web page content."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        text_content = response.text
        match = _secret_token_re.search(text_content)
        if match:
            secret_token = match.group(0)
            return {"answer": secret_token}
        else:
            return {"answer": "Token not found in the page content."}
    except requests.exceptions.RequestException as e:
        return {"answer": f"Error fetching URL: {e}"}


def handle_webpage_request(client, url: str, questions: List[str]):
    """Extract text from web page and answer questions using only that content."""
    try:
        context_text = scrape_text_from_url(url)
    except Exception as e:
        return jsonify({"error": "Failed to scrape URL", "details": str(e)}), 400

    answers: List[str] = [""] * len(questions)

    system_prompt = (
        "You answer strictly from the provided context. Guidelines:"
        " 1) Keep responses short and concise (1-2 lines)."
        " 2) Use plain, everyday language."
        " 3) Be direct and get straight to the point."
        " 4) Never add information not found in the context."
        " If the answer is not in the context, set answer to 'Information not found in the provided context.'"
        " Always respond in JSON exactly as {\"answer\": \"...\"}."
    )

    def _answer_one(idx_q):
        idx, q = idx_q
        try:
            q_lower = q.lower()
            if "secret token" in q_lower or "token" in q_lower:
                token_result = get_secret_token(url)
                return idx, token_result.get("answer", "Token not found in the page content.")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {q}\nRespond as per instructions."},
            ]
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            try:
                import json as _json
                parsed = _json.loads(content)
                return idx, parsed.get("answer", "Information not found in the provided context.")
            except Exception:
                return idx, "Information not found in the provided context."
        except Exception as e:
            return idx, "Information not found in the provided context."

    max_workers = min(len(questions), max(1, (os.cpu_count() or 4)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, ans in executor.map(_answer_one, list(enumerate(questions))):
            answers[idx] = ans

    return jsonify({"answers": answers}), 200


