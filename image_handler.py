from flask import jsonify
import requests
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed


def is_image_url(url: str) -> bool:
    """Detect common image URLs via extension or Content-Type header."""
    try:
        lowered = url.lower().split("?")[0]
        if lowered.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
            return True
        # Fallback to HEAD content-type
        try:
            resp = requests.head(url, timeout=10)
            ctype = resp.headers.get("Content-Type", "").lower()
            if ctype.startswith("image/"):
                return True
        except Exception:
            pass
        return False
    except Exception:
        return False


def handle_image_file(client, url: str, questions: List[str]):
    """Fast path: single vision call to answer all questions, then wrap into {"answers": [...]}
    with the same order as questions. The model is asked to correct obvious factual errors
    using general knowledge within the same pass.
    """
    default_answer = "Information not found in the provided image."

    # Build a single prompt listing all questions and asking for numbered answers 1..N
    q_lines = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

    system_prompt = (
        "Analyze the provided image and answer each question in order. "
        "If any information appears wrong, correct it using well-known general knowledge. "
        f"If a question cannot be answered from the image, reply: '{default_answer}'. "
        "Return answers as a numbered list (1..N), one line per answer, no extra commentary."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Questions:\n{q_lines}\nAnswer in a numbered list (1..{len(questions)})."},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        },
    ]

    raw_output = ""
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=500,
        )
        raw_output = (resp.choices[0].message.content or "").strip()
    except Exception:
        raw_output = ""

    # If many questions, use parallel fallback: chunk questions and call in parallel to speed up
    if not raw_output and len(questions) > 1:
        # Split into chunks of up to 5 questions per request
        chunk_size = 5
        chunks = [questions[i:i+chunk_size] for i in range(0, len(questions), chunk_size)]
        results = [None] * len(chunks)

        def _call_chunk(idx, chunk):
            local_q_lines = "\n".join([f"{i+1}. {q}" for i, q in enumerate(chunk)])
            messages_local = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Questions:\n{local_q_lines}\nAnswer in a numbered list (1..{len(chunk)})."},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                },
            ]
            try:
                r = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=messages_local,
                    temperature=0.1,
                    max_tokens=300,
                )
                return idx, (r.choices[0].message.content or "").strip()
            except Exception:
                return idx, ""

        max_workers = min(len(chunks), max(1, (os.cpu_count() or 4)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_call_chunk, i, ch) for i, ch in enumerate(chunks)]
            for fut in as_completed(futures):
                i, out = fut.result()
                results[i] = out

        # Combine raw outputs
        raw_output = "\n".join([r for r in results if r])

    # Parse numbered answers robustly
    answers: List[str] = [default_answer for _ in questions]
    if raw_output:
        lines = [l.strip() for l in raw_output.splitlines() if l.strip()]
        # Try to map lines starting with an index (e.g., 1., 2), - 1)
        import re as _re
        numbered = {}
        for line in lines:
            m = _re.match(r"^([0-9]{1,3})[\).\-:]\s*(.*)$", line)
            if m:
                idx = int(m.group(1))
                text = m.group(2).strip()
                if 1 <= idx <= len(questions) and text:
                    numbered[idx - 1] = text
        # If we captured enough, fill in order
        if numbered:
            for i in range(len(questions)):
                answers[i] = numbered.get(i, answers[i])
        else:
            # Fallback: take the first N non-empty lines
            for i in range(min(len(lines), len(questions))):
                answers[i] = lines[i]

    return jsonify({"answers": answers}), 200

