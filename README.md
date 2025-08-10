<div align="center">
<img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&size=50&pause=2000&color=00D4FF,7B68EE,FF1493,32CD32,FFD700,FF4500,9370DB&center=true&vCenter=true&width=400&height=80&lines=Apex-RAG" alt="Apex-RAG" />
</div>

A production-ready Flask API for answering questions from documents, webpages, images, and archives. Built for Hackathon submissions with robust routing, caching, and OpenAI-powered reasoning.

### Highlights
- Smart routing by input type (PDF, DOCX, XLSX/CSV, ZIP, images, webpages)
- Small-document fast path for short PDFs (<= 4 pages) with strict context-only answers
- ZIP summarization and content synthesis
- Answer validation and LLM reranking pipeline
- Caching of extracted text chunks for speed
- CORS support and health/debug endpoints

---

## 1) Prerequisites
- Python 3.11
- An OpenAI API key with access to the specified model(s)
- Git (optional)

If you already have Python 3.11 installed, verify:

bash
python --version
# or on Windows if multiple Pythons:
py -3.11 --version


---

## 2) Setup (recommended: virtual environment)

### Windows (PowerShell)
powershell
# From repository root
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt


### macOS/Linux (bash/zsh)
bash
# From repository root
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt


To deactivate the venv later:
bash
deactivate


---

## 3) Configuration
This service reads configuration via environment variables (a local .env file is supported).

Create a file named .env in the repository root:

env
# Required
OPENAI_API_KEY=sk-...
TEAM_TOKEN=replace-with-your-shared-team-token



Notes:
- TEAM_TOKEN protects the main POST endpoint. Clients must send Authorization: Bearer <TEAM_TOKEN>.
- CORS_ALLOW_ORIGINS accepts a comma-separated list. If omitted, CORS is permissive (handy for hackathon testing).
- MAX_CONTENT_LENGTH_MB prevents very large uploads.

---

## 4) Run the server
From the repository root with your virtual environment activated:

bash
python server.py


By default the API listens on http://0.0.0.0:8000. You can override via HOST and PORT in .env or environment.

Health check:
bash
curl http://localhost:8000/health


---

## 5) API

### POST /api/v1/hackrx/run
Answer questions against a document or URL.

Headers:
- Authorization: Bearer <TEAM_TOKEN>
- Content-Type: application/json

Body (typical):
json
{
  "documents": "https://example.com/some.pdf",
  "questions": [
    "What is the total revenue in 2023?",
    "Who is the signatory?"
  ]
}


Example (PowerShell):
powershell
$body = '{
  "documents": "https://example.com/some.pdf",
  "questions": ["What is the total revenue in 2023?", "Who is the signatory?"]
}'

curl -X POST http://localhost:8000/api/v1/hackrx/run `
  -H "Authorization: Bearer YOUR_TEAM_TOKEN" `
  -H "Content-Type: application/json" `
  -d $body


Example (bash):
bash
curl -X POST http://localhost:8000/api/v1/hackrx/run \
  -H "Authorization: Bearer YOUR_TEAM_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/some.pdf",
    "questions": ["What is the total revenue in 2023?", "Who is the signatory?"]
  }'


Response:
json
{
  "answers": ["…", "…"]
}


---

## 6) What gets routed where
The service determines the handler based on the documents URL:

- ZIP archives
  - Summarizes archive structure and synthesizes a context from extracted samples
  - Answers are produced from that synthesized context

- Images (PNG/JPG, etc.)
  - Uses OpenAI vision model to produce strict JSON answers per question

- Web pages (HTTP/HTTPS)
  - Scrapes page content and answers from the extracted context

- Excel/CSV
  - Extracts and parses spreadsheet/tabular data, then answers based on parsed context

- PDFs
  - Small PDFs with 1–4 pages: small-document fast path (sends full extracted text to the LLM with strict system prompt)
  - PDFs with 5+ pages: regular retrieval-and-reranking pipeline

- Other documents (e.g., DOCX)
  - Regular pipeline (retrieval, reranker, answer validator)

Internally, pages are cached (when applicable) under cache/ to avoid repeated downloads and extraction during a session.

---

## 7) Models and limits
- The QA pipeline uses OpenAI Chat Completions with response_format set to JSON object for deterministic output.
- Temperature is kept low for factual responses.
- Small-document path passes the full (trimmed) text to the model for concise answers.

Ensure your OPENAI_API_KEY has access to the configured model(s).

---

## 8) Debugging
- Health endpoint: GET /health
- JSON parsing lab: POST /debug/json (returns multiple parsing attempts for the posted payload)
- Check server logs for exceptions. Common error responses include:
  - 401 Unauthorized — missing/invalid TEAM_TOKEN
  - 400 Invalid JSON format — body failed all parsing attempts
  - 500 Document processing error — handler failed while routing or extracting
  - 500 Server error — top-level uncaught errors

Tips:
- If posting from a browser/extension, verify CORS and headers.
- For binary or odd payloads, try /debug/json to see how the server interprets your request.

---

## 9) Development workflow
- Activate venv and run the server locally.
- Adjust environment in .env.
- Code structure (key modules):
  - server.py — Flask app, routing, and top-level orchestration
  - documents_service.py — main document processing flow
  - small_doc_qa.py — small PDF fast path (strict context-only answers)
  - handlers_files.py, image_handler.py, url_scraper.py — file/web/image utilities
  - zip_processor.py — ZIP archive summarization
  - enhanced_retrieval.py, llm_reranker.py, answer_validation.py, langchain_integration.py — retrieval and QA pipeline
  - cache_utils.py, retrieval_utils.py, analysis_utils.py — helpers and utilities

---

## 10) Troubleshooting
- Python version issues: ensure exactly Python 3.11 is used for your venv.
- OpenAI errors: verify OPENAI_API_KEY and model availability; inspect server logs.
- 401 Unauthorized: confirm the Authorization header value matches TEAM_TOKEN.
- Large file uploads: increase MAX_CONTENT_LENGTH_MB in .env if needed.


