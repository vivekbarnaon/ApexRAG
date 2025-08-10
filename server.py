from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
import re
import hashlib
from typing import List
from openai import OpenAI

# Handlers and helpers
import handlers_files as file_handlers
from data_processor import is_excel_or_csv_url
from documents_service import handle_regular_document as _handle_regular_document_service
from documents_service import extract_text_from_url as _extract_text_pages
from cache_utils import save_cache
from url_scraper import is_plain_web_url, handle_webpage_request
from small_doc_qa import handle_small_document
from zip_processor import is_zip_url as is_zip_archive_url, process_zip_from_url as process_zip_archive_from_url
from image_handler import is_image_url, handle_image_file


# Core components
from enhanced_retrieval import HybridRetriever
from answer_validation import AnswerValidator
from llm_reranker import LLMReranker
from langchain_integration import QueryRouter


load_dotenv()
app = Flask(__name__)

# CORS: allow configured origins (comma separated), default to permissive for hackathon testing
_allowed_origins = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
if _allowed_origins:
    origins_list = [o.strip() for o in _allowed_origins.split(",") if o.strip()]
    CORS(app, resources={r"/*": {"origins": origins_list}})
else:
    CORS(app)

# Safety: cap request size (MB) to avoid very large uploads (default 5MB)
try:
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH_MB', '5')) * 1024 * 1024
except Exception:
    app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0, max_retries=2)
# Auth token now sourced from environment to avoid hardcoding; fallback preserves current behavior
TEAM_TOKEN = os.getenv(
    "TEAM_TOKEN"
)

# Initialize components
hybrid_retriever = HybridRetriever(client)
answer_validator = AnswerValidator(client)
llm_reranker = LLMReranker(client)
query_router = QueryRouter()



# Helper: answer questions using provided context text via OpenAI
def _answer_questions_from_context(client: OpenAI, context_text: str, questions: List[str]) -> List[str]:
    answers = ["Information not found in the provided context." for _ in questions]
    system_prompt = (
        "You are a helpful assistant. Answer strictly based on the provided context. "
        "Return a JSON object with a single key 'answer'. If the answer is not present, say 'Information not found in the provided context.'"
    )
    for i, q in enumerate(questions):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_text[:12000]}\n\nQuestion: {q}\nReturn JSON with key 'answer' only."},
            ]
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            import json as _json
            parsed = _json.loads(content)
            answers[i] = parsed.get("answer", answers[i])
        except Exception:
            pass
    return answers







@app.route("/api/v1/hackrx/run", methods=["POST", "OPTIONS"])
def run_submission():
    # Handle CORS preflight requests
    if request.method == "OPTIONS":
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'
        }
        return '', 204, headers

    try:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != TEAM_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401

        # Extremely lenient JSON parsing
        try:
            try:
                data = request.get_json(force=True)
                if data:
                    pass  # Data is valid, continue with processing
            except Exception:
                try:
                    try:
                        raw_data = request.data.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            raw_data = request.data.decode('latin-1')
                        except Exception:
                            raw_data = request.data.decode('utf-8', errors='replace')

                    url_pattern = r'documents"?\s*:(?:\s*")?((https?://[^"\s,}]+?)(?:\\"|"|\s|,|}|$))'
                    document_match = re.search(url_pattern, raw_data, re.IGNORECASE)

                    questions_pattern = r'questions"?\s*:\s*\[(.*?)\]'
                    questions_match = re.search(questions_pattern, raw_data, re.DOTALL)

                    if document_match and questions_match:
                        document_url = document_match.group(1).split('"')[0].split('\\')[0]
                        questions_text = questions_match.group(1)
                        questions = []
                        for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', questions_text):
                            if match.group(1):
                                questions.append(match.group(1))
                            else:
                                questions.append(match.group(2))
                        data = {"documents": document_url, "questions": questions}
                    else:
                        cleaned = raw_data.strip()
                        cleaned = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', cleaned)
                        data = json.loads(cleaned)
                except Exception as inner_e:
                    error_msg = f"All parsing attempts failed: {str(inner_e)}"
                    return jsonify({"error": "Invalid JSON format", "details": error_msg}), 400
        except Exception as e:
            error_msg = f"JSON parsing error: {str(e)}"
            return jsonify({"error": "Invalid JSON format", "details": error_msg}), 400

        document_url = data.get("documents")
        questions = data.get("questions")
        if not document_url or not questions:
            return jsonify({"error": "Missing 'documents' or 'questions'"}), 400

        # Early PDF shortcut: if PDF with <5 pages, directly call small_doc_qa and return
        try:
            from handlers_files import is_pdf_url as _is_pdf_url_early
            if _is_pdf_url_early(document_url):
                try:
                    pages_early = _extract_text_pages(document_url)
                    page_count_early = len([p for p in pages_early if isinstance(p, str) and p.strip()])
                except Exception:
                    pages_early, page_count_early = [], 0
                if 1 <= page_count_early <= 5:
                    return handle_small_document(client, pages_early, questions)
        except Exception:
            pass

        # Route by document type
        try:
            # 1) ZIP archives: summarize structure and contents
            if is_zip_archive_url(document_url):
                zip_result = process_zip_archive_from_url(document_url)
                # Build a context from the ZIP processor summary and extracted content
                context_parts = [zip_result.get("summary", "")]
                for sample in zip_result.get("extracted_content", []) or []:
                    try:
                        path = sample.get("path", "")
                        content = sample.get("content", "")
                        if path or content:
                            context_parts.append(f"--- {path} ---\n{content}")
                    except Exception:
                        pass
                context_text = "\n\n".join([p for p in context_parts if p])
                answers = _answer_questions_from_context(client, context_text, questions)
                status_code = 200 if zip_result.get("success", False) else 400
                return jsonify({"answers": answers}), status_code

            # 2) Image URLs: call OpenAI vision with strict JSON output per question
            if is_image_url(document_url):
                return handle_image_file(client, document_url, questions)

            # 3) Regular web pages
            if is_plain_web_url(document_url):
                return handle_webpage_request(client, document_url, questions)
            elif file_handlers.is_pptx_url(document_url):
                return file_handlers.handle_pptx_file(client, document_url, questions)
            elif file_handlers.is_bin_url(document_url):
                return file_handlers.handle_bin_file(document_url, questions)
            elif is_excel_or_csv_url(document_url):
                return file_handlers.handle_excel_csv_file(document_url, questions)
            else:
                # For PDFs only: if 1-5 pages, use small-doc QA to send full context.
                # DOCX and other types bypass this shortcut and go to regular processing.
                from handlers_files import is_pdf_url as _is_pdf_url
                is_pdf = False
                try:
                    is_pdf = _is_pdf_url(document_url)
                except Exception:
                    is_pdf = False
                pages = []
                page_count = 0
                if is_pdf:
                    try:
                        pages = _extract_text_pages(document_url)
                        page_count = len([p for p in pages if isinstance(p, str) and p.strip()])
                        # Save pages to cache to avoid re-download in downstream service
                        try:
                            import hashlib
                            cache_key = hashlib.md5(document_url.encode()).hexdigest()
                            save_cache(f"text:{cache_key}", pages)
                        except Exception:
                            pass
                    except Exception:
                        pages = []
                        page_count = 0
                # PDFs with fewer than 5 pages (1-4): bypass all other handlers and use small_doc_qa directly
                if is_pdf and (1 <= page_count < 5):
                    return handle_small_document(client, pages, questions)
                return _handle_regular_document_service(
                    client, hybrid_retriever, answer_validator, llm_reranker, query_router, document_url, questions
                )
        except Exception as e:
            error_msg = f"Error determining document type: {str(e)}"
            return jsonify({"error": "Document processing error", "details": error_msg}), 500

    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        return jsonify({"error": "Server error", "details": error_msg}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route("/debug/json", methods=["POST"])
def debug_json():
    try:
        raw_data = request.data.decode('utf-8')
        parsed = {}
        try:
            parsed["standard"] = json.loads(raw_data)
        except Exception as e:
            parsed["standard_error"] = str(e)
        try:
            parsed["flask"] = request.get_json(force=True)
        except Exception as e:
            parsed["flask_error"] = str(e)
        try:
            cleaned_data = raw_data.strip()
            cleaned_data = re.sub(r'"\s*:\s*"', '":"', cleaned_data)
            cleaned_data = re.sub(r'"\s*,\s*"', '","', cleaned_data)
            cleaned_data = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', cleaned_data)
            parsed["cleaned"] = json.loads(cleaned_data)
        except Exception as e:
            parsed["cleaned_error"] = str(e)
            parsed["cleaned_data"] = cleaned_data
        return jsonify({
            "raw_data": raw_data,
            "parsing_results": parsed,
            "content_type": request.content_type,
            "is_json": request.is_json
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)