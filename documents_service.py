import os
import re
import json
import time
import hashlib
import fitz
import faiss
import numpy as np
import requests
import tempfile
from bs4 import BeautifulSoup
from docx import Document
from email import policy
import email
import extract_msg
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

from llm_utils import call_gpt_fast as _call_gpt_fast_util, classify_user_query
from cache_utils import load_cache, save_cache
from prompts import (
    build_general_prompt,
    get_adaptive_prompt,
    build_insurance_prompt,
    build_health_policy_prompt,
    build_history_prompt,
    build_science_prompt,
)
from langchain_integration import EnhancedDocumentProcessor


ENABLE_SEMANTIC_CHUNKING = os.getenv("ENABLE_SEMANTIC_CHUNKING", "true").lower() == "true"
SEMANTIC_CHUNKING_THRESHOLD = float(os.getenv("SEMANTIC_CHUNKING_THRESHOLD", "85.0"))
MAX_PARALLEL_WORKERS = int(os.getenv("MAX_PARALLEL_WORKERS", "16"))
MAX_PROCESS_WORKERS = int(os.getenv("MAX_PROCESS_WORKERS", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

ADAPTIVE_CHUNKING_CONFIG = {
    "small_doc": {"max_pages": 100, "chunk_size": 700, "overlap": 120},
    "medium_doc": {"max_pages": 300, "chunk_size": 1000, "overlap": 150},
    "large_doc": {"max_pages": float('inf'), "chunk_size": 1200, "overlap": 200}
}

REGEX_PATTERNS = {
    'page_headers': re.compile(r"Page \d+ of \d+"),
    'whitespace': re.compile(r"\s{2,}"),
    'broken_words': re.compile(r"(\w+)-\s*\n\s*(\w+)"),
    'currency': re.compile(r"\$\s+(\d)"),
    'percentage': re.compile(r"(\d)\s+%"),
    'newlines': re.compile(r"\n{3,}"),
    'first_sentence': re.compile(r'[.!?]\s'),
}

@lru_cache(maxsize=1000)
def clean_text(text: str) -> str:
    """Remove page headers, fix spacing, and clean up document text."""
    text = REGEX_PATTERNS['page_headers'].sub("", text)
    text = REGEX_PATTERNS['whitespace'].sub(" ", text)
    text = REGEX_PATTERNS['broken_words'].sub(r"\1\2", text)
    text = REGEX_PATTERNS['currency'].sub(r"$\1", text)
    text = REGEX_PATTERNS['percentage'].sub(r"\1%", text)
    text = REGEX_PATTERNS['newlines'].sub("\n\n", text)
    return text.strip()


def get_adaptive_chunk_config(page_count: int) -> Dict[str, int]:
    """Choose chunk size based on document length - smaller chunks for shorter documents."""
    if page_count < 100:
        config = ADAPTIVE_CHUNKING_CONFIG["small_doc"]
    elif page_count < 300:
        config = ADAPTIVE_CHUNKING_CONFIG["medium_doc"]
    else:
        config = ADAPTIVE_CHUNKING_CONFIG["large_doc"]
    return {"chunk_size": config["chunk_size"], "overlap": config["overlap"], "page_count": page_count}


def get_domain_specific_prompt(client, question: str, context_chunks: List[Dict]) -> str:
    """Get domain-specific prompt based on question classification."""
    try:
        domain = classify_user_query(client, question)

        if domain == "insurance":
            return build_insurance_prompt(question, context_chunks)
        elif domain == "health_policy":
            return build_health_policy_prompt(question, context_chunks)
        elif domain == "history":
            return build_history_prompt(question, context_chunks)
        elif domain == "science":
            return build_science_prompt(question, context_chunks)
        else:
            return build_general_prompt(question, context_chunks)
    except Exception:
        return build_general_prompt(question, context_chunks)



# Top-level helper for multiprocessing: must be picklable on Windows
def _extract_single_page_for_pool(page_data):
    try:
        page_num, page = page_data
        text = extract_enhanced_text_from_page(page)
        cleaned_text = clean_text(text)
        return page_num, cleaned_text
    except Exception as e:
        return page_num, f"Error extracting text from page {page_num}: {str(e)}"

def parallel_extract_text_from_pages(pages: List) -> List[str]:
    """Extract text from pages in parallel using multiprocessing.
    Uses a top-level helper to be picklable on Windows. Falls back to threads if Pool fails.
    """
    if not pages:
        return []

    # FIX: PyMuPDF objects can't be pickled for multiprocessing, so we extract text first
    page_texts = []
    for i, page in enumerate(pages):
        try:
            text = extract_enhanced_text_from_page(page)
            cleaned_text = clean_text(text)
            page_texts.append((i, cleaned_text))
        except Exception as e:
            page_texts.append((i, f"Error extracting text from page {i}: {str(e)}"))

    # Now process the extracted text in parallel (if needed for any additional processing)
    results = page_texts
    
    # Sort by page number and extract text
    results.sort(key=lambda x: x[0])
    final_texts = [text for _, text in results]
    
    return final_texts


def parallel_chunk_generation(text_by_page: List[str]) -> List[Dict]:
    """Generate chunks in parallel for better performance."""
    def chunk_page_text(page_data):
        try:
            page_num, text = page_data
            if not text.strip():
                return []

            chunk_config = get_adaptive_chunk_config(len(text_by_page))

            if ENABLE_SEMANTIC_CHUNKING:
                # Create a new embeddings model instance for each thread to avoid conflicts
                embeddings_model = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    max_retries=2,
                    request_timeout=30,
                )
                return process_page_semantic_chunk((page_num, text, embeddings_model, chunk_config))
            else:
                return process_page_recursive_chunk((page_num, text, None, chunk_config))
        except Exception as e:
            return []

    # Use ThreadPoolExecutor for I/O-bound chunking operations
    max_workers = min(MAX_PARALLEL_WORKERS, len(text_by_page))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(chunk_page_text, (i, text)) for i, text in enumerate(text_by_page)]
        all_chunks = []
        for future in as_completed(futures):
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
            except Exception as e:
                pass

    return all_chunks


def parallel_embedding_creation(client, chunks: List[Dict]) -> Tuple[faiss.IndexFlatIP, List[Dict], np.ndarray]:
    """Create embeddings in parallel batches for better performance."""
    if not chunks:
        return None, [], np.array([])

    try:
        # Extract text content
        texts = [chunk["text"] for chunk in chunks]

        # Process embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch_texts = texts[i:i + EMBEDDING_BATCH_SIZE]

            # Create embeddings for batch
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch_texts
            )

            batch_embeddings = [data.embedding for data in resp.data]
            all_embeddings.extend(batch_embeddings)

        # Convert to numpy array
        embeddings = np.array(all_embeddings, dtype=np.float32)

        # Create FAISS index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        return index, chunks, embeddings
    except Exception as e:
        return None, [], np.array([])


def extract_tables_from_page(page) -> str:
    """Extract table data from a PDF page and format it as readable text."""
    try:
        tables = page.find_tables()
        table_text = ""
        for table_idx, table in enumerate(tables):
            try:
                table_data = table.extract()
                if table_data and len(table_data) > 0:
                    table_text += f"\n[TABLE {table_idx + 1}]\n"
                    for row in table_data:
                        if row and any(cell and str(cell).strip() for cell in row):
                            formatted_row = []
                            for cell in row:
                                if cell is not None:
                                    cell_text = str(cell).strip()
                                    cell_text = re.sub(r'\s+', ' ', cell_text)
                                    formatted_row.append(cell_text)
                                else:
                                    formatted_row.append("")
                            table_text += " | ".join(formatted_row) + "\n"
                    table_text += "[END TABLE]\n"
            except Exception:
                continue
        return table_text
    except Exception:
        return ""


def extract_enhanced_text_from_page(page) -> str:
    """Get both regular text and table data from a PDF page."""
    try:
        # Check if page is valid and not orphaned
        if page is None or not hasattr(page, 'get_text'):
            return "Error: Invalid page object"
        
        # Try to get regular text first
        try:
            regular_text = page.get_text()
        except Exception as e:
            if "orphaned object" in str(e).lower():
                return "Error: Page object is orphaned - cannot extract text"
            else:
                return f"Error extracting text: {str(e)}"
        
        # Try to get table text
        try:
            table_text = extract_tables_from_page(page)
        except Exception as e:
            table_text = ""
        
        # Combine text
        combined_text = regular_text + "\n\n" + table_text if table_text else regular_text
        return clean_text(combined_text)
        
    except Exception as e:
        return f"Error in enhanced text extraction: {str(e)}"


def extract_text_from_url(url: str) -> List[str]:
    """Download document from URL and extract text from each page."""
    start_ts = time.time()

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        content_length = len(response.content or b"")

        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.split('/')
        filename = path_parts[-1].lower() if path_parts and path_parts[-1] else ("document.pdf" if "pdf" in content_type else "document")

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        if "pdf" in content_type or filename.endswith(".pdf"):
            # Open with PyMuPDF and get pages
            doc = fitz.open(tmp_path)
            page_count = len(doc)
            
            # FIX: Extract text while document is still open to prevent orphaned objects
            texts = []
            for i in range(page_count):
                try:
                    page = doc[i]
                    text = extract_enhanced_text_from_page(page)
                    texts.append(text)
                except Exception as e:
                    error_text = f"Error extracting text from page {i}: {str(e)}"
                    texts.append(error_text)
            
            # Close document after extraction
            doc.close()
            
            return texts
        elif "word" in content_type or filename.endswith(".docx"):
            doc = Document(tmp_path)
            text = "\n".join(clean_text(p.text) for p in doc.paragraphs)
            return [text]
        elif "text/plain" in content_type or filename.endswith(".txt"):
            text = clean_text(response.text)
            return [text]
        elif "html" in content_type or filename.endswith(".html"):
            soup = BeautifulSoup(response.text, "lxml")
            text = clean_text(soup.get_text(separator="\n"))
            return [text]
        elif filename.endswith(".eml"):
            with open(tmp_path, "rb") as f:
                msg = email.message_from_binary_file(f, policy=policy.default)
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_payload(decode=True).decode(errors="ignore")
            else:
                body = msg.get_payload(decode=True).decode(errors="ignore")
            text = clean_text(body)
            return [text]
        elif filename.endswith(".msg"):
            msg = extract_msg.Message(tmp_path)
            text = clean_text(msg.body)
            return [text]
        else:
            raise ValueError("Unsupported document type or unknown format.")
    except Exception as e:
        raise


def process_page_semantic_chunk(page_data: Tuple) -> List[Dict]:
    page_num, page_text, embeddings_model, chunk_config = page_data
    if not page_text.strip():
        return []
    try:
        min_chunk_size = max(100, chunk_config["chunk_size"] // 8)
        semantic_splitter = SemanticChunker(
            embeddings_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=SEMANTIC_CHUNKING_THRESHOLD,
            min_chunk_size=min_chunk_size,
        )
        semantic_chunks = semantic_splitter.split_text(page_text)
        processed_chunks = []
        for i, chunk in enumerate(semantic_chunks):
            if len(chunk.strip()) < 50:
                continue
            first_sentence = REGEX_PATTERNS['first_sentence'].split(chunk.strip())[0][:100]
            context_prefix = f"Page {page_num}, Semantic Section {i+1}: {first_sentence}... "
            chunk_lower = chunk.lower()
            contains_definition = 'means' in chunk_lower and bool(re.search(r'\b\w+\s+means\b', chunk_lower))
            contains_exclusion = bool(re.search(r'\bexclusion|\bexcluded|\bnot covered|\bnot eligible', chunk_lower))
            contains_coverage = bool(re.search(r'\bcoverage|\bcovered|\beligible|\bincluded', chunk_lower))
            contains_limit = bool(re.search(r'\blimit|\bcap|\bmaximum|\bupto|\bup to', chunk_lower))
            contains_condition = bool(re.search(r'\bcondition|\bprovided that|\bsubject to|\bif and only if', chunk_lower))
            metadata = []
            if contains_definition: metadata.append("definition")
            if contains_exclusion: metadata.append("exclusion")
            if contains_coverage: metadata.append("coverage")
            if contains_limit: metadata.append("limit")
            if contains_condition: metadata.append("condition")
            processed_chunks.append({
                "text": context_prefix + chunk,
                "page": page_num,
                "section": i+1,
                "raw_text": chunk,
                "metadata": metadata,
                "chunk_type": "semantic",
                "chunk_config": chunk_config,
                "actual_size": len(chunk)
            })
        return processed_chunks
    except Exception:
        return process_page_recursive_chunk((page_num, page_text, chunk_config))


def process_page_recursive_chunk(page_data: Tuple) -> List[Dict]:
    if len(page_data) == 3:
        page_num, page_text, chunk_config = page_data
    else:
        page_num, page_text = page_data
        chunk_config = {"chunk_size": 800, "overlap": 100}
    if not page_text.strip():
        return []
    separators = [
        "\n\nARTICLE", "\n\nSECTION", "\n\nCLAUSE", "\n\nPART",
        "\n\nCOVERAGE", "\n\nBENEFIT", "\n\nEXCLUSION", "\n\nLIMIT",
        "\n\nArticle", "\n\nSection", "\n\nClause", "\n\nPart",
        "\n\nCoverage", "\n\nBenefit", "\n\nExclusion", "\n\nLimit",
        r"\n\d+\.\d+", r"\n\d+\.", r"\n[A-Z]\.", r"\n[a-z]\.", r"\n[ivxIVX]+\.",
        "\n\n", "\n", ". ", "; ", ", ", " "
    ]
    chunk_size = chunk_config["chunk_size"]
    chunk_overlap = chunk_config["overlap"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    page_chunks = splitter.split_text(page_text)
    processed_chunks = []
    for i, chunk in enumerate(page_chunks):
        if len(chunk.strip()) < 50:
            continue
        first_sentence = REGEX_PATTERNS['first_sentence'].split(chunk.strip())[0][:100]
        context_prefix = f"Page {page_num}, Section {i+1}: {first_sentence}... "
        chunk_lower = chunk.lower()
        contains_definition = 'means' in chunk_lower and bool(re.search(r'\b\w+\s+means\b', chunk_lower))
        contains_exclusion = bool(re.search(r'\bexclusion|\bexcluded|\bnot covered|\bnot eligible', chunk_lower))
        contains_coverage = bool(re.search(r'\bcoverage|\bcovered|\beligible|\bincluded', chunk_lower))
        contains_limit = bool(re.search(r'\blimit|\bcap|\bmaximum|\bupto|\bup to', chunk_lower))
        contains_condition = bool(re.search(r'\bcondition|\bprovided that|\bsubject to|\bif and only if', chunk_lower))
        metadata = []
        if contains_definition: metadata.append("definition")
        if contains_exclusion: metadata.append("exclusion")
        if contains_coverage: metadata.append("coverage")
        if contains_limit: metadata.append("limit")
        if contains_condition: metadata.append("condition")
        processed_chunks.append({
            "text": context_prefix + chunk,
            "page": page_num,
            "section": i+1,
            "raw_text": chunk,
            "metadata": metadata,
            "chunk_type": "recursive",
            "chunk_config": chunk_config,
            "actual_size": len(chunk)
        })
    return processed_chunks


def generate_smart_chunks(text_by_page: List[str]) -> List[Dict]:
    """Generate chunks using enhanced parallel processing."""
    page_count = len([p for p in text_by_page if p.strip()])
    chunk_config = get_adaptive_chunk_config(page_count)

    # First, try the EnhancedDocumentProcessor for richer chunking
    try:
        processor = EnhancedDocumentProcessor()
        lc_chunks = processor.process_documents(text_by_page, chunk_config=chunk_config)
        if isinstance(lc_chunks, list) and lc_chunks:
            return lc_chunks
    except Exception:
        # Fall back to parallel chunking strategies below
        pass

    # Use parallel chunk generation for better performance
    return parallel_chunk_generation(text_by_page)


def embed_chunks_openai(client, chunks: List[Dict]) -> Tuple[faiss.IndexFlatIP, List[Dict], np.ndarray]:
    """Create embeddings using parallel processing for better performance."""
    index, processed_chunks, embeddings = parallel_embedding_creation(client, chunks)

    if index is not None:
        # Normalize embeddings for better similarity search
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]

        # Rebuild index with normalized embeddings
        dimension = len(norm_embeddings[0])
        index = faiss.IndexFlatIP(dimension)
        index.add(norm_embeddings)

        return index, processed_chunks, norm_embeddings

    return None, chunks, np.array([])


def answer_questions_with_context(client, question: str, context_chunks: List[Dict]) -> str:
    """Answer questions using domain-specific prompts for better accuracy."""
    try:
        # Use domain-specific prompt based on question classification
        prompt = get_domain_specific_prompt(client, question, context_chunks)
        return _call_gpt_fast_util(client, prompt)
    except Exception as e:
        try:
            prompt = get_adaptive_prompt(client, question, context_chunks)
        except Exception:
            prompt = build_general_prompt(question, context_chunks)
        return _call_gpt_fast_util(client, prompt)


def handle_regular_document(client, hybrid_retriever, answer_validator, llm_reranker, query_router, document_url: str, questions: List[str]) -> Tuple[Dict, int]:
    try:
        cache_key = None
        try:
            import hashlib
            cache_key = hashlib.md5(document_url.encode()).hexdigest()
        except Exception:
            pass

        # Load or extract text pages (cache)
        if cache_key:
            cached_pages = load_cache(f"text:{cache_key}")
        else:
            cached_pages = None
        if cached_pages is not None:
            text_by_page = cached_pages
        else:
            text_by_page = extract_text_from_url(document_url)
            if cache_key:
                try:
                    save_cache(f"text:{cache_key}", text_by_page)
                except Exception:
                    pass

        # Identify sections and cross-checking can be done at higher layer; keep focused here
        # Load or generate chunks (cache)
        if cache_key:
            cached_chunks = load_cache(f"chunks:{cache_key}")
        else:
            cached_chunks = None
        if cached_chunks is not None:
            chunks = cached_chunks
        else:
            chunks = generate_smart_chunks(text_by_page)
            if cache_key:
                try:
                    save_cache(f"chunks:{cache_key}", chunks)
                except Exception:
                    pass
        if not chunks:
            return {"answers": ["Document content could not be processed into chunks."]}, 200

        # Try to build hybrid indices first; if that fails, fall back to local FAISS index
        hybrid_ready = False
        try:
            if hybrid_retriever is not None and hasattr(hybrid_retriever, "build_indices"):

                hybrid_retriever.build_indices(chunks)
                hybrid_ready = True
        except Exception as e:

            hybrid_ready = False

        index = None
        if not hybrid_ready:
            # Try loading cached embeddings; otherwise compute and cache
            cached_emb = load_cache(f"embeddings_norm:{cache_key}") if cache_key else None
            if cached_emb is not None and isinstance(cached_emb, list) and cached_emb:

                import numpy as _np
                norm_embeddings = _np.array(cached_emb, dtype=_np.float32)
                index = faiss.IndexFlatIP(norm_embeddings.shape[1])
                index.add(norm_embeddings)
            else:
                # Embed chunks and build local FAISS index
                index, chunks, norm_embeddings = embed_chunks_openai(client, chunks)
                if cache_key:
                    try:
                        save_cache(f"embeddings_norm:{cache_key}", norm_embeddings.tolist())
                    except Exception:
                        pass

        # Enhanced parallel question processing with domain-specific prompts
        answers: List[str] = [""] * len(questions)

        def _process_question_enhanced(idx_q_q):
            idx, q = idx_q_q
            try:
                # Classify question domain for better prompt selection
                domain = classify_user_query(client, q)

                use_hybrid = hybrid_ready and hasattr(hybrid_retriever, 'hybrid_search')
                k = 8
                if use_hybrid:
                    results = hybrid_retriever.hybrid_search(q, k=k)
                    retrieved = results if isinstance(results, list) else []
                else:
                    q_emb = client.embeddings.create(model="text-embedding-3-small", input=q).data[0].embedding
                    q_vec = np.array(q_emb, dtype=np.float32)
                    q_vec = q_vec / np.linalg.norm(q_vec)
                    scores, indices = index.search(np.array([q_vec]), k)
                    retrieved = [{"text": chunks[i]["text"], "idx": int(i)} for i in indices[0] if 0 <= i < len(chunks)]

                # Optional LLM re-ranking (runs concurrently per question)
                if llm_reranker is not None and hasattr(llm_reranker, 'rerank_chunks'):
                    try:
                        top_k_rerank = min(5, len(retrieved)) if retrieved else 0
                        if top_k_rerank > 0:
                            retrieved = llm_reranker.rerank_chunks(q, retrieved, top_k=top_k_rerank)
                    except Exception:
                        pass

                # Choose relevant chunks based on domain
                relevant_chunks = []
                seen = set()
                max_chunks = 8 if domain in ["insurance", "health_policy"] else 6

                for r in retrieved:
                    text = r.get("text") or (chunks[r["idx"]]["text"] if "idx" in r else None)
                    if text and text not in seen:
                        seen.add(text)
                        relevant_chunks.append({"text": text})
                        if len(relevant_chunks) >= max_chunks:
                            break

                # Use domain-specific prompt for better accuracy
                answer = answer_questions_with_context(client, q, relevant_chunks)

                # Enhanced answer validation with domain-specific checks
                if answer_validator is not None and hasattr(answer_validator, 'validate_answer'):
                    try:
                        report = answer_validator.validate_answer(q, answer, relevant_chunks)
                        confidence = float(report.get("confidence", 1.0)) if isinstance(report, dict) else 1.0

                        # Lower threshold for domain-specific questions
                        threshold = 0.3 if domain in ["insurance", "health_policy"] else 0.4

                        if confidence < threshold:
                            # Try with fewer chunks for better focus
                            fallback_chunks = relevant_chunks[:3]
                            fallback_answer = answer_questions_with_context(client, q, fallback_chunks)
                            answer = fallback_answer
                    except Exception:
                        pass

                return idx, answer
            except Exception as e:
                return idx, f"Error answering question: {str(e)}"

        # Use more workers for better parallelization
        max_workers = min(len(questions), MAX_PARALLEL_WORKERS, max(1, (os.cpu_count() or 4)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_question_enhanced, (idx, q)) for idx, q in enumerate(questions)]

            for future in as_completed(futures):
                try:
                    idx, answer = future.result()
                    answers[idx] = answer
                except Exception as e:
                    pass
                    # Find the index for this failed future
                    for i, f in enumerate(futures):
                        if f == future:
                            answers[i] = f"Error processing question: {str(e)}"
                            break

        return {"answers": answers}, 200
    except Exception as e:
        return {"error": "Regular document processing error", "details": str(e)}, 500

