
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple
import re
from openai import OpenAI
from functools import lru_cache


class HybridRetriever:
    """Combines semantic search (AI embeddings) with keyword search (BM25) for better results."""
    
    def __init__(self, client: OpenAI, embedding_model: str = "text-embedding-3-small"):
        self.client = client
        self.embedding_model = embedding_model
        self.faiss_index = None
        self.bm25_index = None
        self.chunks = []
        self.chunk_embeddings = None
        self.tokenized_chunks = []
        
    def build_indices(self, chunks: List[Dict[str, Any]]) -> None:
        """Build both semantic (FAISS) and keyword (BM25) search indices from document chunks."""
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Build FAISS index
        self._build_faiss_index(texts)
        
        # Build BM25 index
        self._build_bm25_index(texts)
        
    
    def _build_faiss_index(self, texts: List[str]) -> None:
        """Create AI embeddings for each text chunk and build semantic search index."""
        try:
            # Process in batches to avoid API limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                resp = self.client.embeddings.create(
                    model=self.embedding_model, 
                    input=batch_texts
                )
                batch_embeddings = [d.embedding for d in resp.data]
                all_embeddings.extend(batch_embeddings)
            
            # Convert to numpy array and normalize
            embeddings = np.array(all_embeddings, dtype=np.float32)
            self.chunk_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]
            
            # Build FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.chunk_embeddings.shape[1])
            self.faiss_index.add(self.chunk_embeddings)
            
        except Exception as e:
            raise
    
    def _build_bm25_index(self, texts: List[str]) -> None:
        """Create keyword search index using BM25 algorithm."""
        try:
            # Decide whether to parallelize tokenization based on document size
            page_count_hint = None
            try:
                for chunk in (self.chunks or []):
                    if isinstance(chunk, dict):
                        cfg = chunk.get("chunk_config")
                        if isinstance(cfg, dict) and isinstance(cfg.get("page_count"), (int, float)):
                            page_count_hint = int(cfg["page_count"])
                            break
                if page_count_hint is None:
                    pages = {c.get("page") for c in (self.chunks or []) if isinstance(c, dict) and c.get("page") is not None}
                    if pages:
                        page_count_hint = len(pages)
            except Exception:
                page_count_hint = None

            use_parallel = bool(page_count_hint and page_count_hint > 100)

            if use_parallel:
                from concurrent.futures import ThreadPoolExecutor
                import os as _os
                max_workers = min(8, max(1, (_os.cpu_count() or 4)))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    self.tokenized_chunks = list(executor.map(self._tokenize_text, texts))
            else:
                # Tokenize texts for BM25 (sequential)
                self.tokenized_chunks = [self._tokenize_text(text) for text in texts]
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(self.tokenized_chunks)
            
        except Exception as e:
            raise
    
    @lru_cache(maxsize=1000)
    def _tokenize_text(self, text: str) -> List[str]:
        """Break text into individual words for keyword search, filtering out common words."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\-\$\%\.]', ' ', text)
        
        # Split into tokens
        tokens = text.split()
        
        # Filter out very short tokens and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        filtered_tokens = [
            token for token in tokens 
            if len(token) > 2 and token not in stop_words
        ]
        
        return filtered_tokens
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = 10, 
        semantic_weight: float = 0.7, 
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Search using both AI understanding and keyword matching, then combine results."""
        
        if not self.faiss_index or not self.bm25_index:
            raise ValueError("Indices not built. Call build_indices() first.")
        
        # Get semantic search results
        semantic_results = self._semantic_search(query, k * 2)  # Get more candidates
        
        # Get keyword search results
        keyword_results = self._keyword_search(query, k * 2)  # Get more candidates
        
        # Combine and rank results
        combined_results = self._combine_rankings(
            semantic_results, 
            keyword_results, 
            semantic_weight, 
            keyword_weight
        )
        
        # Return top k results
        return combined_results[:k]
    
    def _semantic_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        
        try:
            # Get query embedding
            q_resp = self.client.embeddings.create(
                model=self.embedding_model, 
                input=query
            )
            q_vec = np.array(q_resp.data[0].embedding, dtype=np.float32)
            q_vec = q_vec / np.linalg.norm(q_vec)
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(np.array([q_vec]), k)
            
            # Return results as (index, score) tuples
            return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
            
        except Exception as e:
            return []
    
    def _keyword_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        try:
            # Tokenize query
            query_tokens = self._tokenize_text(query)
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top k indices
            top_indices = np.argsort(scores)[::-1][:k]
            
            # Return results as (index, score) tuples
            return [(int(idx), float(scores[idx])) for idx in top_indices]
            
        except Exception as e:
            return []
    
    def _combine_rankings(
        self, 
        semantic_results: List[Tuple[int, float]], 
        keyword_results: List[Tuple[int, float]], 
        semantic_weight: float, 
        keyword_weight: float
    ) -> List[Dict[str, Any]]:
        
        # Normalize scores to [0, 1] range
        semantic_scores = self._normalize_scores([score for _, score in semantic_results])
        keyword_scores = self._normalize_scores([score for _, score in keyword_results])
        
        # Create score dictionaries
        semantic_dict = {idx: norm_score for (idx, _), norm_score in zip(semantic_results, semantic_scores)}
        keyword_dict = {idx: norm_score for (idx, _), norm_score in zip(keyword_results, keyword_scores)}
        
        # Combine scores
        combined_scores = {}
        all_indices = set(semantic_dict.keys()) | set(keyword_dict.keys())
        
        for idx in all_indices:
            sem_score = semantic_dict.get(idx, 0.0)
            key_score = keyword_dict.get(idx, 0.0)
            combined_score = (semantic_weight * sem_score) + (keyword_weight * key_score)
            combined_scores[idx] = combined_score
        
        # Sort by combined score
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return enriched chunks
        results = []
        for idx, score in sorted_indices:
            chunk = self.chunks[idx].copy()
            chunk['combined_score'] = score
            chunk['semantic_score'] = semantic_dict.get(idx, 0.0)
            chunk['keyword_score'] = keyword_dict.get(idx, 0.0)
            results.append(chunk)
        
        return results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
