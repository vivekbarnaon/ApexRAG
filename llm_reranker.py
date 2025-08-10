"""
LLM Re-ranking System - Uses AI to reorder search results by relevance.
"""

import json
import re
from typing import List, Dict, Any
from openai import OpenAI

class LLMReranker:
    """Uses AI to reorder search results by how well they answer the question."""
    
    def __init__(self, client: OpenAI, model: str = "gpt-4.1-mini"):
        self.client = client
        self.model = model
    
    def rerank_chunks(
        self, 
        question: str, 
        chunks: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Ask AI to rank document chunks by how well they answer the question."""
        try:
            if not chunks:
                return []
            
            # Limit chunks to avoid token limits
            chunks_to_rank = chunks[:min(len(chunks), 15)]
            
            # Create re-ranking prompt
            prompt = self._create_reranking_prompt(question, chunks_to_rank)
            
            # Get LLM re-ranking
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            reranking_result = self._parse_reranking_response(
                response.choices[0].message.content, 
                chunks_to_rank
            )
            
            # Apply re-ranking scores
            reranked_chunks = self._apply_reranking_scores(chunks_to_rank, reranking_result)
            
            # Return top k chunks
            return reranked_chunks[:top_k]
            
        except Exception as e:
            # Fallback to original ranking
            return chunks[:top_k]
    
    def _create_reranking_prompt(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Create prompt for LLM re-ranking.
        
        Args:
            question: User question
            chunks: Chunks to re-rank
            
        Returns:
            Re-ranking prompt
        """
        prompt = f"""You are an expert at evaluating document relevance for insurance policy questions.

TASK: Rank the following document chunks by their relevance to the user's question. Consider:
1. Direct relevance to the question topic
2. Presence of specific information that answers the question
3. Context that helps understand the answer
4. Insurance-specific terminology and concepts

QUESTION: {question}

DOCUMENT CHUNKS:
"""
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("raw_text", chunk.get("text", ""))[:500]  # Limit length
            prompt += f"\nChunk {i+1}:\n{chunk_text}\n"
        
        prompt += f"""
INSTRUCTIONS:
- Rank chunks from 1 (most relevant) to {len(chunks)} (least relevant)
- Assign relevance scores from 0.0 (not relevant) to 1.0 (highly relevant)
- Consider the specific question context and insurance domain
- Prioritize chunks that directly answer the question

Respond in JSON format:
{{
    "rankings": [
        {{"chunk_id": 1, "rank": 1, "relevance_score": 0.95, "reasoning": "Direct answer to question"}},
        {{"chunk_id": 2, "rank": 2, "relevance_score": 0.80, "reasoning": "Provides supporting context"}},
        ...
    ]
}}
"""
        return prompt
    
    def _parse_reranking_response(
        self, 
        response_content: str, 
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Parse LLM re-ranking response.
        
        Args:
            response_content: LLM response content
            chunks: Original chunks
            
        Returns:
            Parsed re-ranking result
        """
        try:
            # Parse JSON response
            result = json.loads(response_content)
            rankings = result.get("rankings", [])
            
            # Validate rankings
            valid_rankings = []
            for ranking in rankings:
                chunk_id = ranking.get("chunk_id", 0)
                if 1 <= chunk_id <= len(chunks):
                    valid_rankings.append({
                        "chunk_id": chunk_id - 1,  # Convert to 0-based index
                        "rank": ranking.get("rank", 999),
                        "relevance_score": min(max(ranking.get("relevance_score", 0.0), 0.0), 1.0),
                        "reasoning": ranking.get("reasoning", "")
                    })
            
            return {"rankings": valid_rankings}
            
        except json.JSONDecodeError as e:
            # Fallback: extract scores using regex
            return self._extract_scores_fallback(response_content, len(chunks))
        except Exception as e:
            return {"rankings": []}
    
    def _extract_scores_fallback(self, response_content: str, num_chunks: int) -> Dict[str, Any]:
        """
        Fallback method to extract scores using regex.
        
        Args:
            response_content: LLM response content
            num_chunks: Number of chunks
            
        Returns:
            Extracted rankings
        """
        rankings = []
        
        # Try to extract chunk rankings using regex
        patterns = [
            r'chunk[_\s]*(\d+)[^0-9]*?(\d+(?:\.\d+)?)',  # chunk 1: 0.95
            r'(\d+)[^0-9]*?chunk[^0-9]*?(\d+(?:\.\d+)?)',  # 1. chunk: 0.95
            r'chunk[_\s]*(\d+)[^0-9]*?rank[^0-9]*?(\d+)',  # chunk 1 rank 2
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_content, re.IGNORECASE)
            if matches:
                for i, (chunk_id, score) in enumerate(matches):
                    try:
                        chunk_idx = int(chunk_id) - 1
                        relevance_score = float(score) if '.' in score else float(score) / 10.0
                        
                        if 0 <= chunk_idx < num_chunks:
                            rankings.append({
                                "chunk_id": chunk_idx,
                                "rank": i + 1,
                                "relevance_score": min(max(relevance_score, 0.0), 1.0),
                                "reasoning": "Extracted from fallback parsing"
                            })
                    except (ValueError, IndexError):
                        continue
                break
        
        return {"rankings": rankings}
    
    def _apply_reranking_scores(
        self, 
        chunks: List[Dict[str, Any]], 
        reranking_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply re-ranking scores to chunks and sort them.
        
        Args:
            chunks: Original chunks
            reranking_result: Re-ranking result from LLM
            
        Returns:
            Re-ranked chunks
        """
        rankings = reranking_result.get("rankings", [])
        
        # Create a mapping of chunk index to ranking info
        ranking_map = {r["chunk_id"]: r for r in rankings}
        
        # Apply scores to chunks
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            
            if i in ranking_map:
                ranking_info = ranking_map[i]
                chunk_copy["llm_relevance_score"] = ranking_info["relevance_score"]
                chunk_copy["llm_rank"] = ranking_info["rank"]
                chunk_copy["llm_reasoning"] = ranking_info["reasoning"]
            else:
                # Default scores for unranked chunks
                chunk_copy["llm_relevance_score"] = 0.1
                chunk_copy["llm_rank"] = 999
                chunk_copy["llm_reasoning"] = "Not ranked by LLM"
            
            scored_chunks.append(chunk_copy)
        
        # Sort by LLM relevance score (descending)
        scored_chunks.sort(key=lambda x: x["llm_relevance_score"], reverse=True)
        
        return scored_chunks
    
    def rerank_with_explanation(
        self, 
        question: str, 
        chunks: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Re-rank chunks and provide detailed explanation.
        
        Args:
            question: User question
            chunks: Retrieved chunks to re-rank
            top_k: Number of top chunks to return
            
        Returns:
            Dictionary with re-ranked chunks and explanation
        """
        try:
            reranked_chunks = self.rerank_chunks(question, chunks, top_k)
            
            # Generate explanation
            explanation = self._generate_reranking_explanation(question, reranked_chunks)
            
            return {
                "reranked_chunks": reranked_chunks,
                "explanation": explanation,
                "original_count": len(chunks),
                "reranked_count": len(reranked_chunks)
            }
            
        except Exception as e:
            return {
                "reranked_chunks": chunks[:top_k],
                "explanation": f"Re-ranking failed: {str(e)}",
                "original_count": len(chunks),
                "reranked_count": min(len(chunks), top_k)
            }
    
    def _generate_reranking_explanation(
        self, 
        question: str, 
        reranked_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate explanation for re-ranking results.
        
        Args:
            question: User question
            reranked_chunks: Re-ranked chunks
            
        Returns:
            Explanation string
        """
        explanation = f"Re-ranking Analysis for: '{question}'\n\n"
        
        for i, chunk in enumerate(reranked_chunks[:3]):  # Top 3 chunks
            score = chunk.get("llm_relevance_score", 0.0)
            reasoning = chunk.get("llm_reasoning", "No reasoning provided")
            
            explanation += f"Rank {i+1} (Score: {score:.2f}):\n"
            explanation += f"Reasoning: {reasoning}\n"
            
            # Show snippet of chunk
            chunk_text = chunk.get("raw_text", chunk.get("text", ""))[:200]
            explanation += f"Content: {chunk_text}...\n\n"
        
        return explanation
