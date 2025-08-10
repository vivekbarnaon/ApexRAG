
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
import re
from functools import lru_cache


class AnswerValidator:
    """Validates LLM answers against document context using multiple scoring methods."""
    
    def __init__(self, client: OpenAI, embedding_model: str = "text-embedding-3-small"):
        self.client = client
        self.embedding_model = embedding_model
        
    def validate_answer(
        self, 
        question: str, 
        answer: str, 
        context_chunks: List[Dict[str, Any]], 
        min_confidence: float = 0.6
    ) -> Dict[str, Any]:
        """Check if an answer is well-supported by the provided context chunks."""
        try:
            similarity_score = self._calculate_embedding_similarity(answer, context_chunks)
            
            content_overlap = self._calculate_content_overlap(answer, context_chunks)
            
            # Verify if the answer's facts match what's in the context
            factual_consistency = self._check_factual_consistency(answer, context_chunks)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(
                similarity_score, 
                content_overlap, 
                factual_consistency
            )
            
            # Determine if answer is valid
            is_valid = confidence >= min_confidence
            
            # Generate validation report
            validation_report = self._generate_validation_report(
                question, answer, context_chunks, confidence, is_valid
            )
            
            return {
                "is_valid": is_valid,
                "confidence": confidence,
                "similarity_score": similarity_score,
                "content_overlap": content_overlap,
                "factual_consistency": factual_consistency,
                "validation_report": validation_report,
                "supporting_chunks": self._get_supporting_chunks(answer, context_chunks)
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _calculate_embedding_similarity(
        self, 
        answer: str, 
        context_chunks: List[Dict[str, Any]]
    ) -> float:
        """Calculate semantic similarity between answer and context using embeddings."""
        try:
            # Get answer embedding
            answer_resp = self.client.embeddings.create(
                model=self.embedding_model,
                input=answer
            )
            answer_embedding = np.array(answer_resp.data[0].embedding)
            
            # Get context embeddings
            context_texts = [chunk.get("raw_text", chunk.get("text", "")) for chunk in context_chunks]
            if not context_texts:
                return 0.0
            
            context_resp = self.client.embeddings.create(
                model=self.embedding_model,
                input=context_texts
            )
            context_embeddings = np.array([d.embedding for d in context_resp.data])
            
            # Calculate cosine similarities
            similarities = []
            for context_embedding in context_embeddings:
                similarity = np.dot(answer_embedding, context_embedding) / (
                    np.linalg.norm(answer_embedding) * np.linalg.norm(context_embedding)
                )
                similarities.append(similarity)
            
            # Return average similarity
            return float(np.mean(similarities)) if similarities else 0.0
            
        except Exception:
            # Return neutral score if embedding calculation fails
            return 0.0
    
    def _calculate_content_overlap(
        self, 
        answer: str, 
        context_chunks: List[Dict[str, Any]]
    ) -> float:
        """Calculate how many important words appear in both answer and context."""
        try:
            # Extract key terms from answer
            answer_terms = self._extract_key_terms(answer)
            if not answer_terms:
                return 0.0
            
            # Extract key terms from context
            context_text = " ".join([
                chunk.get("raw_text", chunk.get("text", "")) 
                for chunk in context_chunks
            ])
            context_terms = self._extract_key_terms(context_text)
            
            if not context_terms:
                return 0.0
            
            # Calculate overlap
            overlap = len(answer_terms & context_terms)
            total_answer_terms = len(answer_terms)
            
            return overlap / total_answer_terms if total_answer_terms > 0 else 0.0
            
        except Exception:
            # Return neutral score if embedding calculation fails
            return 0.0
    
    @lru_cache(maxsize=500)
    def _extract_key_terms(self, text: str) -> set:
        """Extract important words, numbers, and terms from text for comparison."""
        # Convert to lowercase
        text = text.lower()
        
        # Extract important terms (numbers, financial terms, specific words)
        terms = set()
        
        # Extract numbers and financial amounts
        numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', text)
        terms.update(numbers)
        
        # Extract currency amounts
        currency = re.findall(r'\$\s*\d+(?:,\d{3})*(?:\.\d+)?', text)
        terms.update(currency)
        
        # Extract percentages
        percentages = re.findall(r'\d+(?:\.\d+)?%', text)
        terms.update(percentages)
        
        # Extract important words (longer than 3 characters)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        
        # Filter out common words
        stop_words = {
            'this', 'that', 'these', 'those', 'with', 'from', 'they', 'them', 'their',
            'there', 'where', 'when', 'what', 'which', 'will', 'would', 'could',
            'should', 'might', 'must', 'have', 'been', 'being', 'does', 'done'
        }
        
        important_words = {word for word in words if word not in stop_words}
        terms.update(important_words)
        
        return terms
    
    def _check_factual_consistency(
        self, 
        answer: str, 
        context_chunks: List[Dict[str, Any]]
    ) -> float:
        """Check if the answer's key facts can be found in the context."""
        try:
            # Extract factual claims from answer
            factual_claims = self._extract_factual_claims(answer)
            if not factual_claims:
                return 1.0  # No claims to verify
            
            # Check each claim against context
            context_text = " ".join([
                chunk.get("raw_text", chunk.get("text", "")) 
                for chunk in context_chunks
            ]).lower()
            
            verified_claims = 0
            for claim in factual_claims:
                if self._verify_claim_in_context(claim, context_text):
                    verified_claims += 1
            
            return verified_claims / len(factual_claims) if factual_claims else 1.0
            
        except Exception:
            # Return neutral score if consistency check fails
            return 0.5
    
    def _extract_factual_claims(self, answer: str) -> List[str]:
        """Find specific factual statements in the answer that can be verified."""
        claims = []
        
        # Extract numerical claims
        numerical_claims = re.findall(r'[^.!?]*\d+[^.!?]*[.!?]', answer)
        claims.extend(numerical_claims)
        
        # Extract specific statements (sentences with "is", "are", "will", etc.)
        specific_claims = re.findall(r'[^.!?]*(?:is|are|will|must|shall|can)[^.!?]*[.!?]', answer, re.IGNORECASE)
        claims.extend(specific_claims)
        
        # Clean and deduplicate claims
        cleaned_claims = []
        for claim in claims:
            claim = claim.strip()
            if len(claim) > 10 and claim not in cleaned_claims:
                cleaned_claims.append(claim)
        
        return cleaned_claims[:5]  # Limit to top 5 claims
    
    def _verify_claim_in_context(self, claim: str, context_text: str) -> bool:
        """Check if most key words from a claim appear in the context."""
        # Extract key terms from claim
        claim_terms = self._extract_key_terms(claim)
        
        # Check if most key terms are present in context
        found_terms = 0
        for term in claim_terms:
            if term.lower() in context_text:
                found_terms += 1
        
        # Claim is verified if at least 70% of terms are found
        return (found_terms / len(claim_terms)) >= 0.7 if claim_terms else False
    
    def _calculate_confidence(
        self, 
        similarity_score: float, 
        content_overlap: float, 
        factual_consistency: float
    ) -> float:
        """Combine all scores to get overall confidence in the answer."""
        # Weighted combination of scores
        weights = {
            'similarity': 0.4,
            'overlap': 0.3,
            'consistency': 0.3
        }
        
        confidence = (
            weights['similarity'] * similarity_score +
            weights['overlap'] * content_overlap +
            weights['consistency'] * factual_consistency
        )
        
        return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _generate_validation_report(
        self, 
        question: str, 
        answer: str, 
        context_chunks: List[Dict[str, Any]], 
        confidence: float, 
        is_valid: bool
    ) -> str:
        """Create a readable report explaining the validation results."""
        report = f"Answer Validation Report\n"
        report += f"========================\n\n"
        report += f"Question: {question}\n\n"
        report += f"Answer: {answer}\n\n"
        report += f"Confidence Score: {confidence:.2f}\n"
        report += f"Validation Status: {'VALID' if is_valid else 'INVALID'}\n\n"
        
        if confidence < 0.6:
            report += " Low confidence - answer may not be well-supported by context\n"
        elif confidence < 0.8:
            report += " Moderate confidence - answer appears reasonably supported\n"
        else:
            report += " High confidence - answer is well-supported by context\n"
        
        report += f"\nContext chunks analyzed: {len(context_chunks)}\n"
        
        return report
    
    def _get_supporting_chunks(
        self, 
        answer: str, 
        context_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find the context chunks that best support the answer."""
        supporting_chunks = []
        answer_terms = self._extract_key_terms(answer)
        
        for chunk in context_chunks:
            chunk_text = chunk.get("raw_text", chunk.get("text", ""))
            chunk_terms = self._extract_key_terms(chunk_text)
            
            # Calculate support score
            overlap = len(answer_terms & chunk_terms)
            support_score = overlap / len(answer_terms) if answer_terms else 0.0
            
            if support_score > 0.1:  # Only include chunks with some support
                supporting_chunk = chunk.copy()
                supporting_chunk['support_score'] = support_score
                supporting_chunks.append(supporting_chunk)
        
        # Sort by support score
        supporting_chunks.sort(key=lambda x: x['support_score'], reverse=True)
        
        return supporting_chunks[:3]  # Return top 3 supporting chunks
