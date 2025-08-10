import numpy as np
from typing import List, Dict


def insurance_specific_retrieve(client, question: str, index, chunks: List[Dict], k: int = 8) -> List[Dict]:
    """Search for document chunks most relevant to the insurance question."""
    q_resp = client.embeddings.create(model="text-embedding-3-small", input=question)
    q_vec = np.array(q_resp.data[0].embedding, dtype=np.float32)
    q_vec = q_vec / np.linalg.norm(q_vec)
    scores, indices = index.search(np.array([q_vec]), k * 4)
    results = []
    for idx in indices[0][:k]:
        if 0 <= idx < len(chunks):
            results.append({"text": chunks[idx]["text"], "idx": int(idx)})
    return results


def validate_context_relevance(question: str, context_chunks: List[Dict], min_relevance: float = 0.25) -> List[Dict]:
    """Filter chunks to keep only those that share enough words with the question."""
    question_words = set([w.lower() for w in question.split()])
    filtered = []
    for c in context_chunks:
        text = c["text"].lower()
        words = set(text.split())
        relevance = len(question_words & words) / max(len(question_words), 1)
        if relevance >= min_relevance:
            filtered.append(c)
    return filtered[:8]

