import re
from llm_utils import classify_user_query as _classify_user_query_util

from typing import List, Dict

# Analysis helpers

def detect_irrelevant_question(question: str, context_chunks: List[Dict] | None = None) -> bool:
    """Check if question contains inappropriate terms like fraud or hacking."""
    question_lower = (question or "").lower()
    inappropriate_patterns = [
        r"\bfraud\b", r"\bscam\b", r"\bcheat\b",
        r"\bhack\b", r"\bsteal\b", r"\bfake\b", r"\bforge\b",
        r"\bmanipulate\b", r"\bdeceive\b", r"\bmisrepresent\b",
        r"\babuse\b", r"\bexploit\b"
    ]
    return any(re.search(p, question_lower) for p in inappropriate_patterns)


def analyze_context_specificity(question: str, context_chunks: List[Dict]) -> Dict[str, object]:
    """Check if the question asks about specific products mentioned in the context."""
    question_lower = (question or "").lower()
    context_text = " ".join([chunk.get("text", "") for chunk in (context_chunks or [])]).lower()

    indicators = {
        "has_specific_product": False,
        "has_specific_model": False,
        "has_specific_brand": False,
        "specific_terms": [],
        "context_relevance": "low",
    }

    specific_patterns = [
        r"\b(splendor|hero|honda|yamaha|bajaj)\s+(bike|motorcycle)",
        r"\b(iphone|samsung|oneplus|xiaomi)\s+(phone|mobile)",
        r"\b(maruti|hyundai|tata|mahindra)\s+(car|vehicle)",
        r"\b(model|variant|version)\s+\w+",
        r"\b\w+\s+(series|edition|pro|max|plus)\b",
    ]
    for pattern in specific_patterns:
        matches = re.findall(pattern, question_lower)
        if matches:
            indicators["has_specific_product"] = True
            indicators["specific_terms"].extend(matches)

    question_words = set(re.findall(r"\w+", question_lower))
    context_words = set(re.findall(r"\w+", context_text))
    total_question_words = len(question_words)
    overlap = len(question_words & context_words)
    if total_question_words > 0:
        ratio = overlap / total_question_words
        if ratio > 0.6:
            indicators["context_relevance"] = "high"
        elif ratio > 0.3:
            indicators["context_relevance"] = "medium"

    for term in indicators["specific_terms"]:
        term_words = term if isinstance(term, tuple) else [term]
        if any(word in context_text for word in term_words):
            indicators["has_specific_model"] = True
            break

    return indicators


def detect_completely_false_information(question: str, context_chunks: List[Dict]) -> Dict[str, object]:
    """Check if the context has obviously wrong facts like 'Paris is capital of UK'."""
    context_text = " ".join([chunk.get("text", "") for chunk in (context_chunks or [])]).lower()
    question_lower = (question or "").lower()

    false_patterns = {
        "geography": [
            (r"capital.*france.*london", "The capital of France is Paris, not London"),
            (r"capital.*india.*mumbai", "The capital of India is New Delhi, not Mumbai"),
            (r"capital.*usa.*new york", "The capital of USA is Washington D.C., not New York"),
        ],
        "science": [
            (r"water.*boil.*50|water.*boil.*30", "Water boils at 100°C (212°F) at sea level, not at lower temperatures"),
            (r"earth.*flat", "The Earth is spherical, not flat"),
            (r"sun.*revolve.*earth", "The Earth revolves around the Sun, not vice versa"),
        ],
        "basic_facts": [
            (r"7.*continent.*8|8.*continent.*7", "There are 7 continents, not 8"),
            (r"24.*hour.*25|25.*hour.*24", "A day has 24 hours, not 25"),
        ],
    }

    has_false_info = False
    categories: List[str] = []
    corrections: List[str] = []

    for category, patterns in false_patterns.items():
        for pattern, correction in patterns:
            if re.search(pattern, context_text) and any(word in question_lower for word in pattern.split('.*')):
                has_false_info = True
                categories.append(category)
                corrections.append(correction)

    return {"has_false_info": has_false_info, "false_categories": categories, "corrections": corrections}


# Prompt builders

def build_general_prompt(question: str, context_chunks: List[Dict]) -> str:
    if detect_irrelevant_question(question, context_chunks):
        return f"""
You are a helpful assistant. The user has asked an inappropriate or irrelevant question.

QUESTION: {question}

Respond with strong, clear language about the inappropriateness. Respond in JSON format:
{{ "answer": "This question is inappropriate and cannot be answered. Please ask relevant questions about the document content." }}
"""
    specificity_analysis = analyze_context_specificity(question, context_chunks)
    false_info_analysis = detect_completely_false_information(question, context_chunks)
    context = "\n---\n".join([c["text"] for c in context_chunks])
    if false_info_analysis["has_false_info"]:
        strategy = "correct_false_info"
    elif specificity_analysis["has_specific_product"] and specificity_analysis["context_relevance"] == "high":
        strategy = "prefer_context"
    else:
        strategy = "general_knowledge"
    strategy_instructions = {
        "correct_false_info": """
STRATEGY: CORRECT FALSE INFORMATION
- The context contains factually incorrect information
- Use your general knowledge to provide the correct answer
- Clearly mention that you're correcting false information from the document
- Explain why the context information is incorrect
""",
        "prefer_context": """
STRATEGY: PREFER SPECIFIC CONTEXT
- The context contains specific information relevant to your question
- Use the context information as it provides specific details
- Supplement with general knowledge only if context is incomplete
- Mention that you're using specific information from the document
""",
        "general_knowledge": """
STRATEGY: GENERAL KNOWLEDGE PRIMARY
- Use your general knowledge as the primary source
- Reference context only if it provides additional specific details
- Provide comprehensive general knowledge answer
""",
    }
    return f"""
You are a helpful assistant with advanced context analysis capabilities.

{strategy_instructions[strategy]}

CRITICAL REQUIREMENTS:
- Provide informative responses (up to 3 lines maximum)
- Use plain, everyday language instead of technical jargon
- For specific products/models mentioned in context, prefer context information
- For basic facts, correct any false information using general knowledge
- Always mention your information source (context vs general knowledge)
- Be direct and comprehensive within the 3-line limit

CONTEXT (analyze for specificity and accuracy):
{context}

QUESTION: {question}

Provide a comprehensive answer using the appropriate strategy. Always mention your information source. Respond in JSON format:
{{ "answer": "..." }}
"""


def build_insurance_prompt(question: str, context_chunks: List[Dict]) -> str:
    if detect_irrelevant_question(question, context_chunks):
        return f"""
You are an insurance assistant. The user has asked an inappropriate question.

QUESTION: {question}

Respond with strong language about fraud and illegal activities. Respond in JSON format:
{{ "answer": "Fraud is illegal and voids policy coverage. Such activities are strictly prohibited and will result in claim denial and policy cancellation." }}
"""
    context = "\n---\n".join([c["text"] for c in context_chunks])
    question_lower = (question or "").lower()
    cross_check_instructions = ""
    if any(condition in question_lower for condition in ['arthritis', 'abortion', 'hydrocele']):
        cross_check_instructions = """
SPECIAL CROSS-CHECKING REQUIREMENTS:
- For ARTHRITIS: Check both "Permanent Exclusions" and "Waiting Periods" sections
- For ABORTION: Check "Permanent Exclusions" and "Maternity Benefits" sections
- For HYDROCELE: Check "Waiting Periods" table specifically
- Always mention the specific section where information is found
- If condition appears in multiple sections, reference all relevant sections
"""
    return f"""
You are a knowledgeable insurance assistant explaining policy details in simple terms.

CRITICAL REQUIREMENTS:
- Provide informative responses (up to 3 lines maximum)
- Use plain, everyday language instead of technical insurance jargon
- Include essential information like key numbers, conditions, limits, and exclusions
- Extract exact figures (₹ amounts, percentages, waiting periods) from the document
- Cross-check multiple sections for comprehensive coverage information
- Be direct and comprehensive within the 3-line limit
- Never add information not found in the context
- Do not go outside of the provided context

{cross_check_instructions}

CONTEXT FROM POLICY DOCUMENT:
{context}

QUESTION: {question}

Provide a comprehensive answer with exact figures, conditions, and section references. Respond in JSON format:
{{ "answer": "..." }}
"""


def build_health_policy_prompt(question: str, context_chunks: List[Dict]) -> str:
    if detect_irrelevant_question(question, context_chunks):
        return f"""
You are a health insurance specialist. The user has asked an inappropriate question.

QUESTION: {question}

Respond with strong language about fraud and illegal activities. Respond in JSON format:
{{ "answer": "Fraud is illegal and voids policy coverage. Such activities violate insurance regulations and will result in immediate policy termination." }}
"""
    context = "\n---\n".join([c["text"] for c in context_chunks])
    question_lower = (question or "").lower()
    cross_check_instructions = ""
    if any(cond in question_lower for cond in ['arthritis', 'abortion', 'hydrocele', 'maternity', 'pregnancy']):
        cross_check_instructions = """
SPECIAL HEALTH POLICY CROSS-CHECKING:
- For ARTHRITIS: Check "Permanent Exclusions" and "Waiting Periods" for joint conditions
- For ABORTION: Cross-reference "Maternity Benefits" and "Permanent Exclusions" sections
- For HYDROCELE: Verify coverage in "Waiting Periods" table and surgical benefits
- For MATERNITY: Check exclusions like IVF, surrogacy in maternity benefits section
- Always specify which section contains the information
- Note any contradictions between different sections
"""
    return f"""
You are a health insurance specialist explaining medical coverage in simple terms.

CRITICAL REQUIREMENTS:
- Provide informative responses (up to 3 lines maximum)
- Use plain language instead of medical/insurance jargon
- Highlight key coverage amounts, waiting periods, and exclusions
- Extract exact figures (₹ amounts, percentages, waiting periods) from the document
- Note condition-based exclusions (e.g., IVF, surrogacy in maternity)
- Cross-check multiple sections for complete coverage information
- Be direct and comprehensive within the 3-line limit
- Never add information not found in the context
- Do not go outside of the provided context

{cross_check_instructions}

CONTEXT FROM HEALTH POLICY:
{context}

QUESTION: {question}

Provide a comprehensive answer with exact figures, exclusions, and section references. Respond in JSON format:
{{ "answer": "..." }}
"""


def build_history_prompt(question: str, context_chunks: List[Dict]) -> str:
    if detect_irrelevant_question(question, context_chunks):
        return f"""
You are a history assistant. The user has asked an inappropriate question.

QUESTION: {question}

Respond with clear guidance to stay on topic. Respond in JSON format:
{{ "answer": "Please ask relevant questions about historical events, timelines, or figures found in the provided context." }}
"""
    context = "\n---\n".join([c["text"] for c in context_chunks])
    return f"""
You are a history assistant explaining historical events, timelines, and figures in simple terms.

CRITICAL REQUIREMENTS:
- Provide informative responses (up to 3 lines maximum)
- Use everyday language; avoid jargon
- Include key dates, places, people, and outcomes when relevant
- Be precise and stick to facts present in the context
- Never add information not found in the context
- Do not go outside of the provided context

CONTEXT FROM HISTORICAL SOURCES:
{context}

QUESTION: {question}

Provide a concise, fact-based answer with key dates or figures when applicable. Respond in JSON format:
{{ "answer": "..." }}
"""


def build_science_prompt(question: str, context_chunks: List[Dict]) -> str:
    if detect_irrelevant_question(question, context_chunks):
        return f"""
You are a science educator. The user has asked an inappropriate question.

QUESTION: {question}

Respond with guidance to stay on scientific topics. Respond in JSON format:
{{ "answer": "Please ask relevant questions about scientific concepts found in the provided context." }}
"""
    context = "\n---\n".join([c["text"] for c in context_chunks])
    return f"""
You are a science educator explaining concepts across physics, chemistry, and biology in simple terms.

CRITICAL REQUIREMENTS:
- Provide informative responses (up to 3 lines maximum)
- Use everyday examples
- Include key formulas, laws, or definitions when relevant
- Provide real-world applications when helpful
- Be comprehensive within the 3-line limit
- Never add information not found in the context
- Do not go outside of the provided context

CONTEXT FROM SCIENCE RESOURCES:
{context}

QUESTION: {question}

Provide a concise answer with examples or applications when helpful. Respond in JSON format:
{{ "answer": "..." }}
"""

# Adaptive prompt selection

def get_adaptive_prompt(client, question: str, context_chunks: List[Dict]) -> str:
    """Select and build the appropriate prompt based on query classification."""
    query_category = _classify_user_query_util(client, question)
    prompt_functions = {
        "general": build_general_prompt,
        "insurance": build_insurance_prompt,
        "health_policy": build_health_policy_prompt,
        "history": build_history_prompt,
        "science": build_science_prompt,
    }
    prompt_function = prompt_functions.get(query_category, build_general_prompt)
    return prompt_function(question, context_chunks)


