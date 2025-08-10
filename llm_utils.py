import json
from openai import OpenAI


def classify_user_query(client: OpenAI, question: str) -> str:
    """Classify user query into categories using gpt-4.1-mini."""
    classification_prompt = f"""
You are a query classifier. Classify the following user question into one of these categories:

1. general - General knowledge questions not specific to any domain
2. insurance - Insurance policy, claims, coverage, premium related questions
3. health_policy - Health insurance, medical coverage, health benefits related questions
4. history - Historical events, timelines, figures, places related questions
5. science - General science (physics, chemistry, biology), concepts, laws, formulas

Question: {question}

Respond with only the category name (general, insurance, health_policy, history, or science).
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0.1,
            max_tokens=50,
        )
        classification = response.choices[0].message.content.strip().lower()
        valid = ["general", "insurance", "health_policy", "history", "science"]
        return classification if classification in valid else "general"
    except Exception:
        return "general"


def call_gpt_fast(client: OpenAI, prompt: str) -> str:
    """Call a fast, cost-effective model and expect JSON with an 'answer' key."""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600,
            top_p=0.1,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        try:
            parsed = json.loads(content)
            answer = parsed.get("answer")
            if answer and answer.strip():
                return answer
        except json.JSONDecodeError:
            if '{"answer":' in content:
                try:
                    start = content.find('{"answer":')
                    end = content.rfind('}') + 1
                    json_str = content[start:end]
                    return json.loads(json_str).get("answer", "Not found in document.")
                except Exception:
                    pass
        # Last resort regex would be in original code; we just return fallback
        return "Not found in document."
    except Exception:
        return "Not found in document."

