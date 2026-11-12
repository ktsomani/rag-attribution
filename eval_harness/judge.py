import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def grade_answer_with_llm(question: str, context: str, answer: str) -> dict:
    """
    Uses an LLM to grade the faithfulness of an answer based on the retrieved context.
    Returns a JSON object with a score (1-5) and reasoning.
    """
    prompt = f"""
    You are an impartial judge evaluating a RAG system.
    
    Question: {question}
    Retrieved Context: {context}
    Generated Answer: {answer}
    
    Does the generated answer faithfully use the context to answer the question?
    Reply in valid JSON format only:
    {{
        "score": <1 to 5, where 5 is perfectly faithful and 1 is a hallucination>,
        "reasoning": "<one sentence explanation>"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"score": 0, "reasoning": f"Judge error: {str(e)}"}

if __name__ == "__main__":
    print("--- Test 1: Faithful Answer ---")
    good_result = grade_answer_with_llm(
        question="What is the capital of France?",
        context="Paris is the capital of France. It is known for the Eiffel Tower.",
        answer="The capital of France is Paris."
    )
    print(json.dumps(good_result, indent=2))

    print("\n--- Test 2: Hallucination ---")
    bad_result = grade_answer_with_llm(
        question="What is the capital of France?",
        context="Paris is the capital of France. It is known for the Eiffel Tower.",
        answer="The capital of France is London, which is famous for the Eiffel Tower."
    )
    print(json.dumps(bad_result, indent=2))