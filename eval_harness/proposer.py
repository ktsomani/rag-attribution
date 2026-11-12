import os
from openai import OpenAI
from fastapi.concurrency import run_in_threadpool

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)
MODEL_NAME = "openai/gpt-oss-20b"

def _sync_propose_harness_fix(failure_mix: dict, current_code: str) -> str:
    print(f"🤖 Proposer ({MODEL_NAME}) is analyzing the failure mix via Groq API...")
    prompt = f"""
    You are an expert AI Engineer optimizing a RAG pipeline. 
    
    Here is the current Failure Mix from our evaluation harness:
    {failure_mix}
    
    Here is the current Python code for the pipeline (real_adapter.py):
    {current_code}
    
    Your task: Rewrite the `retrieve` function to fix the highest failure rate.
    - If 'rank_inversion' is high, implement a reranker or adjust the top-k.
    - If 'retrieval_gap' is high, adjust the BM25 tokenization or dense search.
    
    CRITICAL INSTRUCTION: Return ONLY the raw Python code. Do not include markdown formatting (like ```python). Do not include any conversational text.
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a senior AI engineer that outputs only raw Python code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        new_code = response.choices[0].message.content.strip()
        
        if new_code.startswith("```python"):
            new_code = new_code.replace("```python\n", "").replace("```", "")
            
        return new_code
    except Exception as e:
        print(f"Error calling Proposer: {e}")
        return current_code

async def propose_harness_fix(failure_mix: dict, current_code: str) -> str:
    """Async wrapper to prevent blocking the event loop."""
    return await run_in_threadpool(_sync_propose_harness_fix, failure_mix, current_code)