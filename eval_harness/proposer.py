import os
from openai import OpenAI

# Initialize the client to point to Groq (FREE & Fast)
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY")   
)

# Groq's fastest and smartest open-source model
MODEL_NAME = "openai/gpt-oss-20b"

def propose_harness_fix(failure_mix: dict, current_code: str) -> str:
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
        
        # Clean up any markdown formatting the LLM might have hallucinated
        if new_code.startswith("```python"):
            new_code = new_code.replace("```python\n", "").replace("```", "")
            
        return new_code
    except Exception as e:
        print(f"Error calling Proposer: {e}")
        return current_code

def run_auto_refinement():
    # 1. Read the current harness code
    with open("eval_harness/real_adapter.py", "r") as f:
        code = f.read()
        
    # 2. Simulated Execution Report (High Rank Inversion)
    current_failures = {
        "retrieval_gap": 0.05,
        "rank_inversion": 0.35, # HIGH! The right docs are there, but ranked poorly.
        "evidence_under_use": 0.10
    }
    
    print("📊 Execution Report:")
    for stage, rate in current_failures.items():
        print(f"  - {stage}: {rate*100:.1f}%")
        
    # 3. Ask the Proposer to rewrite the harness
    new_code = propose_harness_fix(current_failures, code)
    
    # 4. Save the new harness
    with open("eval_harness/real_adapter_v2.py", "w") as f:
        f.write(new_code)
        
    print("\n✅ New harness generated at eval_harness/real_adapter_v2.py")
    print("Review the file to see how the Proposer fixed the Rank Inversion issue!")

if __name__ == "__main__":
    run_auto_refinement()