import json
import math
from pathlib import Path
from eval_harness.real_adapter import LocalPipeline
from eval_harness.cost_tracker import calculate_run_cost

def calculate_metrics(retrieved_ids, gold_ids, k):
    if not retrieved_ids or not gold_ids:
        return 0.0, 0.0, 0.0
        
    top_k = retrieved_ids[:k]
    recall = sum(1 for d in gold_ids if d in top_k) / len(gold_ids)
    
    mrr = 0.0
    for i, d in enumerate(top_k):
        if d in gold_ids:
            mrr = 1.0 / (i + 1)
            break
            
    # nDCG calculation
    dcg = 0.0
    idcg = 0.0
    for i, d in enumerate(top_k):
        rel = 1.0 if d in gold_ids else 0.0
        dcg += rel / math.log2(i + 2)
    for i in range(min(len(gold_ids), k)):
        idcg += 1.0 / math.log2(i + 2)
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return recall, mrr, ndcg

def run_retrieval_eval():
    # 1. Load Golden Set
    golden_set_path = Path("golden_set/data/questions.jsonl")
    questions = []
    with open(golden_set_path, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
            
    # 2. Initialize Pipeline
    print("Initializing pipeline (this may take a few seconds)...")
    pipeline = LocalPipeline()
    
    modes = ["dense", "lexical", "hybrid_rerank"]
    k = 10 
    
    print(f"\nEvaluating Retrieval Modes at k={k} (SciFact Test Split)...\n")
    print(f"{'Mode':<15} | {'Recall@10':<10} | {'MRR':<10} | {'nDCG@10':<10}")
    print("-" * 55)

    total_queries = 0

    for mode in modes:
        total_recall, total_mrr, total_ndcg = 0.0, 0.0, 0.0
        count = 0
        
        for q in questions:
            if q["stratum"] != "answerable": continue
            count += 1
            total_queries += 1
            
            retrieved = pipeline.retrieve(q["question"], k=k, mode=mode)
            r, m, n = calculate_metrics(retrieved, q["gold_doc_ids"], k)
            
            total_recall += r
            total_mrr += m
            total_ndcg += n
            
        if count > 0:
            print(f"{mode:<15} | {total_recall/count:<10.3f} | {total_mrr/count:<10.3f} | {total_ndcg/count:<10.3f}")

    # 3. Cost Tracking Calculation
    # We assume each query used ~100 input tokens (prompt + context) and ~50 output tokens (generation)
    # Note: This is an estimate for the RAG generation phase, since the retrieval phase itself doesn't use LLMs directly.
    avg_input_tokens_per_query = 100
    avg_output_tokens_per_query = 50
    
    total_input_tokens = total_queries * avg_input_tokens_per_query
    total_output_tokens = total_queries * avg_output_tokens_per_query
    
    costs = calculate_run_cost(total_input_tokens, total_output_tokens)
    
    print("\n" + "="*55)
    print("💰 EVALUATION RUN COST REPORT")
    print("="*55)
    print(f"Total Queries Evaluated: {total_queries}")
    print(f"Estimated Input Tokens:  {total_input_tokens}")
    print(f"Estimated Output Tokens: {total_output_tokens}")
    print(f"Estimated Total Cost:    ${costs['total_cost_usd']}")
    print("="*55 + "\n")

if __name__ == "__main__":
    run_retrieval_eval()