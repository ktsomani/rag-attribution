import json
import math
import asyncio
from pathlib import Path
from fastapi.concurrency import run_in_threadpool
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
    dcg = 0.0
    idcg = 0.0
    for i, d in enumerate(top_k):
        rel = 1.0 if d in gold_ids else 0.0
        dcg += rel / math.log2(i + 2)
    for i in range(min(len(gold_ids), k)):
        idcg += 1.0 / math.log2(i + 2)
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, mrr, ndcg

async def evaluate_query(pipeline, q, k, mode):
    retrieved = await run_in_threadpool(pipeline.retrieve, q["question"], k=k, mode=mode)
    return calculate_metrics(retrieved, q["gold_doc_ids"], k)

async def run_retrieval_eval():
    golden_set_path = Path("golden_set/data/questions.jsonl")
    questions = []
    with open(golden_set_path, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
            
    print("Initializing pipeline...")
    pipeline = LocalPipeline()
    modes = ["dense", "lexical", "hybrid_rerank"]
    k = 10 
    
    print(f"\nEvaluating Retrieval Modes at k={k} (SciFact Test Split)...\n")
    print(f"{'Mode':<15} | {'Recall@10':<10} | {'MRR':<10} | {'nDCG@10':<10}")
    print("-" * 55)

    total_queries = 0

    for mode in modes:
        answerable_qs = [q for q in questions if q["stratum"] == "answerable"]
        
        # THE ASYNC FIX: Run all queries concurrently
        tasks = [evaluate_query(pipeline, q, k, mode) for q in answerable_qs]
        results = await asyncio.gather(*tasks)
        
        total_recall = sum(r[0] for r in results)
        total_mrr = sum(r[1] for r in results)
        total_ndcg = sum(r[2] for r in results)
        count = len(results)
        total_queries += count
            
        if count > 0:
            print(f"{mode:<15} | {total_recall/count:<10.3f} | {total_mrr/count:<10.3f} | {total_ndcg/count:<10.3f}")

    # Cost tracking
    total_input_tokens = total_queries * 100
    total_output_tokens = total_queries * 50
    costs = calculate_run_cost(total_input_tokens, total_output_tokens)
    
    print("\n" + "="*55)
    print("💰 EVALUATION RUN COST REPORT")
    print("="*55)
    print(f"Total Queries Evaluated: {total_queries}")
    print(f"Estimated Total Cost:    ${costs['total_cost_usd']}")
    print("="*55 + "\n")

if __name__ == "__main__":
    asyncio.run(run_retrieval_eval())