import json
from pathlib import Path
from eval_harness.real_adapter import LocalPipeline
from eval_harness.metrics import calculate_recall_at_k, calculate_mrr_at_k

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
    k = 3
    
    modes = ["dense", "lexical", "hybrid_rerank"]
    
    print(f"\nEvaluating Retrieval Modes at k={k}...\n")
    print(f"{'Mode':<15} | {'Recall@k':<10} | {'MRR@k':<10}")
    print("-" * 40)

    for mode in modes:
        total_recall = 0.0
        total_mrr = 0.0
        answerable_count = 0
        
        for q in questions:
            if q.get("stratum") != "answerable":
                continue
                
            answerable_count += 1
            gold_ids = q.get("gold_doc_ids", [])
            
            # THIS IS WHERE IT WAS FAILING BEFORE
            retrieved_ids = pipeline.retrieve(
                query=q["question"], 
                k=k, 
                mode=mode
            )
            
            total_recall += calculate_recall_at_k(retrieved_ids, gold_ids, k)
            total_mrr += calculate_mrr_at_k(retrieved_ids, gold_ids, k)
            
        if answerable_count > 0:
            avg_recall = total_recall / answerable_count
            avg_mrr = total_mrr / answerable_count
            print(f"{mode:<15} | {avg_recall:<10.2f} | {avg_mrr:<10.2f}")

if __name__ == "__main__":
    run_retrieval_eval()