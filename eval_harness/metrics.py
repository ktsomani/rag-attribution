def calculate_recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """
    Recall@k: The fraction of gold documents found in the top-k retrieved.
    """
    if not gold_ids:
        return 0.0
    
    top_k_retrieved = retrieved_ids[:k]
    relevant_found = sum(1 for doc_id in gold_ids if doc_id in top_k_retrieved)
    return relevant_found / len(gold_ids)

def calculate_mrr_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """
    MRR@k: The reciprocal rank of the FIRST relevant document found in top-k.
    (If the first relevant doc is at rank 1, score is 1.0. Rank 2 = 0.5. Rank 3 = 0.33)
    """
    top_k_retrieved = retrieved_ids[:k]
    for rank, doc_id in enumerate(top_k_retrieved, start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    return 0.0