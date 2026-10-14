"""Small, dependency-light retrieval metrics for a manually labelled test set."""
def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)

def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, item in enumerate(retrieved_ids, 1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0
