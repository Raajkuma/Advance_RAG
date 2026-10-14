"""Reciprocal Rank Fusion with stable chunk-id deduplication."""
from schemas import Chunk
from observability.tracing import traceable

@traceable("reciprocal_rank_fusion")
def reciprocal_rank_fusion(rankings: list[list[tuple[Chunk, float]]], k: int = 60, limit: int = 20):
    scores, chunks = {}, {}
    for ranking in rankings:
        for rank, (chunk, _score) in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[chunk.chunk_id] = chunk
    ordered = sorted(scores, key=scores.get, reverse=True)[:limit]
    return [(chunks[cid], scores[cid]) for cid in ordered]
