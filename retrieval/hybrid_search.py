"""Hybrid search combining Dense vector retrieval and BM25 sparse retrieval in parallel."""
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
from indexing.vector_index import VectorIndex
from schemas import Chunk

class HybridRetriever:
    def __init__(self, vector_index: VectorIndex):
        self.vector_index = vector_index
        self.chunks = vector_index.chunks
        # Pre-tokenize corpus for BM25 keyword search
        tokenized_corpus = [chunk.text.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def search(self, query: str, k: int = 8, progress_callback=None) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []

        if progress_callback:
            progress_callback("Running parallel hybrid retrieval (Dense + BM25)...")

        # Define individual search worker functions
        def run_dense():
            return self.vector_index.search(query, k=k * 2)

        def run_bm25():
            if not self.bm25:
                return []
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k * 2]
            return [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]

        # Execute dense vector search and BM25 concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(run_dense)
            bm25_future = executor.submit(run_bm25)
            
            dense_results = dense_future.result()
            bm25_results = bm25_future.result()

        if progress_callback:
            progress_callback("Merging results with Reciprocal Rank Fusion (RRF)...")

        # Reciprocal Rank Fusion (RRF) to combine results
        rrf_scores = {}
        k_const = 60
        
        for rank, (chunk, _) in enumerate(dense_results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (k_const + rank + 1))
            
        for rank, (chunk, _) in enumerate(bm25_results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (k_const + rank + 1))

        # Map chunk IDs back to Chunk objects
        chunk_map = {chunk.chunk_id: chunk for chunk in self.chunks}
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        
        final_results = [(chunk_map[cid], rrf_scores[cid]) for cid in sorted_chunk_ids[:k] if cid in chunk_map]
        
        if progress_callback:
            progress_callback(f"Retrieval complete: found {len(final_results)} fused candidate(s)")
            
        return final_results