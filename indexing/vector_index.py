"""Persistent FAISS index and chunk metadata store."""
import json
from pathlib import Path
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from schemas import Chunk
from config import settings

logger = logging.getLogger(__name__)

class VectorIndex:
    def __init__(self, directory: str | Path = settings.index_dir):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "dense.faiss"
        self.chunks_path = self.directory / "chunks.jsonl"
        
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.model = SentenceTransformer(settings.embedding_model)
        
        self.index = None
        self.chunks: list[Chunk] = []
        self.load()
        logger.info(f"Index ready: {len(self.chunks)} chunk(s) loaded")

    def load(self):
        if self.index_path.exists() and self.chunks_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with self.chunks_path.open("r", encoding="utf-8") as handle:
                self.chunks = [Chunk(**json.loads(line)) for line in handle if line.strip()]

    def add(self, chunks: list[Chunk], progress_callback=None):
        known = {c.chunk_id for c in self.chunks}
        new = [c for c in chunks if c.chunk_id not in known]
        if not new:
            return 0
            
        if progress_callback:
            progress_callback(f"Embedding {len(new)} new chunk(s)...")

        batch_size = 32
        encoded_batches = []
        total_batches = (len(new) + batch_size - 1) // batch_size
        
        for batch_number, start in enumerate(range(0, len(new), batch_size), start=1):
            batch = new[start:start + batch_size]
            if progress_callback:
                progress_callback(f"Embedding batch {batch_number}/{total_batches} ({len(batch)} chunk(s))")
                
            vectors_batch = self.model.encode(
                [c.text for c in batch], normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )
            encoded_batches.append(np.asarray(vectors_batch, dtype="float32"))
            
        vectors = np.concatenate(encoded_batches, axis=0)
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        if self.index.d != vectors.shape[1]:
            raise ValueError("Embedding dimensions changed; rebuild the index with one embedding model.")
            
        self.index.add(vectors)
        self.chunks.extend(new)
        self.save()
        return len(new)

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
            tmp = self.chunks_path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                for chunk in self.chunks:
                    handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            tmp.replace(self.chunks_path)

    def search(self, query: str, k: int = 8) -> list[tuple[Chunk, float]]:
        if self.index is None or not self.chunks:
            return []
        vector = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, ids = self.index.search(vector, min(k, len(self.chunks)))
        return [(self.chunks[int(i)], float(score)) for i, score in zip(ids[0], scores[0]) if i >= 0]