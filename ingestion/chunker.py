"""Semantic chunking strategy combined with size and overlap constraints."""
import hashlib
import re
from typing import List
from schemas import Chunk, Document
from sentence_transformers import SentenceTransformer
import numpy as np
from config import settings

# Initialize a lightweight sentence embedding model for semantic distance calculation
_embedding_model = None

def get_chunking_model():
    global _embedding_model
    if _embedding_model is None:
        # Uses a fast, lightweight sentence transformer model for similarity checks
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

def split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences supporting English and multilingual punctuation."""
    # Split by standard sentence endings (. ! ? \n\n)
    sentence_endings = re.compile(r'(?<=[.!?])\s+|\n\s*\n')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_document(doc: Document, chunk_size: int = 500, chunk_overlap: int = 50, similarity_threshold_percentile: float = 85.0) -> List[Chunk]:
    """
    Chunks a document using semantic similarity boundaries first, 
    then enforces maximum chunk_size and chunk_overlap constraints.
    """
    text = doc.text.strip()
    if not text:
        return []

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    # If document is very short, return as a single chunk
    if len(sentences) <= 1:
        chunk_id = f"{doc.source}|{doc.page}|0|{hashlib.sha256(text.encode()).hexdigest()[:12]}"
        return [Chunk(
            chunk_id=chunk_id,
            text=text,
            source=doc.source,
            source_type=doc.source_type or "document",
            page=doc.page,
            title=doc.title,
            language="mixed_or_unknown",
            extraction_method=doc.extraction_method or "standard",
            metadata=doc.metadata or {}
        )]

    # Step 1: Compute embeddings for sentences to find semantic breakpoints
    model = get_chunking_model()
    embeddings = model.encode(sentences, show_progress_bar=False)
    
    # Calculate cosine similarities between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        vec1 = embeddings[i]
        vec2 = embeddings[i+1]
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 > 0 and norm2 > 0:
            sim = np.dot(vec1, vec2) / (norm1 * norm2)
        else:
            sim = 0.0
        similarities.append(sim)

    # Determine breakpoint threshold based on percentile of distance drops
    if similarities:
        threshold = np.percentile(similarities, 100 - similarity_threshold_percentile)
    else:
        threshold = 0.0

    # Step 2: Form initial semantic clusters
    semantic_chunks = []
    current_cluster = [sentences[0]]
    
    for i, sim in enumerate(similarities):
        if sim < threshold:
            # Semantic shift detected: close current cluster and start a new one
            semantic_chunks.append(" ".join(current_cluster))
            current_cluster = [sentences[i+1]]
        else:
            current_cluster.append(sentences[i+1])
    
    if current_cluster:
        semantic_chunks.append(" ".join(current_cluster))

    # Step 3: Enforce chunk_size and chunk_overlap constraints on semantic chunks
    final_chunks = []
    
    for s_idx, cluster_text in enumerate(semantic_chunks):
        words = cluster_text.split()
        
        # If semantic chunk fits within chunk_size, keep it intact
        if len(words) <= chunk_size:
            chunk_id = f"{doc.source}|{doc.page}|{s_idx}|{hashlib.sha256(cluster_text.encode()).hexdigest()[:12]}"
            final_chunks.append(Chunk(
                chunk_id=chunk_id,
                text=cluster_text,
                source=doc.source,
                source_type=doc.source_type or "document",
                page=doc.page,
                title=doc.title,
                language="mixed_or_unknown",
                extraction_method=doc.extraction_method or "standard",
                metadata=doc.metadata or {}
            ))
        else:
            # Otherwise, apply sliding window chunk_size and chunk_overlap
            start = 0
            sub_idx = 0
            while start < len(words):
                end = start + chunk_size
                part_words = words[start:end]
                part = " ".join(part_words)
                
                chunk_id = f"{doc.source}|{doc.page}|{s_idx}_{sub_idx}|{hashlib.sha256(part.encode()).hexdigest()[:12]}"
                final_chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=part,
                    source=doc.source,
                    source_type=doc.source_type or "document",
                    page=doc.page,
                    title=doc.title,
                    language="mixed_or_unknown",
                    extraction_method=doc.extraction_method or "standard",
                    metadata=doc.metadata or {}
                ))
                
                start += chunk_size - chunk_overlap
                sub_idx += 1

    return final_chunks