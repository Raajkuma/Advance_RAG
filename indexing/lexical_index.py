"""BM25 lexical search, rebuilt only when the corpus changes in memory."""
import re
from rank_bm25 import BM25Okapi
from schemas import Chunk

def tokenize(text: str) -> list[str]:
    # Retain Tamil Unicode, English, acronyms, and numeric identifiers.
    return re.findall(r"[\w\u0B80-\u0BFF]+", text.casefold(), flags=re.UNICODE)

class LexicalIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None
    def search(self, query: str, k: int = 8) -> list[tuple[Chunk, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [(self.chunks[i], float(score)) for i, score in ranked if score > 0]
