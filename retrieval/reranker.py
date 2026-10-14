"""Cross-encoder reranking; lazy-loaded to avoid startup cost when unused."""
from config import settings

class Reranker:
    def __init__(self):
        self._model = None
        self.last_status = "not_run"
        self.last_error = ""
    def rerank(self, query, candidates, top_n=5):
        if not candidates:
            return []
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(settings.reranker_model)
            pairs = [(query, chunk.text) for chunk, _ in candidates]
            scores = self._model.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)[:top_n]
            self.last_status = "ok"
            return [(chunk, float(score)) for ((chunk, _), score) in ranked]
        except Exception as exc:
            # Graceful fallback to RRF order; expose the reason in the UI.
            self.last_status = "fallback"
            self.last_error = f"{type(exc).__name__}: {exc}"
            return candidates[:top_n]
