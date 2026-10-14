"""Environment-backed settings for the local prototype."""
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "")
    ocr_languages: str = os.getenv("OCR_LANGUAGES", "tam+eng")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "650"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "8"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "5"))
    data_dir: str = os.getenv("DATA_DIR", "data")
    index_dir: str = os.getenv("INDEX_DIR", "indexes")

settings = Settings()
