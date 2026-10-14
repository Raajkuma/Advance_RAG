"""Parsers for PDF, plain text, Markdown, and URLs."""
from pathlib import Path
import hashlib
from schemas import SourceDocument
from ingestion.pdf_extractor import extract_pdf
from ingestion.web_loader import load_url

def parse_file(path: str | Path, progress_callback=None) -> tuple[list[SourceDocument], list[dict]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path, progress_callback=progress_callback)
    if suffix not in {".txt", ".md", ".markdown"}:
        raise ValueError(f"Unsupported file type: {suffix}")
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    doc = SourceDocument(text=text, source=str(path.resolve()), source_type=suffix.lstrip("."),
        title=path.stem, extraction_method="utf8_decode", metadata={"sha256": hashlib.sha256(raw).hexdigest()})
    return [doc], [{"source": str(path), "status": "ok" if text.strip() else "empty_review_required", "characters": len(text)}]

def parse_url(url: str) -> SourceDocument:
    return load_url(url)
