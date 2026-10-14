"""Page-accounted PDF extraction; every page produces a manifest entry."""
import hashlib
from pathlib import Path
import fitz
from schemas import SourceDocument
from ingestion.ocr_engine import extract_page_text

def extract_pdf(path: str | Path, progress_callback=None) -> tuple[list[SourceDocument], list[dict]]:
    path = Path(path)
    docs, manifest = [], []
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with fitz.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            if progress_callback:
                progress_callback(f"Reading PDF page {page_index + 1}/{len(pdf)}: {path.name}")
            try:
                text, method, confidence = extract_page_text(page)
                status = "ok" if text.strip() else "empty_review_required"
                docs.append(SourceDocument(
                    text=text, source=str(path.resolve()), source_type="pdf", page=page_index + 1,
                    title=path.stem, extraction_method=method,
                    metadata={"sha256": digest, "page_count": len(pdf), "ocr_quality_estimate": round(confidence, 3)},
                ))
                manifest.append({"source": str(path), "sha256": digest, "page": page_index + 1,
                                "method": method, "quality_estimate": round(confidence, 3), "status": status,
                                "characters": len(text)})
            except Exception as exc:
                manifest.append({"source": str(path), "sha256": digest, "page": page_index + 1,
                                "method": "failed", "status": "error_review_required", "error": str(exc)})
                # Preserve the failed page in the audit trail rather than silently dropping it.
                docs.append(SourceDocument(text="", source=str(path.resolve()), source_type="pdf",
                    page=page_index + 1, title=path.stem, extraction_method="failed",
                    metadata={"sha256": digest, "extraction_error": str(exc)}))
    return docs, manifest
