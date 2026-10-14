"""Local Tesseract OCR with optional Gemini vision fallback for poor pages."""
import base64
import io
import os
import re
import fitz
import pytesseract
from PIL import Image
from config import settings

if settings.tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

def _quality(text: str) -> float:
    if not text.strip():
        return 0.0
    chars = len(text)
    letters = sum(ch.isalpha() for ch in text)
    replacement = text.count("\ufffd")
    return max(0.0, min(1.0, (letters / max(chars, 1)) - replacement / max(chars, 1)))

def local_ocr(image: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(image, lang=settings.ocr_languages, config="--psm 3")
    except Exception as exc:
        # Keep extraction moving but expose a useful installation error to the caller.
        raise RuntimeError("Tesseract OCR failed. Install Tesseract and Tamil/English language data; see README.md. Original error: " + str(exc)) from exc

def gemini_ocr_fallback(image: Image.Image) -> str:
    """Optional fallback. Requires GOOGLE_API_KEY and google-genai package."""
    if not settings.google_api_key:
        return ""
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=settings.google_api_key)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        response = client.models.generate_content(
            model=settings.llm_model,
            contents=[
                "Transcribe all visible document text exactly. Preserve Tamil and English, line breaks, headings, table rows, amounts, dates, and identifiers. Do not summarize or infer missing text.",
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
            ],
        )
        return (response.text or "").strip()
    except Exception:
        return ""

def extract_page_text(page: fitz.Page) -> tuple[str, str, float]:
    text = page.get_text("text", sort=True) or ""
    # OCR only when native extraction is empty or suspiciously sparse.
    if len(re.sub(r"\s+", "", text)) >= 45:
        return text, "native_text", _quality(text)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    try:
        ocr_text = local_ocr(image)
    except Exception:
        # Do not discard even sparse native text just because OCR is unavailable.
        if text.strip():
            return text, "native_text_low_confidence_ocr_unavailable", _quality(text)
        raise
    method = "tesseract_ocr"
    if _quality(ocr_text) < 0.35 or len(ocr_text.strip()) < 40:
        fallback = gemini_ocr_fallback(image)
        if len(fallback) > len(ocr_text.strip()) and _quality(fallback) >= _quality(ocr_text):
            ocr_text, method = fallback, "gemini_vision_ocr"
    return ocr_text, method, _quality(ocr_text)
