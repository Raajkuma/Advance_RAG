from schemas import SourceDocument
from ingestion.chunker import chunk_document

def test_chunker_keeps_text_and_metadata():
    source = SourceDocument(text="# Eligibility\n" + ("Tamil scheme eligibility amount documents. " * 100),
        source="demo.pdf", source_type="pdf", page=2, title="Demo", extraction_method="native_text")
    chunks = chunk_document(source, size=80, overlap=10)
    assert chunks
    assert all(chunk.source == "demo.pdf" and chunk.page == 2 for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)

def test_empty_text_returns_no_chunks():
    assert chunk_document(SourceDocument(text="", source="x", source_type="txt")) == []
