from pathlib import Path
from datetime import datetime
import ast
import json
import re
import shutil
import time
import traceback

import streamlit as st
from config import settings
from ingestion.document_parser import parse_file, parse_url
from ingestion.chunker import chunk_document
from indexing.vector_index import VectorIndex
from retrieval.hybrid_search import HybridRetriever
from retrieval.cache import RAGCache
from retrieval.query_enhancer import enhance_query
from generation.answer_generator import generate_answer

st.set_page_config(page_title="Government Agricultural Scheme RAGBot", page_icon="📄", layout="wide")
st.title("Government Agricultural Scheme RAGBot")
st.caption("Tamil + English · Digital/scanned PDFs · TXT/Markdown · URLs · Local Hybrid Retrieval")


def add_event(message: str, event_log: list[str], status_container=None):
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"`{stamp}` · {message}"
    event_log.append(line)
    if status_container is not None:
        status_container.markdown(line)


@st.cache_resource(show_spinner=False)
def get_vector_index():
    return VectorIndex(settings.index_dir)

@st.cache_resource(show_spinner=False)
def get_rag_cache():
    return RAGCache()


if "last_backend_events" not in st.session_state:
    st.session_state.last_backend_events = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

rag_cache = get_rag_cache()

with st.status("Starting backend: loading embedding model and local index…", expanded=True) as startup_status:
    startup_events = []
    try:
        add_event(f"Embedding model configured: `{settings.embedding_model}`", startup_events, startup_status)
        vector = get_vector_index()
        add_event(f"Backend ready · {len(vector.chunks)} indexed chunk(s)", startup_events, startup_status)
        startup_status.update(label="Backend ready", state="complete", expanded=False)
    except Exception as exc:
        vector = None
        add_event(f"Backend initialization failed: {type(exc).__name__}: {exc}", startup_events, startup_status)
        startup_status.update(label="Backend initialization failed", state="error", expanded=True)
        st.error("The embedding model or index could not be initialized.")
        st.exception(exc)
st.session_state.last_backend_events = startup_events

with st.sidebar:
    st.header("Add knowledge")
    uploads = st.file_uploader("Upload PDF, TXT, or Markdown", type=["pdf", "txt", "md", "markdown"], accept_multiple_files=True)
    url = st.text_input("Or ingest a webpage URL", placeholder="https://www.example.gov.in/scheme")
    if st.button("Ingest selected sources", type="primary", disabled=vector is None):
        event_log = []
        docs, manifests = [], []
        started = time.perf_counter()
        ingest_status = None
        try:
            with st.status("Ingestion pipeline running…", expanded=True) as ingest_status:
                progress_bar = st.progress(0, text="Preparing sources…")
                total_sources = len(uploads or []) + (1 if url.strip() else 0)
                if total_sources == 0:
                    add_event("No files or URL selected", event_log, ingest_status)
                    progress_bar.empty()
                    ingest_status.update(label="Nothing to ingest", state="complete")
                else:
                    source_counter = 0
                    for uploaded in uploads or []:
                        source_counter += 1
                        safe_name = Path(uploaded.name).name
                        target = Path(settings.data_dir) / "originals" / safe_name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(uploaded.getvalue())
                        add_event(f"[{source_counter}/{total_sources}] Processing file: `{safe_name}`", event_log, ingest_status)

                        def pdf_progress(message):
                            match = re.search(r"Reading PDF page (\d+)/(\d+)", message)
                            if match:
                                cur_page, total_pages = int(match.group(1)), int(match.group(2))
                                pct = 0.05 + (0.45 * (cur_page / max(total_pages, 1)) * (source_counter / total_sources))
                                progress_bar.progress(min(pct, 0.5), text=f"Reading PDF: page {cur_page}/{total_pages} ({safe_name})")
                            else:
                                add_event(message, event_log)

                        parsed, manifest = parse_file(target, progress_callback=pdf_progress)
                        docs.extend(parsed)
                        manifests.extend(manifest)

                    if url.strip():
                        source_counter += 1
                        add_event(f"[{source_counter}/{total_sources}] Fetching webpage...", event_log, ingest_status)
                        docs.append(parse_url(url.strip()))
                        manifests.append({"source": url.strip(), "status": "ok", "method": "web_extraction"})
                        progress_bar.progress(0.5, text="Webpage extraction finished")

                    progress_bar.progress(0.55, text="Creating chunks from extracted content…")
                    chunks = []
                    for doc in docs:
                        chunks.extend(chunk_document(doc, settings.chunk_size, settings.chunk_overlap))
                    add_event(f"Chunking finished: {len(chunks)} chunk(s) created", event_log, ingest_status)
                    progress_bar.progress(0.65, text=f"Chunks created ({len(chunks)}). Preparing embeddings…")

                    def index_progress(message):
                        match = re.search(r"Embedding batch (\d+)/(\d+)", message)
                        if match:
                            curr_batch, total_batches = int(match.group(1)), int(match.group(2))
                            batch_pct = 0.65 + (0.3 * (curr_batch / max(total_batches, 1)))
                            progress_bar.progress(min(batch_pct, 0.95), text=f"Embedding in progress... batch {curr_batch}/{total_batches}")
                        else:
                            add_event(message, event_log)

                    added = vector.add(chunks, progress_callback=index_progress)
                    progress_bar.progress(1.0, text="Ingestion complete!")
                    ingest_status.update(label="Ingestion completed successfully", state="complete", expanded=True)
                    st.success(f"Processed {len(docs)} source unit(s), created {len(chunks)} chunks, added {added} new chunks.")
                    st.session_state.last_backend_events = event_log
                    rag_cache.clear()
        except Exception as exc:
            if ingest_status is not None:
                ingest_status.update(label="Ingestion failed", state="error", expanded=True)
            add_event(f"Ingestion failed: {type(exc).__name__}: {exc}", event_log)
            st.session_state.last_backend_events = event_log
            st.error("Ingestion failed.")
            st.exception(exc)

    st.divider()
    st.metric("Indexed chunks", len(vector.chunks) if vector is not None else "Unavailable")
    
    st.subheader("Embedded Sources")
    originals_dir = Path(settings.data_dir) / "originals"
    if originals_dir.exists():
        embedded_files = [f.name for f in originals_dir.iterdir() if f.is_file() and f.name != ".gitkeep"]
        if embedded_files:
            st.markdown("**Files:**")
            for file_name in embedded_files:
                st.markdown(f"- 📄 `{file_name}`")
                
    url_sources = {chunk.source for chunk in vector.chunks if chunk.source and chunk.source.startswith("http")}
    if url_sources:
        st.markdown("**Web URLs:**")
        for u in url_sources:
            st.markdown(f"- 🌐 [{u}]({u})")
            
    if not originals_dir.exists() and not url_sources:
        st.caption("No sources embedded yet.")

    st.divider()
    st.subheader("System Reset")
    if st.button("🗑️ Clear All Chunks & Embeddings", type="secondary"):
        try:
            index_path = Path(settings.index_dir)
            if index_path.exists():
                shutil.rmtree(index_path)
                index_path.mkdir(parents=True, exist_ok=True)
            if originals_dir.exists():
                shutil.rmtree(originals_dir)
                originals_dir.mkdir(parents=True, exist_ok=True)
            rag_cache.clear()
            st.session_state.chat_history = []
            st.success("Successfully wiped all embeddings, chunks, files, and cache!")
            time.sleep(1)
            st.rerun()
        except Exception as exc:
            st.error(f"Error resetting system: {exc}")

st.subheader("Ask about a government scheme")
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        content = turn["content"]
        if isinstance(content, str):
            try:
                parsed_eval = ast.literal_eval(content)
                if isinstance(parsed_eval, list) and parsed_eval and isinstance(parsed_eval[0], dict):
                    content = parsed_eval[0].get("text", content)
            except (ValueError, SyntaxError):
                pass
        st.markdown(content)
        
        if turn.get("sources"):
            with st.expander("Retrieved sources"):
                for source in turn["sources"]:
                    st.write(source)
        if turn.get("duration"):
            st.caption(f"⏱️ Response generated in {turn['duration']:.2f} seconds")

query = st.chat_input("Ask about eligibility, benefits, documents, deadlines…", disabled=vector is None)
if query and vector is not None:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        event_log = []
        sources = []
        started = time.perf_counter()
        answer = ""
        query_status = None
        
        cached_record = rag_cache.get(query)
        
        if cached_record:
            answer = cached_record["answer"]
            sources = cached_record["sources"]
            duration = time.perf_counter() - started
            
            if isinstance(answer, str):
                try:
                    answer = ast.literal_eval(answer)
                except (ValueError, SyntaxError):
                    pass

            response = answer[0].get("text", answer[0]) if isinstance(answer, list) and answer and isinstance(answer[0], dict) else answer.get("text", answer) if isinstance(answer, dict) else answer

            st.markdown(response)
            if sources:
                with st.expander("Retrieved sources (⚡ SQLite Cache Hit)"):
                    for source in sources:
                        st.write(source)
            st.caption(f"⏱️ Response served from DB cache in {duration:.4f} seconds")
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources, "duration": duration})
        else:
            try:
                with st.status("Processing your question…", expanded=True) as query_status:
                    progress = st.progress(0, text="Starting retrieval…")

                    def report(message, pct=None):
                        add_event(message, event_log, query_status)
                        if pct is not None:
                            progress.progress(max(0.0, min(1.0, pct)), text=message[:100])

                    report(f"Enhancing query for retrieval...", 0.20)
                    enhanced_query = enhance_query(query)

                    report("Building hybrid retriever", 0.10)
                    retriever = HybridRetriever(vector)

                    # print(f"DEBUG: Enhanced query variants: {enhanced_query}")
                    local_results = retriever.search(enhanced_query)
                    progress.progress(0.60, text="Local retrieval finished")

                    report("Generating response...", 0.85)
                    raw_answer = generate_answer(query, local_results)
                    
                    if isinstance(raw_answer, list) and len(raw_answer) > 0:
                        if isinstance(raw_answer[0], dict):
                            answer = raw_answer[0].get("text", str(raw_answer))
                        else:
                            answer = str(raw_answer[0])
                    elif isinstance(raw_answer, dict):
                        answer = raw_answer.get("text", str(raw_answer))
                    elif hasattr(raw_answer, "text"):
                        answer = raw_answer.text
                    else:
                        answer = str(raw_answer)
                    
                    # Formatted strictly with Filename, Page Number, Score, and Chunk Text
                    for source_number, (chunk, score) in enumerate(local_results, start=1):
                        file_name = Path(chunk.source).name if not chunk.source.startswith("http") else chunk.source
                        page_num = chunk.page if chunk.page else "N/A"
                        formatted_score = f"{score:.4f}" if isinstance(score, (float, int)) else str(score)
                        
                        sources.append(
                            f"**File:** `{file_name}` | **Page:** {page_num} | **RRF_Score:** `{formatted_score}`\n\n"
                            f"> {chunk.text}"
                        )
                        
                    progress.progress(1.0, text="Complete")
                    query_status.update(label="Question answered", state="complete", expanded=False)

                rag_cache.set(query, answer, sources)
                duration = time.perf_counter() - started

                if isinstance(answer, str):
                    try:
                        answer = ast.literal_eval(answer)
                    except (ValueError, SyntaxError):
                        pass

                response = answer[0].get("text", answer[0]) if isinstance(answer, list) and answer and isinstance(answer[0], dict) else answer.get("text", answer) if isinstance(answer, dict) else answer

                st.markdown(response)
                if sources:
                    with st.expander("Retrieved sources"):
                        for source in sources:
                            st.write(source)
                st.caption(f"⏱️ Total query duration: {duration:.2f} seconds")
                st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources, "duration": duration})
            except Exception as exc:
                if query_status is not None:
                    query_status.update(label="Question processing failed", state="error", expanded=True)
                message = f"Something went wrong: {type(exc).__name__}: {exc}"
                add_event(message, event_log)
                st.error(message)
                st.session_state.chat_history.append({"role": "assistant", "content": message, "sources": [], "duration": time.perf_counter() - started})