# Government Scheme RAG (Tamil & English)

A production-grade, local-first Retrieval-Augmented Generation (RAG) system specifically designed for Tamil Nadu and Indian Government Schemes. Built with Streamlit, FAISS, BM25, and Google Gemini, this system supports multilingual document parsing (digital/scanned PDFs, TXT, Markdown, and Web URLs), hybrid retrieval with Reciprocal Rank Fusion (RRF), and SQLite caching.

## 📽️ System Demo Video

Watch the walkthrough video below to see the Streamlit user interface, source inspection, and hybrid search in action:

https://github.com/user-attachments/assets/your-video-attachment-id-here

## 🏗️ System Architecture & Visual Walkthrough

To make the architecture and processing flow easy to understand, the system is broken down into two core pipelines: **Ingestion** and **Chunking**.

### 1. Ingestion Pipeline
![Ingestion Pipeline](./assets/ingestionpipeline.png)

Documents pass through a rigorous multi-stage parser capable of handling complex government notification formats, structured tables, and scanned text via OCR fallbacks before being indexed.

### 2. Intelligent Chunking Strategy
![Chunking Architecture](./assets/chunking.png)

To preserve semantic context across long government scheme regulations, documents undergo **semantic-based splitting** first to identify logical section boundaries, followed by structured chunking over the extracted text with overlapping windows optimized for dense embedding retrieval.

## 🔍 Retrieval Logic & Hybrid Search

* **Hybrid Search:** Combines **FAISS** (dense semantic vector search with persistent memory) and **BM25** (sparse keyword search).
* **Reciprocal Rank Fusion (RRF):** Merges results using an RRF constant ($k=60$) to ensure high recall for both technical scheme names and exact keyword parameters.
* **SQLite Caching:** Uses **SQLite** for cache memory to store generated responses and retrieved chunks, providing instantaneous responses for repeat user queries.

## ✨ Key Features

* **Multilingual Support:** Seamlessly handles queries and documents in both English and Tamil (Tanglish/Tamil script).
* **Comprehensive Source Inspection:** Each generated response expands to show exact metadata including `Filename`, `Page Number`, `RRF Relevance Score`, and `Chunk Text`.
* **Flexible Ingestion:** Upload multi-page PDFs, text documents, or scrape live government portal URLs directly from the UI sidebar.
* **Persistent Memory & Cache:** Utilizes FAISS for robust vector persistence and SQLite for lightning-fast response caching.
* **System Diagnostics:** Built-in event logging container tracking backend initialization, embedding status, and retrieval steps.

## 📁 Directory Structure

```text
government-scheme-rag/
├── app.py                     # Main Streamlit web application interface
├── config.py                  # Global settings, environment paths, and model configurations
├── data/                      # Local data storage for raw originals and vector indices
├── ingestion/
│   ├── document_parser.py     # PDF, TXT, Markdown, and URL text extractors
│   └── chunker.py             # Semantic-based splitting & chunking logic
├── indexing/
│   └── vector_index.py        # FAISS vector store management and persistent memory
├── retrieval/
│   ├── hybrid_search.py       # BM25 + FAISS fusion and RRF ranking
│   └── cache.py               # SQLite-based query and response caching layer
├── generation/
│   └── answer_generator.py    # Gemini-powered response synthesizer
└── assets/
    ├── chunking.png           # Chunking architecture diagram
    └── ingestionpipeline.png  # Ingestion workflow diagram
```

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/government-scheme-rag.git
cd government-scheme-rag
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

## 🚀 Usage

Launch the Streamlit application locally:
```bash
streamlit run app.py
```

1. Open your browser to `http://localhost:8501`.
2. Use the **sidebar** to upload government scheme PDFs or ingest portal URLs.
3. Once indexed, type your questions into the chat input (e.g., *“What are the eligibility criteria for the Pudhumai Penn scheme?”*).
4. Expand the **Retrieved sources** section under any answer to audit chunk text, page numbers, and RRF relevance scores.

## 🛠️ System Reset & Maintenance

You can completely wipe local persistent indexes, cached SQLite responses, and uploaded source documents directly from the Streamlit sidebar using the **System Reset** button if you need to re-initialize your knowledge base.
