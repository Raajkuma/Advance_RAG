"""Robust web page loader for RAG ingestion."""
import requests
from bs4 import BeautifulSoup
from schemas import Document

def parse_url(url: str) -> Document:
    """Fetches a URL, extracts main text content, and sets the source as the URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Remove non-content elements
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()
        
    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    cleaned_text = "\n".join(chunk for chunk in chunks if chunk)
    
    title = soup.title.string.strip() if soup.title and soup.title.string else url

    return Document(
        title=title,
        text=cleaned_text,
        source=url  # Explicitly store URL as the source identifier
    )

# Alias for compatibility with document_parser.py imports
load_url = parse_url