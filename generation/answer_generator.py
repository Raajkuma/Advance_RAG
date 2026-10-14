"""Answer generation module strictly using local document context."""
import os
import google.generativeai as genai
from config import settings

def generate_answer(query: str, local_results: list) -> str:
    """Generates an answer to the user query strictly using retrieved local document context."""
    api_key = settings.google_api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: Google API key is missing."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(settings.llm_model)

    # Format local chunks
    context_blocks = []
    for i, (chunk, score) in enumerate(local_results, start=1):
        context_blocks.append(f"[Local Chunk {i}]\nSource: {chunk.source}\nPage: {chunk.page or 'N/A'}\nContent:\n{chunk.text}")

    context_str = "\n\n".join(context_blocks) if context_blocks else "No local document context retrieved."

    system_prompt = f"""You are an AI assistant specialized in Tamil Nadu and Indian Government Schemes.
Your task is to answer user queries accurately, concisely, and strictly based on the provided context chunks.

User Question: {query}

Retrieved Context Chunks:
{context_str}

Instructions:
1. Base your response primarily on the provided context chunks.
2. If the context does not contain sufficient information to answer the question, state clearly that the provided local documents do not contain the answer.
3. Keep the answer structured, clear, and easy to read. Answer in the same language as the user query (Tamil or English).
"""

    response = model.generate_content(system_prompt)
    return response.text