"""Conditional Tavily search for current, missing, or conflicting local evidence."""
from config import settings
from observability.tracing import traceable
from generation.answer_generator import generate_answer # or a lightweight Gemini call

@traceable("web_search_routing")
def should_search_web(query: str, local_results: list) -> bool:
  """Web search provider using query classification."""
import os
import google.generativeai as genai
from config import settings
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

"""Web search provider using modern google-genai client."""
import os
from config import settings

def should_search_web(query: str, local_results: list) -> bool:
    if not local_results:
        return True

    api_key = settings.google_api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False

    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        prompt = f"""
        You are a strict context evaluator for a Tamil Nadu Government Schemes RAG system.

        Your goal is to determine if a live WEB SEARCH is required to answer the user query, or if the provided local documents are SUFFICIENT.

        User Query: "{query}"

        Retrieved Local Documents:
        {local_results}

        EVALUATION CRITERIA:
        1. Is the retrieved local document text directly relevant to the user query?
        2. Does the local text contain enough explicit facts/details to provide a complete and accurate answer?
        3. Does the query require real-time or off-topic information NOT present in static scheme documents (e.g., current Chief Minister, weather updates, general knowledge like "Eiffel tower height", or recent 2026 scheme notifications)?

        DECISION RULES:
        - Reply "NO" (meaning NO web search needed) ONLY IF the local documents are completely sufficient and directly answer the query.
        - Reply "YES" (meaning YES, web search IS required) IF the local documents are empty, off-topic, incomplete, or if the query asks for real-time external facts.

        Respond with strictly ONE word: "YES" or "NO".
        """
       
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )
        print(f"DEBUG: Evaluating query for web search necessity:\n{response.text}\n")
        return "YES" in response.text.upper()
    except Exception:
        # Fallback keyword rules
        q = query.casefold()
        return any(k in q for k in ["who is", "height", "latest", "current", "2026"])
    
@traceable("tavily_search")
def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    if not settings.tavily_api_key:
        return []
    from tavily import TavilyClient
    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(query=query, search_depth="advanced", max_results=max_results,
                             include_raw_content=True, country="india")
    results = []
    for item in response.get("results", []):
        results.append({"title": item.get("title", "Untitled"), "url": item.get("url", ""),
                        "content": item.get("raw_content") or item.get("content", ""),
                        "score": item.get("score"), "source_type": "web"})
    return results
