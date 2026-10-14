"""LangSmith is activated by environment variables; this helper adds optional custom traces."""
from config import settings

def traceable(name: str):
    def decorate(fn):
        if not settings.langsmith_api_key:
            return fn
        try:
            from langsmith import traceable as ls_traceable
            return ls_traceable(name=name)(fn)
        except Exception:
            return fn
    return decorate
