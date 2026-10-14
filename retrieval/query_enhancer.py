"""Conservative multilingual query expansion; original query is always retained."""
import re
from observability.tracing import traceable

@traceable("query_enhancement")
def enhance_query(query: str) -> list[str]:
    q = query.strip()
    if not q:
        return []
    variants = [q]
    # Preserve original Tamil/Tanglish and add a normalized punctuation/spacing variant.
    normalized = re.sub(r"[\t ]+", " ", q).strip()
    if normalized.casefold() != q.casefold():
        variants.append(normalized)
    # Keep exact amount/date/ID terms; avoid automatic translation that may alter scheme names.
    return ",".join(list(dict.fromkeys(variants)))
