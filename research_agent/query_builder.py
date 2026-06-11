"""Build optimized arXiv queries from interpreted topic context."""

from __future__ import annotations

import re

from research_agent.models import KeywordExpansion


class QueryBuilder:
    """Combine topic and expanded keywords into an arXiv search query."""

    def build(self, topic: str, keywords: KeywordExpansion) -> str:
        """Return an arXiv-compatible query string."""
        # The LLM has already identified the most important terms. Trust its output.
        primary = keywords.primary_terms
        context = keywords.related_terms + keywords.domain_terms
 
        primary_group = self._or_group(primary)
        context_group = self._or_group(context)
 
        if primary_group and context_group:
            return f"({primary_group} AND {context_group})"
 
        # Fallback to a simpler query if one of the groups is empty, or use the original topic.
        return primary_group or context_group or self._or_group([topic])

    def _or_group(self, terms: list[str]) -> str:
        clean_terms = [self._term(term) for term in terms if term.strip()]
        if not clean_terms:
            return ""
        return "(" + " OR ".join(clean_terms) + ")"

    @staticmethod
    def _term(term: str) -> str:
        sanitized = re.sub(r"\s+", " ", term.strip())
        sanitized = sanitized.replace('"', "")
        return f'all:"{sanitized}"'
