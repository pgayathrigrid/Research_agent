"""Build optimized arXiv queries from interpreted topic context."""

from __future__ import annotations

import re

from research_agent.models import KeywordExpansion


class QueryBuilder:
    """Combine topic and expanded keywords into an arXiv search query."""

    def build(self, topic: str, keywords: KeywordExpansion) -> str:
        """Return an arXiv-compatible query string."""
        primary = self._select_core_terms(topic, keywords)
        context_terms = self._select_context_terms(keywords)

        groups = [
            self._or_group(primary),
            self._or_group(context_terms),
        ]
        return " AND ".join(group for group in groups if group)

    @staticmethod
    def _select_core_terms(topic: str, keywords: KeywordExpansion) -> list[str]:
        """Prefer concrete scientific object terms over the raw user phrase."""
        all_terms = keywords.all_terms()
        core_markers = ["nv", "nitrogen", "vacancy", "diamond", "graphene", "transformer", "neural"]
        selected = [
            term
            for term in all_terms
            if any(marker in term.lower() for marker in core_markers)
        ]
        return selected[:5] or keywords.primary_terms[:4] or [topic]

    @staticmethod
    def _select_context_terms(keywords: KeywordExpansion) -> list[str]:
        """Select research context terms that broaden the core concept usefully."""
        all_terms = keywords.all_terms()
        context_markers = [
            "ecg",
            "electrocardiography",
            "magnetocardiography",
            "cardiac",
            "biomedical",
            "sensing",
            "sensor",
            "magnetometry",
        ]
        selected = [
            term
            for term in all_terms
            if any(marker in term.lower() for marker in context_markers)
        ]
        return selected[:8] or (keywords.related_terms + keywords.domain_terms)[:8]

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
