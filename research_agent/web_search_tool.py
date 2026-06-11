"""Lightweight web search context gathering."""

from __future__ import annotations

import logging
from urllib.parse import quote_plus
from urllib.parse import urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

from research_agent.models import WebSearchResult

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Search the public web and return concise contextual snippets."""

    def __init__(self, timeout_seconds: int = 10) -> None:
        self.timeout_seconds = timeout_seconds

    def search(self, topic: str, max_results: int = 5) -> list[WebSearchResult]:
        """Return web search result titles, snippets, and URLs."""
        topic = topic.strip()
        if not topic:
            raise ValueError("Research topic cannot be empty.")
        if max_results < 1:
            raise ValueError("max_results must be at least 1.")

        logger.info("Searching web context for topic=%r", topic)
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(topic)}"
        headers = {"User-Agent": "ResearchAgentPrototype/1.0"}

        try:
            response = requests.get(url, headers=headers, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException:
            logger.exception("Web search failed for topic=%r", topic)
            return []

        results = self._parse_duckduckgo_html(response.text, max_results=max_results)
        logger.info("Retrieved %s web context result(s)", len(results))
        return results

    @staticmethod
    def _parse_duckduckgo_html(html: str, max_results: int) -> list[WebSearchResult]:
        """Parse DuckDuckGo's lightweight HTML results page."""
        soup = BeautifulSoup(html, "html.parser")
        parsed_results = []

        for result in soup.select(".result"):
            title_link = result.select_one(".result__a")
            snippet_node = result.select_one(".result__snippet")
            if not title_link:
                continue

            title = title_link.get_text(" ", strip=True)
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            raw_href = title_link.get("href", "")
            
            # Clean DuckDuckGo redirect URLs if present
            if raw_href.startswith("/l/?kh="):
                parsed_query = parse_qs(urlparse(raw_href).query)
                href = parsed_query.get("uddg", [raw_href])[0]
            else:
                href = raw_href

            if title:
                parsed_results.append(WebSearchResult(title=title, snippet=snippet, url=href))

            if len(parsed_results) >= max_results:
                break

        return parsed_results
