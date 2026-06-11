"""Agent-facing orchestration layer."""

from __future__ import annotations

import logging

from research_agent.arxiv_tool import ArxivTool
from research_agent.llm_tool import LLMTool
from research_agent.models import DateFilter, KeywordExpansion, Paper, SortMode, WebSearchResult
from research_agent.query_builder import QueryBuilder
from research_agent.report_generator import ReportGenerator
from research_agent.web_search_tool import WebSearchTool

logger = logging.getLogger(__name__)


class ResearchAgent:
    """Coordinate paper retrieval and report generation."""

    def __init__(
        self,
        arxiv_tool: ArxivTool | None = None,
        report_generator: ReportGenerator | None = None,
        web_search_tool: WebSearchTool | None = None,
        llm_tool: LLMTool | None = None,
        query_builder: QueryBuilder | None = None,
    ) -> None:
        self.arxiv_tool = arxiv_tool or ArxivTool()
        self.report_generator = report_generator or ReportGenerator()
        self.web_search_tool = web_search_tool or WebSearchTool()
        self.llm_tool = llm_tool or LLMTool()
        self.query_builder = query_builder or QueryBuilder()

    def gather_web_context(self, topic: str, max_results: int = 5) -> list[WebSearchResult]:
        """Retrieve web context before arXiv search."""
        return self.web_search_tool.search(topic, max_results=max_results)

    def expand_keywords(
        self,
        topic: str,
        web_results: list[WebSearchResult],
    ) -> KeywordExpansion:
        """Interpret the topic and generate research keywords."""
        return self.llm_tool.expand_keywords(topic, web_results)

    def build_query(self, topic: str, keywords: KeywordExpansion) -> str:
        """Build the optimized arXiv query."""
        return self.query_builder.build(topic, keywords)

    def retrieve_papers(
        self,
        topic: str,
        max_results: int = 5,
        sort_mode: SortMode = SortMode.RELEVANCE,
        date_filter: DateFilter | None = None,
    ) -> list[Paper]:
        """Retrieve relevant papers through the arXiv tool."""
        logger.debug("ResearchAgent received retrieval request for %r", topic)
        return self.arxiv_tool.search(
            topic,
            max_results=max_results,
            sort_mode=sort_mode,
            date_filter=date_filter,
        )

    def create_report(self, topic: str, papers: list[Paper], metadata=None) -> str:
        """Generate a Markdown report from retrieved papers."""
        logger.debug("ResearchAgent generating report for %r", topic)
        return self.report_generator.generate(topic, papers, metadata=metadata)

    def create_pdf_report(self, topic: str, papers: list[Paper], metadata=None) -> bytes:
        """Generate a PDF report from retrieved papers."""
        logger.debug("ResearchAgent generating PDF report for %r", topic)
        return self.report_generator.generate_pdf(topic, papers, metadata=metadata)
