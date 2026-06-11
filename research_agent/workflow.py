"""High-level research workflow for CLI and Streamlit entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_agent.agent import ResearchAgent
from research_agent.models import DateFilter, KeywordExpansion, Paper, ReportMetadata, SortMode, WebSearchResult


@dataclass(frozen=True)
class ResearchResult:
    """Complete output of one research workflow run."""

    topic: str
    papers: list[Paper]
    report_markdown: str
    report_pdf: bytes
    metadata: ReportMetadata
    web_results: list[WebSearchResult]
    keyword_expansion: KeywordExpansion
    generated_query: str


class ResearchWorkflow:
    """User Query -> ResearchAgent -> ArxivTool -> ReportGenerator."""

    def __init__(self, agent: ResearchAgent | None = None) -> None:
        self.agent = agent or ResearchAgent()

    def run(
        self,
        topic: str,
        max_results: int = 5,
        sort_mode: SortMode = SortMode.RELEVANCE,
        date_filter: DateFilter | None = None,
        web_results_count: int = 5,
    ) -> ResearchResult:
        """Retrieve papers and generate the report for one topic."""
        date_filter = date_filter or DateFilter()
        web_results = self.agent.gather_web_context(topic, max_results=web_results_count)
        keyword_expansion = self.agent.expand_keywords(topic, web_results)
        generated_query = self.agent.build_query(topic, keyword_expansion)

        papers = self.agent.retrieve_papers(
            generated_query,
            max_results=max_results,
            sort_mode=sort_mode,
            date_filter=date_filter,
        )
        metadata = ReportMetadata(
            sort_mode=sort_mode,
            date_filter=date_filter,
            requested_results=max_results,
            retrieved_results=len(papers),
            original_topic=topic,
            generated_query=generated_query,
            topic_understanding=keyword_expansion.topic_understanding,
            llm_model=self.agent.llm_tool.model,
            expanded_keywords=keyword_expansion,
            web_results=web_results,
        )
        report_markdown = self.agent.create_report(topic, papers, metadata=metadata)
        report_pdf = self.agent.create_pdf_report(topic, papers, metadata=metadata)
        return ResearchResult(
            topic=topic,
            papers=papers,
            report_markdown=report_markdown,
            report_pdf=report_pdf,
            metadata=metadata,
            web_results=web_results,
            keyword_expansion=keyword_expansion,
            generated_query=generated_query,
        )

    def run_and_save(
        self,
        topic: str,
        max_results: int = 5,
        output_path: str | Path = "research_report.md",
        sort_mode: SortMode = SortMode.RELEVANCE,
        date_filter: DateFilter | None = None,
        pdf_output_path: str | Path | None = None,
        web_results_count: int = 5,
    ) -> ResearchResult:
        """Run the workflow and save requested report files."""
        result = self.run(
            topic,
            max_results=max_results,
            sort_mode=sort_mode,
            date_filter=date_filter,
            web_results_count=web_results_count,
        )
        self.agent.report_generator.save(result.report_markdown, output_path)
        if pdf_output_path:
            self.agent.report_generator.save_pdf(result.report_pdf, pdf_output_path)
        return result
