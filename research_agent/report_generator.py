"""Markdown and PDF report generation for retrieved papers."""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from fpdf import FPDF

from research_agent.models import Paper, ReportMetadata

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Build and save structured research reports."""

    @staticmethod
    def _get_syn(synthesis: dict, key: str, fallback: str) -> str:
        """Extract synthesis strings, safely joining lists if the LLM hallucinated an array."""
        val = synthesis.get(key)
        if not val:
            return fallback
        if isinstance(val, list):
            return "\n".join(f"- {str(item)}" for item in val)
        return str(val)

    def generate(
        self,
        topic: str,
        papers: list[Paper],
        metadata: ReportMetadata | None = None,
        synthesis: dict | None = None,
    ) -> str:
        """Create a readable report from retrieved papers."""
        metadata = metadata or ReportMetadata()
        synthesis = synthesis or {}
        
        overview = self._build_overview(topic, papers)
        executive_summary = self._get_syn(synthesis, "executive_summary", self._build_research_summary(papers))
        key_papers = self._build_key_papers(papers)
        key_findings = self._get_syn(synthesis, "key_findings", self._build_key_findings(papers))
        research_gaps = self._get_syn(synthesis, "research_gaps", self._build_research_gaps(papers))
        future_directions = self._get_syn(synthesis, "future_directions", self._build_future_directions(papers))
        references = self._build_references(papers)

        return "\n\n".join(
            [
                f"# Research Report: {topic}",
                f"## Topic Understanding\n{self._build_topic_understanding(metadata)}",
                f"## Executive Summary\n{executive_summary}",
                f"## Key Findings\n{key_findings}",
                f"## Research Gaps\n{research_gaps}",
                f"## Future Directions\n{future_directions}",
                f"## Key Papers\n{key_papers}",
                f"## References\n{references}",
                f"\n---\n",
                f"<details>\n<summary>Advanced Details / Developer Mode</summary>\n",
                f"### Search Settings\n{self._build_search_settings(metadata)}\n",
                f"### Expanded Keywords\n{self._build_expanded_keywords(metadata)}\n",
                f"### Generated Search Query\n`{metadata.generated_query or topic}`\n",
                f"</details>",
            ]
        )

    def save(self, report: str, output_path: str | Path) -> Path:
        """Save a Markdown report and return the final path."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        logger.info("Saved research report to %s", path)
        return path

    def generate_pdf(
        self,
        topic: str,
        papers: list[Paper],
        metadata: ReportMetadata | None = None,
        synthesis: dict | None = None,
    ) -> bytes:
        """Create a clean PDF report from the same analysis."""
        metadata = metadata or ReportMetadata()
        synthesis = synthesis or {}
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_title(f"Research Report: {topic}")
        pdf.set_author("Research Agent Prototype")

        self._pdf_heading(pdf, f"Research Report: {topic}", size=18)
        self._pdf_section(pdf, "Topic", topic)
        self._pdf_section(pdf, "Search Settings", self._plain_text(self._build_search_settings(metadata)))
        self._pdf_section(pdf, "Topic Understanding", self._plain_text(self._build_topic_understanding(metadata)))
        self._pdf_section(pdf, "Expanded Keywords", self._plain_text(self._build_expanded_keywords(metadata)))
        self._pdf_section(pdf, "Generated Search Query", metadata.generated_query or topic)
        self._pdf_section(pdf, "Overview", self._plain_text(self._build_overview(topic, papers)))
        self._pdf_section(pdf, "Executive Summary", self._plain_text(self._get_syn(synthesis, "executive_summary", self._build_research_summary(papers))))
        self._pdf_section(pdf, "Key Papers", self._plain_text(self._build_key_papers(papers)))
        self._pdf_section(pdf, "Key Findings", self._plain_text(self._get_syn(synthesis, "key_findings", self._build_key_findings(papers))))
        self._pdf_section(pdf, "Research Gaps", self._plain_text(self._get_syn(synthesis, "research_gaps", self._build_research_gaps(papers))))
        self._pdf_section(pdf, "Future Directions", self._plain_text(self._get_syn(synthesis, "future_directions", self._build_future_directions(papers))))
        self._pdf_section(pdf, "References", self._plain_text(self._build_references(papers)))

        return bytes(pdf.output(dest="S"))

    def save_pdf(self, report_pdf: bytes, output_path: str | Path) -> Path:
        """Save a PDF report and return the final path."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(report_pdf)
        logger.info("Saved PDF research report to %s", path)
        return path

    @staticmethod
    def _build_search_settings(metadata: ReportMetadata) -> str:
        return "\n".join(
            [
                f"- Requested papers: {metadata.requested_results}",
                f"- Retrieved papers: {metadata.retrieved_results}",
                f"- Sort mode: {metadata.sort_mode.label}",
                f"- Date filter: {metadata.date_filter.label}",
                f"- LLM model: {metadata.llm_model}",
            ]
        )

    @staticmethod
    def _build_topic_understanding(metadata: ReportMetadata) -> str:
        if metadata.topic_understanding:
            return metadata.topic_understanding
        return "No topic interpretation was generated."

    @staticmethod
    def _build_expanded_keywords(metadata: ReportMetadata) -> str:
        keywords = metadata.expanded_keywords
        sections = [
            ("Primary Terms", keywords.primary_terms),
            ("Related Terms", keywords.related_terms),
            ("Domain Terms", keywords.domain_terms),
            ("Synonyms", keywords.synonyms),
            ("Research Keywords", keywords.research_keywords),
        ]
        lines = []
        for label, terms in sections:
            value = ", ".join(terms) if terms else "None"
            lines.append(f"- {label}: {value}")

        if metadata.web_results:
            lines.append("- Web Context:")
            for result in metadata.web_results[:3]:
                lines.append(f"  - {result.title}: {result.snippet}")
        return "\n".join(lines)

    @staticmethod
    def _build_overview(topic: str, papers: list[Paper]) -> str:
        if not papers:
            return f"No arXiv papers were found for **{topic}**."

        combined_abstracts = " ".join(paper.abstract for paper in papers)
        preview = " ".join(combined_abstracts.split())[:900]
        return (
            f"This report summarizes {len(papers)} relevant arXiv paper(s) for "
            f"**{topic}**. The retrieved abstracts suggest the following research "
            f"context: {preview}..."
        )

    @staticmethod
    def _build_research_summary(papers: list[Paper]) -> str:
        if not papers:
            return "No research summary could be generated because no papers were retrieved."

        themes = ReportGenerator._top_terms(papers, limit=6)
        theme_text = ", ".join(themes) if themes else "the retrieved topic"
        return (
            f"Across the retrieved papers, common themes include {theme_text}. "
            "The abstracts indicate a cluster of work around methods, evaluation, "
            "system design, and practical limitations. This summary is generated "
            "from repeated terms and leading abstract sentences rather than an external LLM."
        )

    @staticmethod
    def _build_key_papers(papers: list[Paper]) -> str:
        if not papers:
            return "No key papers available."

        lines = []
        for index, paper in enumerate(papers, start=1):
            lines.append(
                f"{index}. **{paper.title}** ({paper.published_date})\n"
                f"   - Authors: {paper.authors_text}\n"
                f"   - Abstract: {paper.short_abstract(350)}\n"
                f"   - URL: {paper.url}"
            )
        return "\n".join(lines)

    @staticmethod
    def _build_key_findings(papers: list[Paper]) -> str:
        if not papers:
            return "No findings could be generated because no papers were retrieved."

        findings = []
        for paper in papers[:5]:
            first_sentence = paper.abstract.replace("\n", " ").split(". ")[0].strip()
            if first_sentence:
                findings.append(f"- {first_sentence}.")

        return "\n".join(findings) if findings else "No clear findings were extractable."

    @staticmethod
    def _build_research_gaps(papers: list[Paper]) -> str:
        if not papers:
            return "No research gaps could be identified because no papers were retrieved."

        abstracts = ReportGenerator._combined_abstracts(papers).lower()
        gaps = []
        gap_signals = ["limitation", "challenge", "open problem", "future work", "scalability", "evaluation"]
        found_signals = [signal for signal in gap_signals if signal in abstracts]

        if found_signals:
            gaps.append(
                "- Several abstracts mention signals such as "
                f"{', '.join(found_signals[:4])}, suggesting unresolved practical or evaluation issues."
            )
        gaps.append(
            "- Cross-paper comparison may be limited because abstracts often emphasize each paper's method "
            "more than standardized benchmarks or negative results."
        )
        gaps.append(
            "- The retrieved set may underrepresent adjacent work outside arXiv or papers that use different terminology."
        )
        return "\n".join(gaps)

    @staticmethod
    def _build_future_directions(papers: list[Paper]) -> str:
        if not papers:
            return "No future directions could be suggested because no papers were retrieved."

        themes = ReportGenerator._top_terms(papers, limit=4)
        directions = [
            "- Compare the retrieved approaches on shared datasets and consistent evaluation metrics.",
            "- Study robustness, reproducibility, and failure cases across realistic usage settings.",
            "- Connect the strongest ideas from multiple papers into a unified experimental baseline.",
        ]
        if themes:
            directions.insert(
                0,
                f"- Explore deeper combinations of recurring themes such as {', '.join(themes)}.",
            )
        return "\n".join(directions)

    @staticmethod
    def _build_references(papers: list[Paper]) -> str:
        if not papers:
            return "No references available."

        return "\n".join(
            f"- {paper.authors_text}. **{paper.title}**. arXiv, {paper.published_date}. {paper.url}"
            for paper in papers
        )

    @staticmethod
    def _combined_abstracts(papers: list[Paper]) -> str:
        return " ".join(paper.abstract for paper in papers)

    @staticmethod
    def _top_terms(papers: list[Paper], limit: int = 6) -> list[str]:
        stopwords = {
            "about",
            "across",
            "also",
            "analysis",
            "approach",
            "based",
            "between",
            "demonstrate",
            "different",
            "from",
            "have",
            "into",
            "method",
            "methods",
            "model",
            "models",
            "paper",
            "present",
            "propose",
            "results",
            "show",
            "study",
            "that",
            "their",
            "these",
            "this",
            "through",
            "using",
            "with",
        }
        words = re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", ReportGenerator._combined_abstracts(papers).lower())
        counts = Counter(word for word in words if word not in stopwords)
        return [word for word, _ in counts.most_common(limit)]

    @staticmethod
    def _plain_text(markdown_text: str) -> str:
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", markdown_text)
        text = text.replace("   - ", "  - ")
        return text

    @staticmethod
    def _pdf_heading(pdf: FPDF, text: str, size: int = 14) -> None:
        pdf.set_font("Helvetica", "B", size)
        pdf.multi_cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    @staticmethod
    def _pdf_section(pdf: FPDF, heading: str, body: str) -> None:
        ReportGenerator._pdf_heading(pdf, heading, size=13)
        pdf.set_font("Helvetica", "", 10)
        safe_body = body.encode("latin-1", "replace").decode("latin-1")
        for paragraph in safe_body.split("\n"):
            if paragraph.strip():
                pdf.multi_cell(0, 6, paragraph, new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.ln(3)
        pdf.ln(3)
