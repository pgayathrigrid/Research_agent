"""Console entry point for the research agent prototype."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from research_agent.models import DateFilter, DateFilterPreset, SortMode
from research_agent.workflow import ResearchResult, ResearchWorkflow


def configure_logging(debug: bool = False) -> None:
    """Configure human-readable application logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if not debug:
        logging.getLogger("arxiv").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse optional console arguments while still supporting interactive input."""
    parser = argparse.ArgumentParser(description="Search arXiv and generate a research report.")
    parser.add_argument("topic", nargs="?", help="Research topic to search for.")
    parser.add_argument(
        "-n",
        "--max-results",
        type=int,
        default=5,
        help="Number of papers to retrieve. Default: 5.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="research_report.md",
        help="Markdown report output path. Default: research_report.md.",
    )
    parser.add_argument(
        "--pdf-output",
        default=None,
        help="PDF report output path. Default: same name as --output with .pdf extension.",
    )
    parser.add_argument(
        "--export-format",
        choices=["md", "pdf", "both"],
        default="md",
        help="Report file format to save. Default: md.",
    )
    parser.add_argument(
        "--sort",
        choices=[mode.value for mode in SortMode],
        default=SortMode.RELEVANCE.value,
        help="Sort mode: relevance or publication_date. Default: relevance.",
    )
    parser.add_argument(
        "--date-filter",
        choices=[preset.value for preset in DateFilterPreset],
        default=DateFilterPreset.ALL.value,
        help="Publication date filter. Use custom with --start-date and --end-date.",
    )
    parser.add_argument(
        "--web-results",
        type=int,
        default=5,
        help="Number of web search snippets to use for query expansion. Default: 5.",
    )
    parser.add_argument("--start-date", help="Custom start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="Custom end date in YYYY-MM-DD format.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def parse_iso_date(value: str | None, argument_name: str) -> date | None:
    """Parse a YYYY-MM-DD date argument."""
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{argument_name} must use YYYY-MM-DD format.") from error


def build_date_filter(args: argparse.Namespace) -> DateFilter:
    """Build the workflow date filter from CLI arguments."""
    return DateFilter.from_preset(
        preset=args.date_filter,
        start_date=parse_iso_date(args.start_date, "--start-date"),
        end_date=parse_iso_date(args.end_date, "--end-date"),
    )


def default_pdf_path(markdown_path: str) -> str:
    """Return a PDF path beside the Markdown output path."""
    return str(Path(markdown_path).with_suffix(".pdf"))


def display_papers(result: ResearchResult) -> None:
    """Print retrieved papers in a readable console format."""
    print("\nSearch Settings:")
    print(f"Original topic: {result.topic}")
    print(f"LLM model: {result.metadata.llm_model}")
    print(f"Sort mode: {result.metadata.sort_mode.label}")
    print(f"Date filter: {result.metadata.date_filter.label}")
    print(f"Total papers retrieved: {len(result.papers)}")
    print(f"Generated arXiv query: {result.generated_query}")
    print("\nTopic Understanding:")
    print(result.keyword_expansion.topic_understanding or "No topic interpretation generated.")
    print("\nExpanded Keywords:")
    print(", ".join(result.keyword_expansion.all_terms()) or "No expanded keywords generated.")
    print("\nTop Papers:\n")
    if not result.papers:
        print("No papers found.")
        return

    for index, paper in enumerate(result.papers, start=1):
        print(f"{index}. {paper.title}")
        print(f"Authors: {paper.authors_text}")
        print(f"Published: {paper.published_date}")
        print(f"Abstract: {paper.short_abstract(500)}")
        print(f"URL: {paper.url}")
        print("-" * 80)


def main() -> None:
    """Run the console workflow from topic input to saved report."""
    args = parse_args()
    configure_logging(debug=args.debug)

    topic = args.topic or input("Enter research topic: ").strip()

    try:
        date_filter = build_date_filter(args)
        sort_mode = SortMode(args.sort)
        workflow = ResearchWorkflow()
        if args.export_format == "pdf":
            result = workflow.run(
                topic=topic,
                max_results=args.max_results,
                sort_mode=sort_mode,
                date_filter=date_filter,
                web_results_count=args.web_results,
            )
            pdf_output = args.pdf_output or default_pdf_path(args.output)
            workflow.agent.report_generator.save_pdf(result.report_pdf, pdf_output)
        else:
            pdf_output = args.pdf_output or default_pdf_path(args.output)
            result = workflow.run_and_save(
                topic=topic,
                max_results=args.max_results,
                output_path=args.output,
                sort_mode=sort_mode,
                date_filter=date_filter,
                pdf_output_path=pdf_output if args.export_format == "both" else None,
                web_results_count=args.web_results,
            )
    except ValueError as error:
        logging.error("Invalid input: %s", error)
        raise SystemExit(1) from error
    except RuntimeError as error:
        logging.error("%s", error)
        raise SystemExit(1) from error

    display_papers(result)
    if args.export_format in {"md", "both"}:
        print(f"\nMarkdown report saved to: {args.output}")
    if args.export_format in {"pdf", "both"}:
        print(f"PDF report saved to: {args.pdf_output or default_pdf_path(args.output)}")


if __name__ == "__main__":
    main()
