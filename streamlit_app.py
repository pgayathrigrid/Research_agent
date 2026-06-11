"""Streamlit UI for the research agent prototype."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import streamlit as st

from research_agent.models import DateFilter, DateFilterPreset, SortMode
from research_agent.workflow import ResearchWorkflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logging.getLogger("arxiv").setLevel(logging.WARNING)


DATE_FILTER_OPTIONS = {
    "All dates": DateFilterPreset.ALL,
    "Last 1 year": DateFilterPreset.LAST_1_YEAR,
    "Last 2 years": DateFilterPreset.LAST_2_YEARS,
    "Last 5 years": DateFilterPreset.LAST_5_YEARS,
    "Custom date range": DateFilterPreset.CUSTOM,
}

SORT_OPTIONS = {
    "Relevance": SortMode.RELEVANCE,
    "Publication Date": SortMode.PUBLICATION_DATE,
}


def build_date_filter(selected_label: str, start_date: date, end_date: date) -> DateFilter:
    """Build a DateFilter from UI controls."""
    preset = DATE_FILTER_OPTIONS[selected_label]
    if preset == DateFilterPreset.CUSTOM:
        return DateFilter.from_preset(
            preset=preset,
            start_date=start_date,
            end_date=end_date,
        )
    return DateFilter.from_preset(preset=preset)


def render_paper(paper, index: int) -> None:
    """Render one retrieved paper in the browser."""
    with st.expander(f"{index}. {paper.title}", expanded=index == 1):
        st.write(f"**Authors:** {paper.authors_text}")
        st.write(f"**Published:** {paper.published_date}")
        st.write(f"**Abstract:** {paper.short_abstract(500)}")
        st.link_button("Open arXiv page", paper.url)


def main() -> None:
    """Render the Streamlit research workflow."""
    st.set_page_config(page_title="Research Agent", layout="wide")
    st.title("Research Agent")

    with st.sidebar:
        topic = st.text_input("Research topic", placeholder="Example: retrieval augmented generation")
        max_results = st.slider("Number of papers", min_value=1, max_value=20, value=5)
        web_results_count = st.slider("Web context snippets", min_value=1, max_value=10, value=5)
        sort_label = st.selectbox("Sort by", options=list(SORT_OPTIONS.keys()))
        date_filter_label = st.selectbox("Publication date", options=list(DATE_FILTER_OPTIONS.keys()))
        default_end = date.today()
        default_start = default_end - timedelta(days=365)
        if DATE_FILTER_OPTIONS[date_filter_label] == DateFilterPreset.CUSTOM:
            start_date = st.date_input("Start date", value=default_start)
            end_date = st.date_input("End date", value=default_end)
        else:
            start_date = default_start
            end_date = default_end
        run_search = st.button("Generate report", type="primary")

    if not run_search:
        st.info("Enter a topic in the sidebar and generate a report.")
        return

    try:
        date_filter = build_date_filter(date_filter_label, start_date, end_date)
        with st.spinner("Searching arXiv and generating report..."):
            result = ResearchWorkflow().run(
                topic=topic,
                max_results=max_results,
                sort_mode=SORT_OPTIONS[sort_label],
                date_filter=date_filter,
                web_results_count=web_results_count,
            )
    except ValueError as error:
        st.error(str(error))
        return
    except RuntimeError as error:
        st.error(str(error))
        return

    st.success("Research report generated.")
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Papers retrieved", len(result.papers))
    metric_two.metric("Sort mode", result.metadata.sort_mode.label)
    metric_three.metric("Date filter", result.metadata.date_filter.label)

    st.subheader("Retrieval Transparency")
    st.write(f"**Original topic:** {result.topic}")
    st.write(f"**LLM model:** {result.metadata.llm_model}")
    st.write("**Topic understanding:**")
    st.write(result.keyword_expansion.topic_understanding or "No topic interpretation generated.")
    st.write("**Expanded keywords:**")
    st.write(", ".join(result.keyword_expansion.all_terms()) or "No expanded keywords generated.")
    st.write("**Generated arXiv query:**")
    st.code(result.generated_query, language="text")

    with st.expander("Web search context", expanded=False):
        if not result.web_results:
            st.info("No web context was retrieved. The agent used deterministic topic extraction only.")
        for item in result.web_results:
            st.write(f"**{item.title}**")
            st.write(item.snippet)
            if item.url:
                st.caption(item.url)

    papers_column, report_column = st.columns([1, 1])

    with papers_column:
        st.subheader("Retrieved Papers")
        if not result.papers:
            st.warning("No papers found.")
        for index, paper in enumerate(result.papers, start=1):
            render_paper(paper, index)

    with report_column:
        st.subheader("Research Report")
        st.markdown(result.report_markdown)
        st.download_button(
            label="Download Markdown report",
            data=result.report_markdown,
            file_name="research_report.md",
            mime="text/markdown",
        )
        st.download_button(
            label="Download PDF report",
            data=result.report_pdf,
            file_name="research_report.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()
