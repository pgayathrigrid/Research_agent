"""arXiv retrieval tool used by the agent workflow."""

from __future__ import annotations

import logging

import arxiv

from research_agent.models import DateFilter, Paper, SortMode

logger = logging.getLogger(__name__)


class ArxivTool:
    """Search arXiv and convert API results into local Paper objects."""

    def __init__(self, client: arxiv.Client | None = None) -> None:
        # arxiv.Client is the current API for fetching Search results. Keeping it
        # injectable makes the class easy to test later.
        self.client = client or arxiv.Client()

    def search(
        self,
        topic: str,
        max_results: int = 5,
        sort_mode: SortMode = SortMode.RELEVANCE,
        date_filter: DateFilter | None = None,
    ) -> list[Paper]:
        """Return arXiv papers for a topic after applying filters."""
        topic = topic.strip()
        date_filter = date_filter or DateFilter()
        if not topic:
            raise ValueError("Research topic cannot be empty.")
        if max_results < 1:
            raise ValueError("max_results must be at least 1.")

        logger.info(
            "Searching arXiv for topic=%r max_results=%s sort=%s date_filter=%s",
            topic,
            max_results,
            sort_mode.label,
            date_filter.label,
        )

        search = arxiv.Search(
            query=topic,
            max_results=self._candidate_limit(max_results, date_filter),
            sort_by=self._sort_criterion(sort_mode),
        )

        try:
            results = list(self.client.results(search))
        except Exception:
            logger.exception("arXiv search failed for topic=%r", topic)
            raise RuntimeError("Could not retrieve papers from arXiv. Please try again.") from None

        papers = [
            paper
            for paper in (self._to_paper(result) for result in results)
            if date_filter.includes(paper.published)
        ][:max_results]
        logger.info("Retrieved %s paper(s) from arXiv", len(papers))
        return papers

    @staticmethod
    def _candidate_limit(max_results: int, date_filter: DateFilter) -> int:
        """Fetch extra candidates when local date filtering is active."""
        if date_filter.start_date or date_filter.end_date:
            return min(max(max_results * 5, 25), 100)
        return max_results

    @staticmethod
    def _sort_criterion(sort_mode: SortMode) -> arxiv.SortCriterion:
        """Map local sort names to arXiv sort constants."""
        if sort_mode == SortMode.PUBLICATION_DATE:
            return arxiv.SortCriterion.SubmittedDate
        return arxiv.SortCriterion.Relevance

    @staticmethod
    def _to_paper(result: arxiv.Result) -> Paper:
        """Map an arxiv.Result into the project's smaller Paper model."""
        return Paper(
            title=result.title,
            authors=[author.name for author in result.authors],
            published_date=result.published.date().isoformat(),
            abstract=result.summary,
            url=result.entry_id,
        )
