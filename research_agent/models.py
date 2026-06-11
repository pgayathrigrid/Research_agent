"""Shared data models for the research agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class SortMode(str, Enum):
    """Supported arXiv result ordering options."""

    RELEVANCE = "relevance"
    PUBLICATION_DATE = "publication_date"

    @property
    def label(self) -> str:
        """Return display text for reports and UI."""
        labels = {
            SortMode.RELEVANCE: "Relevance",
            SortMode.PUBLICATION_DATE: "Publication Date",
        }
        return labels[self]


class DateFilterPreset(str, Enum):
    """Date filter choices exposed by CLI and Streamlit."""

    ALL = "all"
    LAST_1_YEAR = "last_1_year"
    LAST_2_YEARS = "last_2_years"
    LAST_5_YEARS = "last_5_years"
    CUSTOM = "custom"


@dataclass(frozen=True)
class DateFilter:
    """Publication date filter for retrieved papers."""

    preset: DateFilterPreset = DateFilterPreset.ALL
    start_date: date | None = None
    end_date: date | None = None

    @classmethod
    def from_preset(
        cls,
        preset: str | DateFilterPreset = DateFilterPreset.ALL,
        start_date: date | None = None,
        end_date: date | None = None,
        today: date | None = None,
    ) -> "DateFilter":
        """Build a date filter from a named preset or custom dates."""
        selected_preset = DateFilterPreset(preset)
        today = today or date.today()

        if selected_preset == DateFilterPreset.ALL:
            return cls(preset=selected_preset)
        if selected_preset == DateFilterPreset.LAST_1_YEAR:
            return cls(preset=selected_preset, start_date=today - timedelta(days=365), end_date=today)
        if selected_preset == DateFilterPreset.LAST_2_YEARS:
            return cls(preset=selected_preset, start_date=today - timedelta(days=365 * 2), end_date=today)
        if selected_preset == DateFilterPreset.LAST_5_YEARS:
            return cls(preset=selected_preset, start_date=today - timedelta(days=365 * 5), end_date=today)

        if start_date is None or end_date is None:
            raise ValueError("Custom date range requires both start_date and end_date.")
        if start_date > end_date:
            raise ValueError("start_date cannot be after end_date.")
        return cls(preset=selected_preset, start_date=start_date, end_date=end_date)

    @property
    def label(self) -> str:
        """Return readable date filter text for reports and UI."""
        if self.preset == DateFilterPreset.ALL:
            return "All publication dates"
        if self.start_date and self.end_date:
            return f"{self.start_date.isoformat()} to {self.end_date.isoformat()}"
        return self.preset.value.replace("_", " ").title()

    def includes(self, paper_date: date) -> bool:
        """Return whether a paper date is inside the active filter."""
        if self.start_date and paper_date < self.start_date:
            return False
        if self.end_date and paper_date > self.end_date:
            return False
        return True


@dataclass(frozen=True)
class ReportMetadata:
    """Context that explains how a report was produced."""

    sort_mode: SortMode = SortMode.RELEVANCE
    date_filter: DateFilter = field(default_factory=DateFilter)
    requested_results: int = 5
    retrieved_results: int = 0
    original_topic: str = ""
    generated_query: str = ""
    topic_understanding: str = ""
    llm_model: str = "gpt-5.4-mini"
    expanded_keywords: "KeywordExpansion" = field(default_factory=lambda: KeywordExpansion())
    web_results: list["WebSearchResult"] = field(default_factory=list)


@dataclass(frozen=True)
class WebSearchResult:
    """Structured web search context used before arXiv retrieval."""

    title: str
    snippet: str
    url: str


@dataclass(frozen=True)
class KeywordExpansion:
    """Research-oriented interpretation of a user topic."""

    primary_terms: list[str] = field(default_factory=list)
    related_terms: list[str] = field(default_factory=list)
    domain_terms: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    research_keywords: list[str] = field(default_factory=list)
    topic_understanding: str = ""

    def all_terms(self) -> list[str]:
        """Return deduplicated keywords in priority order."""
        ordered_terms = (
            self.primary_terms
            + self.related_terms
            + self.domain_terms
            + self.synonyms
            + self.research_keywords
        )
        seen = set()
        deduped = []
        for term in ordered_terms:
            normalized = term.strip()
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                deduped.append(normalized)
        return deduped


@dataclass(frozen=True)
class Paper:
    """A small, UI-friendly representation of an arXiv paper."""

    title: str
    authors: list[str]
    published_date: str
    abstract: str
    url: str

    @property
    def authors_text(self) -> str:
        """Return authors as readable text for console and reports."""
        return ", ".join(self.authors) if self.authors else "Unknown authors"

    def short_abstract(self, max_characters: int = 500) -> str:
        """Return a compact abstract preview without cutting words harshly."""
        clean_abstract = " ".join(self.abstract.split())
        if len(clean_abstract) <= max_characters:
            return clean_abstract

        trimmed = clean_abstract[:max_characters].rsplit(" ", 1)[0]
        return f"{trimmed}..."

    @property
    def published(self) -> date:
        """Return the publication date as a date object."""
        return date.fromisoformat(self.published_date)
