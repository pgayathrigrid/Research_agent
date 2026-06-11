"""Topic understanding and keyword expansion tool."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter

from research_agent.models import KeywordExpansion, WebSearchResult

logger = logging.getLogger(__name__)


class LLMTool:
    """Use an LLM for keyword expansion, with a deterministic local fallback."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("RESEARCH_AGENT_LLM_MODEL", "gpt-5.4-mini")

    def expand_keywords(
        self,
        topic: str,
        web_results: list[WebSearchResult],
    ) -> KeywordExpansion:
        """Interpret a topic and return research-oriented search terms."""
        if os.getenv("OPENAI_API_KEY"):
            try:
                return self._expand_with_openai(topic, web_results)
            except Exception:
                logger.exception("LLM keyword expansion failed; using deterministic fallback.")

        return self._expand_with_rules(topic, web_results)

    def _expand_with_openai(
        self,
        topic: str,
        web_results: list[WebSearchResult],
    ) -> KeywordExpansion:
        """Call OpenAI Responses API and parse a JSON keyword expansion."""
        from openai import OpenAI

        client = OpenAI()
        prompt = self._build_prompt(topic, web_results)
        response = client.responses.create(model=self.model, input=prompt)
        output_text = getattr(response, "output_text", "") or ""
        payload = self._extract_json(output_text)
        return KeywordExpansion(
            primary_terms=self._clean_terms(payload.get("primary_terms", [])),
            related_terms=self._clean_terms(payload.get("related_terms", [])),
            domain_terms=self._clean_terms(payload.get("domain_terms", [])),
            synonyms=self._clean_terms(payload.get("synonyms", [])),
            research_keywords=self._clean_terms(payload.get("research_keywords", [])),
            topic_understanding=str(payload.get("topic_understanding", "")).strip(),
        )

    @staticmethod
    def _build_prompt(topic: str, web_results: list[WebSearchResult]) -> str:
        context_lines = [
            f"- Title: {result.title}\n  Snippet: {result.snippet}"
            for result in web_results[:5]
        ]
        context = "\n".join(context_lines) if context_lines else "No web context was available."
        return f"""
You are helping a lightweight research agent improve arXiv search relevance.

Original topic:
{topic}

Web context:
{context}

Return only valid JSON with this exact schema:
{{
  "topic_understanding": "one concise sentence explaining the interpreted research topic",
  "primary_terms": ["core phrase 1", "core phrase 2"],
  "related_terms": ["related phrase"],
  "domain_terms": ["domain phrase"],
  "synonyms": ["synonym"],
  "research_keywords": ["search keyword"]
}}

Prefer terminology used in academic papers. Keep each list to 3-8 concise items.
"""

    @staticmethod
    def _extract_json(output_text: str) -> dict:
        match = re.search(r"\{.*\}", output_text, flags=re.DOTALL)
        if not match:
            raise ValueError("LLM response did not contain JSON.")
        return json.loads(match.group(0))

    @staticmethod
    def _expand_with_rules(topic: str, web_results: list[WebSearchResult]) -> KeywordExpansion:
        """Deterministic fallback for environments without an API key."""
        context = " ".join([topic] + [result.title + " " + result.snippet for result in web_results])
        glossary = {
            "nv": ["NV diamond", "nitrogen vacancy diamond", "nitrogen-vacancy center"],
            "nv-diamond": ["NV diamond", "nitrogen vacancy diamond", "nitrogen-vacancy center"],
            "nitrogen": ["nitrogen vacancy diamond", "nitrogen-vacancy center"],
            "vacancy": ["nitrogen vacancy diamond", "nitrogen-vacancy center"],
            "diamond": ["diamond quantum sensor", "solid-state quantum sensor"],
            "ecg": ["ECG", "electrocardiography", "magnetocardiography", "cardiac monitoring"],
            "ekg": ["ECG", "electrocardiography", "magnetocardiography", "cardiac monitoring"],
            "heart": ["cardiac monitoring", "magnetocardiography"],
            "cardiac": ["cardiac monitoring", "magnetocardiography"],
            "magnetic": ["magnetometry", "magnetic sensing"],
            "sensor": ["sensing", "biomedical sensor"],
        }

        tokens = re.findall(r"[A-Za-z][A-Za-z-]{1,}", context.lower())
        split_tokens = re.findall(r"[A-Za-z][A-Za-z]{1,}", context.lower().replace("-", " "))
        tokens = tokens + split_tokens
        counts = Counter(tokens)
        frequent_terms = [
            term
            for term, _ in counts.most_common(12)
            if term
            not in {
                "and",
                "for",
                "from",
                "that",
                "the",
                "this",
                "use",
                "uses",
                "with",
                "we",
                "of",
                "to",
                "in",
                "or",
            }
        ]

        expanded = []
        for token in tokens:
            expanded.extend(glossary.get(token, []))

        primary_terms = LLMTool._clean_terms(expanded[:6] + [topic])
        related_terms = LLMTool._clean_terms(expanded[4:10] + frequent_terms[:4])
        domain_terms = LLMTool._clean_terms(
            ["quantum sensing", "biomedical sensing", "magnetometry", "cardiac monitoring"]
            if any(token in tokens for token in ["nv", "diamond", "ecg", "cardiac", "heart"])
            else frequent_terms[4:10]
        )
        research_keywords = LLMTool._clean_terms(frequent_terms[:8])

        return KeywordExpansion(
            primary_terms=primary_terms[:6],
            related_terms=related_terms[:8],
            domain_terms=domain_terms[:8],
            synonyms=LLMTool._clean_terms(expanded)[:8],
            research_keywords=research_keywords,
            topic_understanding=(
                "The topic is interpreted as a research search about "
                f"{topic.strip()}, expanded with terminology from web snippets and a fixed domain glossary."
            ),
        )

    @staticmethod
    def _clean_terms(terms: list[str]) -> list[str]:
        seen = set()
        cleaned = []
        for term in terms:
            normalized = str(term).strip().strip(".,;:")
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                cleaned.append(normalized)
        return cleaned
