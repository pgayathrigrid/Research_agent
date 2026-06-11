"""Topic understanding and keyword expansion tool."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import replace
import hashlib
from pathlib import Path

from research_agent.models import KeywordExpansion, Paper, WebSearchResult

logger = logging.getLogger(__name__)


class LLMTool:
    """Use an LLM for keyword expansion, with a deterministic local fallback."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("RESEARCH_AGENT_LLM_MODEL", "gemini-2.5-flash")
        
        # Setup persistent cache to save API calls
        self.cache_dir = Path(".agent_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "llm_cache.json"
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save LLM cache: %s", e)

    def _get_cache_key(self, prompt: str) -> str:
        return hashlib.md5((self.model + prompt).encode("utf-8")).hexdigest()

    def expand_keywords(
        self,
        topic: str,
        web_results: list[WebSearchResult],
    ) -> KeywordExpansion:
        """Interpret a topic and return research-oriented search terms."""
        if os.getenv("GEMINI_API_KEY"):
            try:
                return self._expand_with_gemini(topic, web_results)
            except Exception:
                logger.exception("LLM keyword expansion failed; using deterministic fallback.")

        return self._expand_with_rules(topic, web_results)

    def _expand_with_gemini(
        self,
        topic: str,
        web_results: list[WebSearchResult],
    ) -> KeywordExpansion:
        """Call Gemini API and parse a JSON keyword expansion."""
        prompt = self._build_prompt(topic, web_results)
        payload = self._generate_json_with_retry(prompt)
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

    def evaluate_papers(
        self,
        topic: str,
        papers: list[Paper],
    ) -> list[Paper]:
        """Evaluate and score papers based on relevance to the topic."""
        if not papers:
            return []
        
        if not os.getenv("GEMINI_API_KEY"):
            # Fallback: assign arbitrary scores cleanly via dataclass replacement
            return [
                replace(p, relevance_score=5.0, relevance_reasoning="Fallback: No API key available for scoring.") 
                for p in papers
            ]

        try:
            prompt = self._build_evaluation_prompt(topic, papers)
            payload = self._generate_json_with_retry(prompt)
            
            scores = payload.get("scores", [])
            evaluated_papers = []
            for idx, p in enumerate(papers):
                if idx < len(scores):
                    score_info = scores[idx]
                    evaluated_papers.append(
                        replace(
                            p, 
                            relevance_score=float(score_info.get("score", 5.0)), 
                            relevance_reasoning=str(score_info.get("reasoning", ""))
                        )
                    )
                else:
                    evaluated_papers.append(
                        replace(p, relevance_score=5.0, relevance_reasoning="Not scored by LLM.")
                    )
            return evaluated_papers
        except Exception as e:
            logger.exception("LLM paper evaluation failed.")
            error_reason = f"Fallback: LLM scoring failed. Reason: {e}"
            return [
                replace(p, relevance_score=5.0, relevance_reasoning=error_reason) 
                for p in papers
            ]

    @staticmethod
    def _build_evaluation_prompt(topic: str, papers: list[Paper]) -> str:
        papers_context = ""
        for idx, p in enumerate(papers):
            papers_context += f"Paper {idx}:\nTitle: {p.title}\nAbstract: {p.abstract}\n\n"
        
        return f"""
You are an expert research assistant.
Evaluate the relevance of the following papers to the user's specific research topic.

User Topic:
{topic}

Papers:
{papers_context}

For each paper, provide a relevance score from 1.0 to 10.0, and a concise 1-sentence reasoning for why this paper matters (or doesn't) for the topic.

STRICT GROUNDING RULES FOR REASONING:
- Use ONLY the provided Title and Abstract.
- Do NOT claim a paper "demonstrates" something unless the abstract explicitly states it. Prefer safer wording: "explores", "investigates", "proposes", "discusses".
- No unsupported claims. No assumptions. No invented clinical results.

Scoring Rubric:
- 10.0: Directly answers the user's topic.
- 8.0 to 9.9: Strongly related and highly useful.
- 5.0 to 7.9: Generally related but not focused on the user's topic.
- 0.0 to 4.9: Weakly related or irrelevant.

Return ONLY valid JSON with this exact schema:
{{
  "scores": [
    {{
      "score": 8.5,
      "reasoning": "This paper is relevant because..."
    }}
  ]
}}
Ensure the length of the "scores" array exactly matches the number of papers provided ({len(papers)}), in the same order.
"""

    def synthesize_report(self, topic: str, papers: list[Paper]) -> dict:
        """Synthesize a comprehensive report from the retrieved papers."""
        if not papers:
            return {}
        
        if not os.getenv("GEMINI_API_KEY"):
            return {}
            
        try:
            prompt = self._build_synthesis_prompt(topic, papers)
            return self._generate_json_with_retry(prompt)
        except Exception as e:
            logger.exception("LLM report synthesis failed.")
            return {"error": f"LLM report synthesis failed. Reason: {e}"}

    @staticmethod
    def _build_synthesis_prompt(topic: str, papers: list[Paper]) -> str:
        papers_context = ""
        for idx, p in enumerate(papers, start=1):
            papers_context += f"[{idx}] {p.title}\nAbstract: {p.abstract}\n\n"
            
        return f"""
You are an expert research assistant. Write a high-quality research report answering the user's topic based ONLY on the following papers.

User Topic:
{topic}

Papers:
{papers_context}

STRICT GROUNDING RULES:
1. Accuracy > Creativity. Evidence > Assumptions. Grounded Conclusions > Speculation.
2. NEVER claim a paper "demonstrates" something unless the abstract explicitly states it. Use safer wording: "explores", "investigates", "proposes", "discusses".
3. Research Gaps must ONLY be generated from missing evidence in the provided papers (e.g., if no human trials are mentioned, cite "limited human validation").
4. Future Directions must ONLY be derived from existing findings, limitations, or gaps in these specific papers.

Return ONLY valid JSON with this exact schema:
{{
  "executive_summary": "A concise executive summary (150-250 words) answering the user's topic. Use abstracts as citations (e.g. [1]). Escape newlines as \\n.",
  "key_findings": ["Finding 1 grounded STRICTLY in abstracts", "Finding 2..."],
  "research_gaps": ["Gap 1 based on missing evidence or limitations", "Gap 2..."],
  "future_directions": ["Direction 1 derived from existing findings", "Direction 2..."]
}}
"""

    def _generate_json_with_retry(self, prompt: str) -> dict:
        """Call Gemini API with retries, safety settings disabled, and JSON parsing."""
        cache_key = self._get_cache_key(prompt)
        if cache_key in self._cache:
            logger.info("LLM cache hit. Skipping API call.")
            return self._cache[cache_key]

        from google import genai
        from google.genai import types

        client = genai.Client()
        
        last_error = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        safety_settings=[
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                        ]
                    ),
                )
                
                try:
                    output_text = response.text or "{}"
                except ValueError as ve:
                    logger.warning("Gemini response.text raised ValueError (possibly safety blocked): %s", ve)
                    output_text = "{}"

                result = self._extract_json(output_text)
                self._cache[cache_key] = result
                self._save_cache()
                return result
                
            except Exception as e:
                last_error = e
                logger.warning("LLM generation attempt %d failed: %s", attempt + 1, e)
                if attempt < 2:
                    backoff = 10 * (attempt + 1)  # Wait 10s, then 20s
                    logger.info("Waiting %d seconds to clear potential rate limits...", backoff)
                    time.sleep(backoff)
                    continue

        logger.error("LLM generation completely failed after 3 attempts. Last error: %s", last_error)
        raise Exception(f"LLM generation failed: {last_error}")

    @staticmethod
    def _extract_json(output_text: str) -> dict:
        try:
            # Attempt direct parsing first
            return json.loads(output_text, strict=False)
        except json.JSONDecodeError:
            pass
            
        # Fallback to regex extraction if there are markdown backticks
        match = re.search(r"\{.*\}", output_text, flags=re.DOTALL)
        if not match:
            raise ValueError("LLM response did not contain JSON.")
        return json.loads(match.group(0), strict=False)

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
