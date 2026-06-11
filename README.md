# Research Agent Prototype

An educational research agent that expands a user topic before searching arXiv, then generates Markdown or PDF research reports.

## LLM Choice

The default LLM is `gemini-2.5-flash`.

Why this model:

- The LLM step is keyword expansion and terminology extraction, not full deep research.
- It is a better fit for low-latency, lower-cost structured text work than a frontier or deep-research model.
- The tool is isolated behind `LLMTool`, so the model can be changed with `RESEARCH_AGENT_LLM_MODEL`.
- If `GEMINI_API_KEY` is not set, `LLMTool` falls back to deterministic rule-based extraction so the prototype still runs locally.

## Updated Architecture

```text
User Topic
  -> ResearchWorkflow
  -> ResearchAgent
  -> WebSearchTool
  -> LLMTool
  -> Keyword Expansion
  -> QueryBuilder
  -> ArxivTool
  -> ReportGenerator
  -> Research Report
```

## Updated File Structure

```text
research_agent/
  __init__.py
  agent.py
  arxiv_tool.py
  llm_tool.py
  models.py
  query_builder.py
  report_generator.py
  web_search_tool.py
  workflow.py
cli.py
streamlit_app.py
research_agent_prototype.py
requirements.txt
README.md
```

## What The Upgrade Adds

- `WebSearchTool` gathers concise web titles and snippets for the original topic.
- `LLMTool` interprets the topic and extracts:
  - primary terms
  - related terms
  - domain terms
  - synonyms
  - research keywords
- `QueryBuilder` converts expanded terminology into an arXiv query.
- The generated arXiv query is visible in both CLI and Streamlit.
- Reports now include:
  - Topic Understanding
  - Expanded Keywords
  - Generated Search Query
  - Research Summary
  - Research Gaps
  - Future Directions
- Existing date filtering, sorting, Markdown export, and PDF export are preserved.

## Setup

```bash
cd /Users/pgayathri/Documents/Research_agent
python -m pip install -r requirements.txt
```

Optional LLM setup:

```bash
export GEMINI_API_KEY="your_gemini_api_key"
export RESEARCH_AGENT_LLM_MODEL="gemini-2.5-flash"
```

Without `OPENAI_API_KEY`, the app uses deterministic keyword expansion.

## CLI Usage

Basic:

```bash
python cli.py "NV-diamond use-cases for ECG detection"
```

With publication date sorting, date filter, web context, and both exports:

```bash
python cli.py "NV-diamond use-cases for ECG detection" --max-results 5 --web-results 5 --sort publication_date --date-filter last_5_years --export-format both --output reports/nv_ecg.md --pdf-output reports/nv_ecg.pdf
```

Custom date range:

```bash
python cli.py "quantum sensing cardiac monitoring" --date-filter custom --start-date 2022-01-01 --end-date 2026-06-08 --export-format both
```

## Streamlit Usage

```bash
python -m streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## Example Output

For:

```text
NV-diamond use-cases for ECG detection
```

The workflow may produce:

```text
Original topic: NV-diamond use-cases for ECG detection
LLM model: gpt-5.4-mini
Topic understanding: The topic is interpreted as biomedical quantum sensing using nitrogen-vacancy diamond sensors for cardiac signal detection.
Expanded keywords: NV diamond, nitrogen vacancy diamond, nitrogen-vacancy center, ECG, electrocardiography, magnetocardiography, quantum sensing, biomedical sensing, cardiac monitoring
Generated arXiv query:
(all:"NV diamond" OR all:"nitrogen vacancy diamond")
AND (all:"ECG" OR all:"electrocardiography" OR all:"magnetocardiography")
AND (all:"quantum sensing" OR all:"biomedical sensing" OR all:"cardiac monitoring")
```

Streamlit displays:

```text
- Original topic
- LLM model
- Topic understanding
- Expanded keywords
- Generated arXiv query
- Web search context
- Retrieved papers
- Markdown report download
- PDF report download
```

## Testing

Syntax check:

```bash
python -X pycache_prefix=/private/tmp/research_agent_pycache -m compileall .
```

CLI help:

```bash
python cli.py --help
```

Live retrieval:

```bash
python cli.py "NV-diamond use-cases for ECG detection" --max-results 3 --export-format both --output /private/tmp/nv_ecg.md --pdf-output /private/tmp/nv_ecg.pdf
```
# Research_agent
