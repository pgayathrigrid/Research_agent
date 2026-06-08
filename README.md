# Research Agent Prototype

An educational research agent that searches arXiv, filters papers, displays results, and generates Markdown or PDF research reports.

## Architecture

```text
User Query
  -> ResearchWorkflow
  -> ResearchAgent
  -> ArxivTool
  -> Paper Retrieval
  -> ReportGenerator
  -> Research Report
```

## Design Decisions

- `models.py` contains shared data models such as `Paper`, `SortMode`, `DateFilter`, and `ReportMetadata`.
- `arxiv_tool.py` owns arXiv-specific retrieval, sort mapping, and publication-date filtering.
- `report_generator.py` owns all Markdown and PDF report generation.
- `agent.py` coordinates retrieval and report creation without UI or CLI concerns.
- `workflow.py` provides the end-to-end application flow used by both Streamlit and the CLI.
- `streamlit_app.py` contains only UI controls, display code, and download buttons.
- `cli.py` contains only console argument parsing, console display, and file-output choices.
- Report analysis is deterministic and rule-based. It does not call OpenAI, Claude, Gemini, or any external LLM.

## Updated File Structure

```text
research_agent/
  __init__.py
  agent.py
  arxiv_tool.py
  models.py
  report_generator.py
  workflow.py
cli.py
streamlit_app.py
research_agent_prototype.py
requirements.txt
README.md
```

## Enhancements

- Date range filtering:
  - All dates
  - Last 1 year
  - Last 2 years
  - Last 5 years
  - Custom start and end date
- Sort options:
  - Relevance
  - Publication Date
- Export options:
  - Markdown `.md`
  - PDF `.pdf`
- Report analysis sections:
  - Overview
  - Research Summary
  - Key Papers
  - Main Findings
  - Research Gaps
  - Future Directions
  - References
- Streamlit UX:
  - Active filters
  - Total papers retrieved
  - Selected sort mode
  - Markdown and PDF download buttons
  - Clear success and error messages

## Setup

```bash
cd /Users/pgayathri/Documents/Research_agent
python -m pip install -r requirements.txt
```

## Console Usage

Interactive:

```bash
python cli.py
```

Markdown export, relevance sort, last 2 years:

```bash
python cli.py "retrieval augmented generation" --max-results 5 --date-filter last_2_years --sort relevance --output reports/rag_report.md
```

PDF export:

```bash
python cli.py "retrieval augmented generation" --max-results 5 --export-format pdf --pdf-output reports/rag_report.pdf
```

Markdown and PDF export with custom dates:

```bash
python cli.py "graph neural networks" --max-results 10 --sort publication_date --date-filter custom --start-date 2024-01-01 --end-date 2026-06-08 --export-format both --output reports/gnn_report.md --pdf-output reports/gnn_report.pdf
```

The original launcher still works:

```bash
python research_agent_prototype.py
```

## Streamlit Usage

```bash
python -m streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

## Example Output Descriptions

CLI output shows:

```text
Search Settings:
Sort mode: Publication Date
Date filter: 2024-01-01 to 2026-06-08
Total papers retrieved: 5

Top Papers:
1. Paper title
Authors: ...
Published: ...
Abstract: ...
URL: ...
```

Streamlit shows:

```text
Sidebar:
- Research topic
- Number of papers
- Sort by
- Publication date filter
- Custom start/end dates when selected

Main page:
- Success message
- Papers retrieved metric
- Sort mode metric
- Date filter metric
- Retrieved paper expanders
- Markdown report preview
- Download Markdown report button
- Download PDF report button
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

Live arXiv retrieval:

```bash
python cli.py "retrieval augmented generation" --max-results 1 --export-format both --output /private/tmp/research_agent_test.md --pdf-output /private/tmp/research_agent_test.pdf
```
# Prototype_for_ResearchAgent
