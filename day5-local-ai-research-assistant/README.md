# W8D5 - 2M Capstone: Local AI Research Assistant

## Objective

Build a local AI research assistant prototype using the approved AI/ML 3M stack.

The application accepts a research question, creates a research plan using LangGraph, and generates a research summary using a locally running Ollama model.

## Technology Stack

- Python 3.12.5
- LangGraph
- Ollama 0.32.9
- qwen2.5:3b
- pytest 9.1.1

## Architecture

```text
Research Question
       |
       v
+----------------------+
| Research Planning    |
| LangGraph Node       |
+----------------------+
       |
       v
+----------------------+
| Local AI Research    |
| Ollama qwen2.5:3b    |
+----------------------+
       |
       v
Research Answer