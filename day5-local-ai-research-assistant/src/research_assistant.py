from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import ollama


class ResearchState(TypedDict):
    question: str
    research_plan: str
    answer: str


def create_research_plan(state: ResearchState) -> ResearchState:
    """Create a simple research plan from the user's question."""
    question = state["question"]

    if not question.strip():
        return {
            **state,
            "research_plan": "No research question was provided.",
        }

    plan = (
        f"Research question: {question}\n"
        "Plan:\n"
        "1. Identify the main concepts.\n"
        "2. Explain the important facts.\n"
        "3. Summarize the findings clearly."
    )

    return {
        **state,
        "research_plan": plan,
    }


def generate_research_answer(state: ResearchState) -> ResearchState:
    """Use the local Ollama model to generate a research response."""
    question = state["question"]
    research_plan = state["research_plan"]

    if not question.strip():
        return {
            **state,
            "answer": "Please provide a research question.",
        }

    prompt = f"""
You are a concise AI research assistant.

Research question:
{question}

Research plan:
{research_plan}

Provide a clear, factual research summary.
Use short paragraphs and bullet points where useful.
Do not invent sources or citations.
"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return {
        **state,
        "answer": response["message"]["content"].strip(),
    }


def build_research_graph():
    """Build and compile the LangGraph research workflow."""
    graph = StateGraph(ResearchState)

    graph.add_node("plan", create_research_plan)
    graph.add_node("research", generate_research_answer)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", END)

    return graph.compile()


def run_research_assistant(question: str) -> ResearchState:
    """Run the complete local AI research assistant workflow."""
    graph = build_research_graph()

    initial_state: ResearchState = {
        "question": question,
        "research_plan": "",
        "answer": "",
    }

    return graph.invoke(initial_state)


if __name__ == "__main__":
    question = "What are the main applications of artificial intelligence in healthcare?"

    print("=" * 70)
    print("LOCAL AI RESEARCH ASSISTANT")
    print("=" * 70)
    print(f"Question: {question}")
    print("\nGenerating research response using local Ollama model...\n")

    result = run_research_assistant(question)

    print("RESEARCH PLAN")
    print("-" * 70)
    print(result["research_plan"])

    print("\nRESEARCH ANSWER")
    print("-" * 70)
    print(result["answer"])

    print("=" * 70)