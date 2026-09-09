from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class ReviewState(TypedDict):
    document: str
    word_count: int
    review: str


def analyze_document(state: ReviewState) -> ReviewState:
    """Analyze the input document and calculate its word count."""
    document = state["document"]
    word_count = len(document.split())

    return {
        **state,
        "word_count": word_count,
    }


def generate_review(state: ReviewState) -> ReviewState:
    """Generate a simple rule-based document review."""
    word_count = state["word_count"]

    if word_count == 0:
        review = "The document is empty. Content should be added."
    elif word_count < 20:
        review = "The document is very short. More details are recommended."
    elif word_count < 100:
        review = "The document has reasonable content but could be expanded."
    else:
        review = "The document contains sufficient content for review."

    return {
        **state,
        "review": review,
    }


def build_review_graph():
    """Build and compile the LangGraph document review workflow."""
    graph = StateGraph(ReviewState)

    graph.add_node("analyze", analyze_document)
    graph.add_node("review", generate_review)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "review")
    graph.add_edge("review", END)

    return graph.compile()


def review_document(document: str) -> ReviewState:
    """Run the complete document review workflow."""
    graph = build_review_graph()

    initial_state: ReviewState = {
        "document": document,
        "word_count": 0,
        "review": "",
    }

    return graph.invoke(initial_state)


if __name__ == "__main__":
    sample_document = (
        "Artificial intelligence is transforming modern software development. "
        "AI systems can automate repetitive tasks, improve decision making, "
        "and help developers build intelligent applications."
    )

    result = review_document(sample_document)

    print("=" * 60)
    print("LANGGRAPH DOCUMENT REVIEW")
    print("=" * 60)
    print(f"Word count : {result['word_count']}")
    print(f"Review     : {result['review']}")
    print("=" * 60)