from src.research_assistant import (
    create_research_plan,
    generate_research_answer,
    run_research_assistant,
)


def test_research_plan_for_question():
    state = {
        "question": "What is artificial intelligence?",
        "research_plan": "",
        "answer": "",
    }

    result = create_research_plan(state)

    assert "Research question:" in result["research_plan"]
    assert "What is artificial intelligence?" in result["research_plan"]


def test_empty_question_plan():
    state = {
        "question": "",
        "research_plan": "",
        "answer": "",
    }

    result = create_research_plan(state)

    assert result["research_plan"] == (
        "No research question was provided."
    )


def test_empty_question_answer():
    state = {
        "question": "",
        "research_plan": "No research question was provided.",
        "answer": "",
    }

    result = generate_research_answer(state)

    assert result["answer"] == "Please provide a research question."


def test_complete_research_workflow():
    question = "What are the benefits of artificial intelligence?"

    result = run_research_assistant(question)

    assert result["question"] == question
    assert result["research_plan"] != ""
    assert result["answer"] != ""


def test_research_result_fields():
    question = "What is machine learning?"

    result = run_research_assistant(question)

    assert "question" in result
    assert "research_plan" in result
    assert "answer" in result