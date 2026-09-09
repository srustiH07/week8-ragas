from src.document_review import (
    analyze_document,
    generate_review,
    review_document,
)


def test_analyze_document_counts_words():
    state = {
        "document": "Artificial intelligence improves software development.",
        "word_count": 0,
        "review": "",
    }

    result = analyze_document(state)

    assert result["word_count"] == 5


def test_empty_document_review():
    state = {
        "document": "",
        "word_count": 0,
        "review": "",
    }

    result = generate_review(state)

    assert result["review"] == (
        "The document is empty. Content should be added."
    )


def test_short_document_review():
    state = {
        "document": "AI improves software development.",
        "word_count": 4,
        "review": "",
    }

    result = generate_review(state)

    assert result["review"] == (
        "The document is very short. More details are recommended."
    )


def test_complete_review_workflow():
    document = (
        "Artificial intelligence helps organizations automate tasks "
        "and improve decision making."
    )

    result = review_document(document)

    assert result["word_count"] == 10
    assert result["review"] != ""


def test_review_contains_expected_fields():
    document = "AI is useful for modern applications."

    result = review_document(document)

    assert "document" in result
    assert "word_count" in result
    assert "review" in result