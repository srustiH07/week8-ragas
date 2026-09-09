import json
import time

from datasets import Dataset

from langchain_ollama import OllamaEmbeddings, ChatOllama

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from ragas.run_config import RunConfig


# ============================================================
# Configuration
# ============================================================

DATASET_FILE = "ragas_dataset.json"

LLM_MODEL = "qwen2.5:3b"
EMBEDDING_MODEL = "nomic-embed-text"

OLLAMA_BASE_URL = "http://localhost:11434"

LLM_TIMEOUT = 300
MAX_RETRIES = 2


# ============================================================
# Load evaluation dataset
# ============================================================

def load_dataset_file():
    print("=" * 70)
    print("Loading RAG evaluation dataset")
    print("=" * 70)

    with open(DATASET_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Loaded Q&A pairs: {len(data)}")

    return data


# ============================================================
# Configure local Ollama models
# ============================================================

def configure_models():
    print("\nConfiguring local Ollama models...")

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    print(f"Evaluation LLM: {LLM_MODEL}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("Evaluation mode: sequential")
    print(f"LLM timeout: {LLM_TIMEOUT} seconds")

    return evaluator_llm, evaluator_embeddings


# ============================================================
# Convert data to Hugging Face Dataset
# ============================================================

def create_dataset(data):
    questions = []
    answers = []
    contexts = []
    references = []

    for item in data:
        questions.append(item["question"])
        answers.append(item["answer"])
        contexts.append(item["contexts"])
        references.append(item["reference"])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "reference": references,
        }
    )

    return dataset


# ============================================================
# Run Ragas evaluation
# ============================================================

def run_evaluation(dataset, evaluator_llm, evaluator_embeddings):

    print("\n" + "=" * 70)
    print("Running Ragas baseline evaluation")
    print("=" * 70)

    print("Metrics:")
    print("- Faithfulness")
    print("- Answer Relevancy")
    print("- Context Precision")
    print("- Context Recall")

    print("\nRunning with a single evaluation thread...")
    print("This may take several minutes on CPU.\n")

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    # Ragas 0.3.9 uses RunConfig for timeout/retry settings.
    # It does NOT accept max_workers in evaluate().
    run_config = RunConfig(
        timeout=LLM_TIMEOUT,
        max_retries=MAX_RETRIES,
        max_wait=30,
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )

    return result


# ============================================================
# Display results
# ============================================================

def display_results(result):

    print("\n" + "=" * 70)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 70)

    print(result)

    print("\nIndividual metric scores:")

    try:
        scores = result.scores

        for index, score in enumerate(scores, start=1):
            print(f"\nQuestion {index}:")

            for metric_name, value in score.items():
                print(f"  {metric_name}: {value}")

    except Exception as error:
        print(f"Could not display individual scores: {error}")


# ============================================================
# Save results
# ============================================================

def save_results(result):

    output_file = "ragas_results.json"

    try:
        scores = result.scores

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                scores,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print("\n" + "=" * 70)
        print(f"Saved evaluation results: {output_file}")
        print("=" * 70)

    except Exception as error:
        print(f"Could not save detailed results: {error}")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("W8D2 - RAGAS EVALUATION")
    print("RAG Evaluation with Local Ollama Models")
    print("=" * 70)

    start_time = time.time()

    # Load dataset
    data = load_dataset_file()

    # Configure models
    evaluator_llm, evaluator_embeddings = configure_models()

    # Convert dataset
    dataset = create_dataset(data)

    # Run evaluation
    result = run_evaluation(
        dataset,
        evaluator_llm,
        evaluator_embeddings,
    )

    # Display results
    display_results(result)

    # Save results
    save_results(result)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("RAGAS EVALUATION COMPLETED")
    print("=" * 70)
    print(f"Questions evaluated: {len(data)}")
    print(f"Time taken: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()