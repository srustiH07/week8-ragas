from pathlib import Path
import json

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# Configuration
# ============================================================

DATASET_FILE = "dataset.txt"
CHROMA_DIR = "chroma_db"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_K = 3


# ============================================================
# 10 evaluation questions and reference answers
# ============================================================

QA_PAIRS = [
    {
        "question": "What is RAG?",
        "reference": (
            "RAG is a technique that combines information retrieval "
            "with text generation by retrieving relevant external "
            "information and providing it to a language model as context."
        ),
    },
    {
        "question": "What are the major stages of a typical RAG pipeline?",
        "reference": (
            "The major stages are document loading, document splitting, "
            "embedding and indexing, and retrieval followed by generation."
        ),
    },
    {
        "question": "What is document chunking?",
        "reference": (
            "Document chunking divides large documents into smaller pieces "
            "so they can be processed and retrieved more effectively."
        ),
    },
    {
        "question": "What are embeddings?",
        "reference": (
            "Embeddings are numerical vector representations of text that "
            "allow semantic similarity between texts to be measured."
        ),
    },
    {
        "question": "What is ChromaDB used for in a RAG system?",
        "reference": (
            "ChromaDB is a vector database used to store document embeddings "
            "and retrieve documents that are semantically similar to a query."
        ),
    },
    {
        "question": "What does the retrieval parameter k control?",
        "reference": (
            "The retrieval parameter k controls how many documents or chunks "
            "are returned by the retrieval process."
        ),
    },
    {
        "question": "What is LangChain used for in RAG?",
        "reference": (
            "LangChain provides components for connecting document loaders, "
            "text splitters, embeddings, vector stores, retrievers, and "
            "language models into a RAG workflow."
        ),
    },
    {
        "question": "What are Ollama, qwen2.5:3b, and nomic-embed-text used for?",
        "reference": (
            "Ollama runs language models locally. qwen2.5:3b is used for "
            "generating answers, while nomic-embed-text is used to create "
            "embeddings for document retrieval."
        ),
    },
    {
        "question": "What is Ragas?",
        "reference": (
            "Ragas is a framework used to evaluate RAG and LLM applications "
            "using metrics that measure answer and retrieval quality."
        ),
    },
    {
        "question": "What does faithfulness measure in Ragas?",
        "reference": (
            "Faithfulness measures whether a generated answer is supported "
            "by the retrieved context and does not contain unsupported claims."
        ),
    },
]


# ============================================================
# Load dataset
# ============================================================

def load_dataset():
    path = Path(DATASET_FILE)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_FILE}"
        )

    text = path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError("The dataset is empty.")

    print(f"Loaded dataset: {DATASET_FILE}")
    print(f"Dataset characters: {len(text)}")

    return text


# ============================================================
# Create vector database
# ============================================================

def create_vector_store(text):
    print("\nCreating document chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    documents = splitter.create_documents([text])

    print(f"Created {len(documents)} chunks")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")

    print("\nLoading Ollama embedding model...")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    print("Creating ChromaDB vector store...")

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="w8d2_ragas_collection",
    )

    print("ChromaDB vector store created successfully.")

    return vector_store


# ============================================================
# Create RAG chain
# ============================================================

def create_rag_chain():
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful question-answering assistant.

Answer the user's question using ONLY the supplied context.

If the context does not contain enough information to answer,
say that the information is not available in the provided context.

Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    return llm, prompt


# ============================================================
# Generate RAG answers
# ============================================================

def run_evaluation_questions(vector_store):
    llm, prompt = create_rag_chain()

    results = []

    print("\n" + "=" * 70)
    print("GENERATING 10 RAG QUESTION-ANSWER PAIRS")
    print("=" * 70)

    for index, qa in enumerate(QA_PAIRS, start=1):
        question = qa["question"]
        reference = qa["reference"]

        print(f"\nQuestion {index}/10")
        print(f"Q: {question}")

        retrieved_docs = vector_store.similarity_search(
            question,
            k=RETRIEVAL_K,
        )

        contexts = [
            document.page_content
            for document in retrieved_docs
        ]

        context_text = "\n\n".join(contexts)

        formatted_prompt = prompt.invoke(
            {
                "context": context_text,
                "question": question,
            }
        )

        response = llm.invoke(formatted_prompt)

        answer = response.content

        print(f"A: {answer}")

        results.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "reference": reference,
            }
        )

    return results


# ============================================================
# Save results for Ragas
# ============================================================

def save_results(results):
    output_file = "ragas_dataset.json"

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print(f"Saved Ragas evaluation dataset: {output_file}")
    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("W8D2 - RAG PIPELINE")
    print("LangChain + ChromaDB + Ollama")
    print("=" * 70)

    text = load_dataset()

    vector_store = create_vector_store(text)

    results = run_evaluation_questions(vector_store)

    save_results(results)

    print("\nRAG pipeline completed successfully.")
    print(f"Questions evaluated: {len(results)}")
    print(f"Retrieval k: {RETRIEVAL_K}")
    print(f"Chunk size: {CHUNK_SIZE}")


if __name__ == "__main__":
    main()