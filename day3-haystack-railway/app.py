# W8D3 - Haystack Dense Retrieval
# Cynaris Solutions Internship

from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder
)


print("=" * 60)
print("       W8D3 - DENSE RETRIEVAL")
print("=" * 60)


# --------------------------------------------------
# 1. Create document store using cosine similarity
# --------------------------------------------------

print("\nCreating document store...")

document_store = InMemoryDocumentStore(
    embedding_similarity_function="cosine"
)

print("Document store created successfully.")


# --------------------------------------------------
# 2. Create the same 5 documents
# --------------------------------------------------

documents = [
    Document(
        content="Artificial Intelligence is a field of computer science that enables machines to perform tasks requiring human intelligence.",
        meta={"topic": "AI"}
    ),

    Document(
        content="Machine Learning allows computers to learn patterns from data and make predictions without explicit programming.",
        meta={"topic": "ML"}
    ),

    Document(
        content="Retrieval Augmented Generation combines document retrieval with a language model to generate answers using relevant context.",
        meta={"topic": "RAG"}
    ),

    Document(
        content="ChromaDB is a vector database designed to store embeddings and perform similarity search.",
        meta={"topic": "ChromaDB"}
    ),

    Document(
        content="Haystack is an open-source framework for building search, question answering, and retrieval augmented generation pipelines.",
        meta={"topic": "Haystack"}
    )
]


# --------------------------------------------------
# 3. Create document embedder
# --------------------------------------------------

print("\nLoading document embedding model...")

document_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

document_embedder.warm_up()

print("Document embedding model loaded.")


# --------------------------------------------------
# 4. Generate embeddings
# --------------------------------------------------

print("\nGenerating document embeddings...")

embedded_documents = document_embedder.run(
    documents=documents
)["documents"]

print("Document embeddings generated successfully.")


# --------------------------------------------------
# 5. Store embedded documents
# --------------------------------------------------

document_store.write_documents(embedded_documents)

print("Embedded documents stored successfully.")


# --------------------------------------------------
# 6. Create text embedder and dense retriever
# --------------------------------------------------

text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

text_embedder.warm_up()

retriever = InMemoryEmbeddingRetriever(
    document_store=document_store
)


# --------------------------------------------------
# 7. Build dense retrieval pipeline
# --------------------------------------------------

pipeline = Pipeline()

pipeline.add_component(
    instance=text_embedder,
    name="text_embedder"
)

pipeline.add_component(
    instance=retriever,
    name="retriever"
)

pipeline.connect(
    "text_embedder.embedding",
    "retriever.query_embedding"
)

print("Dense retrieval pipeline created successfully.")


# --------------------------------------------------
# 8. Same 10 questions used for BM25 comparison
# --------------------------------------------------

questions = [
    "What is Artificial Intelligence?",
    "What is Machine Learning?",
    "What is RAG?",
    "What is ChromaDB?",
    "What is Haystack?",
    "How does Machine Learning work?",
    "What is retrieval augmented generation?",
    "What is a vector database?",
    "What can Haystack be used for?",
    "How does RAG use retrieved documents?"
]


# --------------------------------------------------
# 9. Run dense retrieval
# --------------------------------------------------

print("\n" + "=" * 60)
print("       DENSE RETRIEVAL RESULTS")
print("=" * 60)


for i, question in enumerate(questions, start=1):

    result = pipeline.run(
        {
            "text_embedder": {
                "text": question
            },
            "retriever": {
                "top_k": 1
            }
        }
    )

    retrieved_documents = result["retriever"]["documents"]

    print(f"\nQuestion {i}: {question}")

    if retrieved_documents:

        document = retrieved_documents[0]

        print("Retrieved:")
        print(document.content)

        print("Topic:", document.meta.get("topic"))

        if document.score is not None:
            print("Similarity score:", round(document.score, 4))

    else:
        print("No document retrieved.")


# --------------------------------------------------
# 10. Completion
# --------------------------------------------------

print("\n" + "=" * 60)
print("W8D3 DENSE RETRIEVAL COMPLETED SUCCESSFULLY")
print("=" * 60)