# W8D3 - Haystack API on Railway

## Objective

Build and compare retrieval pipelines using Haystack.

## Implemented

- Haystack InMemoryDocumentStore
- BM25 retrieval
- Dense retrieval
- Sentence Transformers embeddings
- Cosine similarity
- 10 test questions
- Manual retrieval evaluation

## BM25 Retrieval

BM25 retrieves documents using keyword-based matching.

During testing:
- AI queries retrieved AI documents.
- ML queries retrieved ML documents.
- Some semantic queries produced less accurate results.
- One RAG-related query returned no document.

## Dense Retrieval

Dense retrieval converts documents and queries into embeddings and compares their semantic similarity.

Model used:

`sentence-transformers/all-MiniLM-L6-v2`

## Technologies

- Python
- Haystack
- Sentence Transformers
- Scikit-learn
- In-Memory Document Store

## Result

Both BM25 and Dense Retrieval pipelines were implemented and tested using the same set of questions.