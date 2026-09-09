import os

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

DOCUMENTS_FOLDER = "pdfs"

PDF_FILES = [
    os.path.join(DOCUMENTS_FOLDER, "pdf1.pdf"),
    os.path.join(DOCUMENTS_FOLDER, "pdf2.pdf"),
    os.path.join(DOCUMENTS_FOLDER, "pdf3.pdf"),
    os.path.join(DOCUMENTS_FOLDER, "pdf4.pdf"),
    os.path.join(DOCUMENTS_FOLDER, "pdf5.pdf"),
]


# ---------------------------------------------------------
# 2. Check PDF files
# ---------------------------------------------------------

print("\nChecking PDF files...")

for pdf_file in PDF_FILES:
    if os.path.exists(pdf_file):
        print(f"[OK] {pdf_file}")
    else:
        print(f"[ERROR] Missing file: {pdf_file}")


# ---------------------------------------------------------
# 3. Create Document Store
# ---------------------------------------------------------

document_store = InMemoryDocumentStore()

print("\nDocument store created.")


# ---------------------------------------------------------
# 4. Create PDF Converter
# ---------------------------------------------------------

converter = PyPDFToDocument()


# ---------------------------------------------------------
# 5. Convert PDFs to Haystack Documents
# ---------------------------------------------------------

print("\nConverting PDFs...")

documents = []

for pdf_file in PDF_FILES:

    if not os.path.exists(pdf_file):
        continue

    result = converter.run(
        sources=[pdf_file]
    )

    converted_documents = result["documents"]

    documents.extend(converted_documents)

    print(
        f"[OK] {os.path.basename(pdf_file)} "
        f"-> {len(converted_documents)} document(s)"
    )


# ---------------------------------------------------------
# 6. Store documents
# ---------------------------------------------------------

writer = DocumentWriter(document_store=document_store)

writer.run(documents=documents)

print(f"\nTotal documents stored: {document_store.count_documents()}")


# ---------------------------------------------------------
# 7. Create BM25 Retriever
# ---------------------------------------------------------

retriever = InMemoryBM25Retriever(
    document_store=document_store
)

print("BM25 retriever created.")


# ---------------------------------------------------------
# 8. Create Haystack Pipeline
# ---------------------------------------------------------

pipeline = Pipeline()

pipeline.add_component(
    "retriever",
    retriever
)


# ---------------------------------------------------------
# 9. Test Questions
# ---------------------------------------------------------

questions = [
    "What is the main topic of the documents?",
    "What are the important concepts discussed?",
    "What are the main objectives mentioned?",
    "What methods or techniques are described?",
    "What problems are discussed?",
    "What solutions are proposed?",
    "What are the major advantages mentioned?",
    "What are the limitations mentioned?",
    "What future improvements are suggested?",
    "What are the key conclusions?"
]


# ---------------------------------------------------------
# 10. Run BM25 Retrieval
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BM25 RETRIEVAL RESULTS")
print("=" * 70)

for number, question in enumerate(questions, start=1):

    result = pipeline.run(
        {
            "retriever": {
                "query": question,
                "top_k": 3
            }
        }
    )

    retrieved_documents = result["retriever"]["documents"]

    print(f"\nQuestion {number}: {question}")
    print("-" * 70)

    if not retrieved_documents:
        print("No documents retrieved.")
        continue

    for rank, document in enumerate(retrieved_documents, start=1):

        content = document.content

        if content is None:
            content = ""

        # Display only a short preview
        preview = content.replace("\n", " ")[:500]

        print(f"\nResult {rank}:")
        print(preview)

        if len(content) > 500:
            print("...")


# ---------------------------------------------------------
# 11. Final Verification
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PIPELINE VERIFICATION")
print("=" * 70)

print(f"PDF files checked       : {len(PDF_FILES)}")
print(f"Documents stored        : {document_store.count_documents()}")
print(f"Questions tested        : {len(questions)}")
print("Retriever               : BM25")
print("Status                  : COMPLETED")
print("=" * 70)