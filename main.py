if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    from ingest import load_pdf
    from chunking import split_documents
    from embeddings import get_embeddings
    from vector_store import create_vector_store, load_vector_store
    import config

    pdf_path = "sample.pdf"

    # Load PDF documents
    try:
        docs = load_pdf(pdf_path)
    except FileNotFoundError as e:
        raise SystemExit(f"PDF not found: {pdf_path}. Please provide a valid file.")

    # Split into chunks
    chunks = split_documents(docs)

    # Create embeddings (must match whatever produced the persisted store)
    embeddings = get_embeddings(config.EMBEDDINGS_PROVIDER)

    # Load the persisted store; Chroma creates an empty one if none exists
    # yet, so check the actual document count rather than relying on an
    # exception to decide whether ingestion is needed.
    vectorstore = load_vector_store(embeddings)
    if vectorstore._collection.count() == 0:
        vectorstore = create_vector_store(chunks, embeddings)

    # Run a similarity search example
    results = vectorstore.similarity_search("specials", k=3)

    for r in results:
        print(r.page_content[:200])
        print(r.metadata)
        print("-" * 50)
