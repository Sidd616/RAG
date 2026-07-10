from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
import os


VECTOR_STORE_DIR = "data/vector_store"


def create_vector_store(
    documents: List[Document],
    embeddings,
    persist_directory: str = VECTOR_STORE_DIR
):
    """
    Create and persist a Chroma vector store from documents.
    """
    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    return vectorstore


def load_vector_store(
    embeddings,
    persist_directory: str = VECTOR_STORE_DIR
):
    """
    Load an existing Chroma vector store.
    """
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
