from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain_core.documents import Document
import os


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file and return LangChain Document objects.
    
    Each page becomes one Document with metadata:
    - source
    - page number
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    return documents
