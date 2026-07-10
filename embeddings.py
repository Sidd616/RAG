from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os

def get_embeddings(provider: str = "openai"):
    """
    Returns an embeddings object based on provider.
    Supported providers:
    - openrouter 
    - huggingface
    """

    if provider == "openrouter":
        return OpenAIEmbeddings(
            model="openai/text-embedding-3-small",  
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),  
            openai_api_base="https://openrouter.ai/api/v1"
        )

    elif provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    else:
        raise ValueError("Unsupported embeddings provider")
