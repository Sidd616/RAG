import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openrouter")


def get_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )
    return api_key
