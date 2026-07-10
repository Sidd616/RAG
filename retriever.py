from langchain_classic.retrievers.multi_query import MultiQueryRetriever
#from langchain_openai import ChatOpenAI


def get_base_retriever(vectorstore, k: int = 5):
    """
    Basic similarity-based retriever
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def get_mmr_retriever(vectorstore, k: int = 6, lambda_mult: float = 0.5):
    """
    Maximal Marginal Relevance retriever
    Improves diversity of retrieved chunks
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "lambda_mult": lambda_mult
        }
    )


def get_multi_query_retriever(vectorstore, llm, k: int = 5):
    """
    Multi-query retriever:
    LLM rewrites the question into multiple queries
    """
    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
