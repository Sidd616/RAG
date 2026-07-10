from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import config

def get_rag_chain(retriever):
    """
    Create a Retrieval-Augmented Generation chain
    with citation-aware responses.
    """

    llm = ChatOpenAI(
        model="openai/gpt-oss-20b:free",
        temperature=0.7,
        openai_api_key=config.get_openrouter_api_key(),
        openai_api_base="https://openrouter.ai/api/v1"
    )

    prompt = PromptTemplate(
        template="""
You are an AI assistant answering questions based on the provided context.
Be polite and energetic in your responses. Answer in a human-like manner. 

Rules:
- Use ONLY the given context to answer.
- You can use natural conversation style in the start and end of your answer.
- If the answer is not present, say: "I don't know based on the provided document. Please reach out to a human expert for more information."
- Do not use Emojis in your answers.

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain
