import streamlit as st
import os
from dotenv import load_dotenv
from ingest import load_pdf
from chunking import split_documents
from embeddings import get_embeddings
from vector_store import create_vector_store, load_vector_store
from retriever import get_multi_query_retriever
from rag_chain import get_rag_chain
from langchain_openai import ChatOpenAI

# Load env variables
load_dotenv()

st.set_page_config(page_title="RAG Q&A System", layout="wide")

st.title("📄 Ask Your Documents (RAG System)")
st.write("Upload a PDF and ask questions grounded in its content.")

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Session State
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# Sidebar 
st.sidebar.header("📂 Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

# Document Processing 
if uploaded_file and not st.session_state.vectorstore_ready:
    with st.spinner("Processing document..."):

        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Step 1: Load
        docs = load_pdf(file_path)

        # Step 2: Chunk
        chunks = split_documents(docs)

        # Step 3: Embeddings
        embeddings = get_embeddings("openrouter")

        # Step 4: Vector Store
        vectorstore = create_vector_store(chunks, embeddings)

        st.session_state.vectorstore_ready = True
        st.success("Document processed successfully!")

# Q&A Section 
if st.session_state.vectorstore_ready:
    st.subheader("💬 Ask a Question")

    question = st.text_input("Enter your question")

    if question:
        embeddings = get_embeddings("openrouter")
        vectorstore = load_vector_store(embeddings)

        retrieval_llm = ChatOpenAI(
            model="openai/gpt-oss-20b:free",  
            temperature=0.7,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"
        )

        retriever = get_multi_query_retriever(
            vectorstore,
            llm=retrieval_llm
        )

        rag_chain = get_rag_chain(retriever)

        with st.spinner("Generating answer..."):
            response = rag_chain.invoke({"query": question})

        st.markdown("### Answer")
        st.write(response["result"])

        st.markdown("### Sources")
        sources = set()
        for doc in response["source_documents"]:
            src = f"{doc.metadata.get('source')} — Page {doc.metadata.get('page')}"
            sources.add(src)

        for s in sources:
            st.write(f"- {s}")

else:
    st.info("Upload a PDF to get started.")


