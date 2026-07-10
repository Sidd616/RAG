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
import config

# Load env variables
load_dotenv()

st.set_page_config(page_title="RAG Q&A System", layout="wide")

st.title("📄 Ask Your Documents (RAG System)")
st.write("Upload a PDF and ask questions grounded in its content.")

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Session State
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# Sidebar
st.sidebar.header("📂 Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

# Document Processing
# Keyed on filename (not a plain flag) so uploading a different PDF
# triggers reprocessing instead of silently reusing the old store.
if uploaded_file and st.session_state.processed_file != uploaded_file.name:
    with st.spinner("Processing document..."):
        try:
            filename = os.path.basename(uploaded_file.name)
            file_path = os.path.join(UPLOAD_DIR, filename)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            # Step 1: Load
            docs = load_pdf(file_path)

            # Step 2: Chunk
            chunks = split_documents(docs)

            # Step 3: Embeddings
            embeddings = get_embeddings(config.EMBEDDINGS_PROVIDER)

            # Step 4: Vector Store
            vectorstore = create_vector_store(chunks, embeddings)

            st.session_state.processed_file = uploaded_file.name
            st.success("Document processed successfully!")
        except Exception as e:
            st.session_state.processed_file = None
            st.error(f"Failed to process document: {e}")

# Q&A Section
if st.session_state.processed_file:
    st.subheader("💬 Ask a Question")

    question = st.text_input("Enter your question")

    if question:
        try:
            embeddings = get_embeddings(config.EMBEDDINGS_PROVIDER)
            vectorstore = load_vector_store(embeddings)

            retrieval_llm = ChatOpenAI(
                model="openai/gpt-oss-20b:free",
                temperature=0.7,
                openai_api_key=config.get_openrouter_api_key(),
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
        except Exception as e:
            st.error(f"Failed to generate an answer: {e}")

else:
    st.info("Upload a PDF to get started.")


