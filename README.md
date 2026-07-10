# RAG Q&A System 📄🤖

A robust Retrieval-Augmented Generation (RAG) system built with LangChain, Streamlit, and ChromaDB. This application allows users to upload PDF documents and ask questions grounded in the content of those documents, providing accurate answers with citations.

## 🚀 Features

- **PDF Ingestion:** Seamlessly load and process PDF documents.
- **Smart Chunking:** Semantically splits text using `RecursiveCharacterTextSplitter` to maintain context.
- **Vector Search:** Uses **ChromaDB** for efficient similarity and MMR (Maximal Marginal Relevance) retrieval.
- **Multi-Query Retrieval:** Uses an LLM to rewrite user questions into multiple perspectives for better retrieval accuracy.
- **Interactive UI:** Built with **Streamlit** for a clean, user-friendly experience.
- **LLM Integration:** Powered by **OpenRouter** (supporting various models like GPT-4, Claude, or free-tier models).
- **Embeddings:** Supports both OpenAI (via OpenRouter) and local HuggingFace embeddings.

## 🛠️ Tech Stack

- **Framework:** [LangChain](https://github.com/langchain-ai/langchain)
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Vector Store:** [ChromaDB](https://www.trychroma.com/)
- **Embeddings:** [HuggingFace Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **LLM Provider:** [OpenRouter](https://openrouter.ai/)

## 📋 Prerequisites

- Python 3.10+
- An [OpenRouter API Key](https://openrouter.ai/keys)

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd rag_app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your API key:
   ```env
   OPENROUTER_API_KEY=your_sk_or_v1_key_here
   ```

## 🏃 Usage

### 1. Run the Web App (Streamlit)
The most interactive way to use the system:
```bash
streamlit run app.py
```
- Upload a PDF in the sidebar.
- Wait for "Document processed successfully!"
- Type your question and get a grounded answer with sources.

### 2. Run CLI Test
To test the retrieval logic quickly without the UI:
```bash
python main.py
```
*Note: This script uses the `sample.pdf` provided in the root directory.*

## 📂 Project Structure

- `app.py`: Main Streamlit application entry point.
- `main.py`: CLI script for testing core logic.
- `ingest.py`: PDF loading logic.
- `chunking.py`: Text splitting and preprocessing.
- `embeddings.py`: Embedding model configuration.
- `vector_store.py`: ChromaDB initialization and management.
- `retriever.py`: Advanced retrieval strategies (MMR, Multi-Query).
- `rag_chain.py`: LangChain RetrievalQA chain definition.

## 🛡️ License

[MIT](LICENSE)
