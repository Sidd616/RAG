# RAG Document Q&A

Upload a PDF, ask a question, get an answer grounded in the document along with the pages it came from. A small retrieval-augmented generation app built with LangChain, Chroma and Streamlit.

I built this to understand RAG properly instead of just reading about it: how chunk size affects retrieval, why one phrasing of a question is often not enough, and how to make the model say "I don't know" instead of guessing.

## What it does

- Takes a PDF upload in the sidebar, splits it into chunks, embeds them and stores them in a local Chroma database.
- Rewrites your question into several variations with the LLM (multi-query retrieval), searches with all of them, and merges the results. This catches answers that a single phrasing would miss.
- Answers strictly from the retrieved chunks. If the answer isn't in the document, the prompt tells the model to say so rather than make something up.
- Lists the source file and page numbers under every answer.

## How it works

```
PDF ──► PyPDFLoader (one document per page, page number kept as metadata)
    ──► RecursiveCharacterTextSplitter (800 characters, 150 overlap)
    ──► embeddings ──► Chroma, persisted to data/vector_store

question ──► MultiQueryRetriever (LLM writes query variants, top-k for each, union)
         ──► RetrievalQA "stuff" chain with a grounded prompt
         ──► answer + source pages
```

Embeddings come from OpenRouter (`openai/text-embedding-3-small`) or run locally with `sentence-transformers/all-MiniLM-L6-v2`, chosen with `EMBEDDINGS_PROVIDER`. The chat model is `openai/gpt-oss-20b:free` through OpenRouter, so answering costs nothing while you experiment.

`retriever.py` also has a plain similarity retriever and an MMR retriever if you want to swap strategies and compare.

## Run it locally

```bash
git clone https://github.com/Sidd616/RAG.git
cd RAG
pip install -r requirements.txt
echo "OPENROUTER_API_KEY=your_key" > .env    # optional: EMBEDDINGS_PROVIDER=huggingface
streamlit run app.py
```

`python main.py` runs a quick command-line check against `sample.pdf` without the UI.

## Limitations, and what I'd do next

- One document at a time. Uploading a new PDF replaces the store.
- No conversation memory. Each question is answered on its own.
- I want to build a small set of question-answer pairs and measure retrieval hit rate for the three retriever strategies rather than comparing by feel.
- A reranker between retrieval and generation would probably help more than tuning chunk size further.
