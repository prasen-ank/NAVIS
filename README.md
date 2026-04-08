# RAG File Chatbot

This project is a minimal RAG (retrieval-augmented generation) chatbot that:

- Accepts multiple file formats (CSV, Excel, JSON, DOCX, PPTX, plain text).
- Extracts text, chunks it, computes embeddings (using sentence-transformers), and stores vectors in ChromaDB.
- Exposes a FastAPI backend with endpoints to upload files and query the RAG bot.
- Uses Anthropic Claude API (optional) to generate final answers. If no API key is provided, the backend returns a mock response.
- Provides a Streamlit frontend for uploads and chat UI.

## Structure

- backend/app: FastAPI app and helpers
- frontend: Streamlit app
- static: generated images (charts)

## Setup (Windows PowerShell)

1. Create a Python environment and activate it (example with venv):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Create a `.env` file based on `.env.example` and set your `CLAUDE_API_KEY` if you have one.

3. Start the backend:

```powershell
$env:CLAUDE_API_KEY = "your_key_here"; uvicorn backend.app.main:app --reload
```

4. Start the Streamlit frontend:

```powershell
streamlit run frontend/streamlit_app.py
```

## Notes and next steps

- The Anthropic/Claude API wrapper is minimal. Update `CLAUDE_API_URL` and request/response handling if your account requires a different endpoint or headers.
- For production, secure the API key and persist Chroma somewhere durable (S3, network filesystem, or a proper DB-backed Chroma setup).
- The code includes a simple chart generation heuristic. You can extend it to support more chart types and more robust table extraction.

