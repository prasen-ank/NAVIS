import os
from dotenv import load_dotenv
load_dotenv()
import uuid
import time
import logging
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from .ingest import process_and_ingest
from .jina_client import jina_chat, jina_embed
from .openrouter_llm import openrouter_chat
from .openrouter_client import openrouter_embed
from .db import get_collection, clear_collection
from .utils import is_chart_request, generate_chart_from_docs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), namespace: str = Form("default")):
    """Accept a file, extract text and ingest into Chroma under a namespace"""
    start = time.time()
    try:
        logger.info(f"Upload starting: file={file.filename}, namespace={namespace}")
        content = await file.read()
        collection = get_collection(namespace)
        ingest_result = process_and_ingest(collection, file.filename, content)
        elapsed = time.time() - start
        logger.info(f"Upload completed in {elapsed:.2f}s: {ingest_result}")
        return JSONResponse({"status": "ok", "ingested": ingest_result, "elapsed_seconds": elapsed})
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Upload failed in {elapsed:.2f}s: {e}", exc_info=True)
        return JSONResponse(
            {"status": "error", "message": str(e), "elapsed_seconds": elapsed},
            status_code=400
        )


@app.post("/chat")
async def chat(namespace: str = Form("default"), question: str = Form(...)):
    """Run a RAG query: retrieve top docs from chroma and ask Claude."""
    start = time.time()
    try:
        logger.info(f"Chat starting: namespace={namespace}, question={question[:50]}...")
        collection = get_collection(namespace)

        # search
        search_start = time.time()
        query_embeddings = None
        try:
            query_embeddings = jina_embed([question])
        except Exception as e:
            logger.warning(f"Jina query embedding failed: {e}. Falling back to OpenRouter embedding.")
            query_embeddings = openrouter_embed([question])

        if not query_embeddings:
            raise RuntimeError("Failed to generate query embeddings for search.")

        docs = collection.query(query_embeddings=query_embeddings, n_results=5)
        search_elapsed = time.time() - search_start
        logger.info(f"Search completed in {search_elapsed:.2f}s")
        
        hits = docs.get("documents", [[ ]])[0]
        metadatas = docs.get("metadatas", [[ ]])[0]

        # Build context
        context_parts = []
        for i, d in enumerate(hits):
            md = metadatas[i] if i < len(metadatas) else {}
            title = md.get("source", f"doc_{i}")
            context_parts.append(f"Source: {title}\n{d}")

        context = "\n\n---\n\n".join(context_parts)

        # If the user asked for a chart/diagram, try to return an image
        if is_chart_request(question):
            chart_start = time.time()
            img_path = generate_chart_from_docs(hits, namespace)
            if img_path:
                chart_elapsed = time.time() - chart_start
                total_elapsed = time.time() - start
                logger.info(f"Chart generated in {chart_elapsed:.2f}s, total {total_elapsed:.2f}s")
                return JSONResponse({
                    "type": "image",
                    "url": f"/static/{os.path.basename(img_path)}",
                    "elapsed_seconds": total_elapsed
                })

        # Prepare messages for LLM
        system_prompt = "You are an assistant that answers questions using the provided context. Answer concisely and cite sources when possible."
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        def get_llm_response(messages):
            try:
                return jina_chat(messages)
            except Exception as e:
                logger.warning(f"Jina LLM failed: {e}. Falling back to OpenRouter.")
                return openrouter_chat(messages)

        llm_start = time.time()
        resp = get_llm_response(messages)
        llm_elapsed = time.time() - llm_start
        total_elapsed = time.time() - start
        return JSONResponse({
            "type": "text",
            "answer": resp.get("content", ""),
            "reasoning_details": resp.get("reasoning_details", None),
            "elapsed_seconds": total_elapsed,
            "llm_elapsed_seconds": llm_elapsed
        })
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Chat failed in {elapsed:.2f}s: {e}", exc_info=True)
        return JSONResponse(
            {"status": "error", "message": str(e), "elapsed_seconds": elapsed},
            status_code=400
        )


@app.post("/clear")
async def clear_namespace(namespace: str = Form("default")):
    start = time.time()
    try:
        logger.info(f"Clearing Chroma namespace: {namespace}")
        clear_collection(namespace)
        elapsed = time.time() - start
        return JSONResponse({"status": "ok", "namespace": namespace, "elapsed_seconds": elapsed})
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Clear failed in {elapsed:.2f}s: {e}", exc_info=True)
        return JSONResponse(
            {"status": "error", "message": str(e), "elapsed_seconds": elapsed},
            status_code=400
        )


@app.get("/static/{filename}")
async def static_file(filename: str):
    path = os.path.join("static", filename)
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
