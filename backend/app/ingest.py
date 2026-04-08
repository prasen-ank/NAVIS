import io
import json
import math
import os
import time
import logging
from typing import List
import pandas as pd
from docx import Document
from pptx import Presentation
from sentence_transformers import SentenceTransformer
import numpy as np
from .claude_client import claude_embed

logger = logging.getLogger(__name__)

# Lazy-load the embedding model to avoid long blocking downloads at import time
_MODEL = None

def get_model():
    """Return a loaded SentenceTransformer instance, loading on first use.

    Honors the LOCAL_EMBEDDING_MODEL environment variable to load from a local
    directory (useful if you pre-cloned the model with git-lfs).
    Implements simple retry/backoff on network errors.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model_source = os.environ.get("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    attempts = 3
    delay = 5
    last_exc = None
    for i in range(attempts):
        try:
            _MODEL = SentenceTransformer(model_source)
            return _MODEL
        except Exception as e:
            last_exc = e
            # If it's the last attempt, reraise after logging
            if i + 1 < attempts:
                time.sleep(delay)
                delay *= 2
            else:
                raise
    # fallback (should not reach)
    raise last_exc


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def _extract_text_from_docx(content: bytes) -> str:
    f = io.BytesIO(content)
    doc = Document(f)
    paras = [p.text for p in doc.paragraphs]
    return "\n".join(paras)


def _extract_text_from_pptx(content: bytes) -> str:
    f = io.BytesIO(content)
    prs = Presentation(f)
    texts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                texts.append(shape.text)
    return "\n".join(texts)


def _extract_text_from_csv(content: bytes) -> str:
    f = io.BytesIO(content)
    try:
        df = pd.read_csv(f)
    except Exception:
        f.seek(0)
        df = pd.read_csv(f, encoding="latin1")
    return df.to_csv(index=False)


def _extract_text_from_excel(content: bytes) -> str:
    f = io.BytesIO(content)
    df = pd.read_excel(f, sheet_name=None)
    parts = []
    for sheet, data in df.items():
        parts.append(f"Sheet: {sheet}\n")
        parts.append(data.to_csv(index=False))
    return "\n".join(parts)


def _extract_text_from_json(content: bytes) -> str:
    try:
        obj = json.loads(content.decode("utf-8"))
    except Exception:
        obj = json.loads(content.decode("latin1"))
    return json.dumps(obj, indent=2)


def _extract_text_from_text(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except Exception:
        return content.decode("latin1")


def _embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    
    from .jina_client import jina_embed
    try:
        start = time.time()
        emb_resp = jina_embed(texts)
        if emb_resp is not None:
            elapsed = time.time() - start
            logger.info(f"✓ Jina AI embeddings succeeded in {elapsed:.2f}s for {len(emb_resp)} vectors")
            return emb_resp
    except Exception as e:
        logger.warning(f"Jina AI embeddings failed: {e}. Falling back to local SentenceTransformer model.")

    # Fall back to local SentenceTransformer model
    logger.info(f"Using local SentenceTransformer model for {len(texts)} texts...")
    start = time.time()
    model = get_model()
    embs = model.encode(texts, show_progress_bar=False)
    elapsed = time.time() - start
    logger.info(f"✓ Local embeddings completed in {elapsed:.2f}s for {len(embs)} vectors")
    return embs.tolist() if isinstance(embs, np.ndarray) else embs


def process_and_ingest(collection, filename: str, content: bytes):
    name = filename.lower()
    if name.endswith(".csv"):
        text = _extract_text_from_csv(content)
    elif name.endswith(":xls") or name.endswith(".xlsx") or name.endswith(".xlsm"):
        text = _extract_text_from_excel(content)
    elif name.endswith(".docx"):
        text = _extract_text_from_docx(content)
    elif name.endswith(".pptx"):
        text = _extract_text_from_pptx(content)
    elif name.endswith(".json"):
        text = _extract_text_from_json(content)
    else:
        # fallback: try text
        text = _extract_text_from_text(content)

    chunks = _chunk_text(text)
    embeddings = _embed_texts(chunks)

    # add to chroma collection in batches to avoid Chroma max-batch limits
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]

    def batch_add(col, ids_list, docs_list, metas_list, embs_list, batch_size=4000):
        total = len(ids_list)
        for i in range(0, total, batch_size):
            j = min(i + batch_size, total)
            logger.info(f"Adding batch {i}-{j} to collection {col.name}...")
            col.add(ids=ids_list[i:j], documents=docs_list[i:j], metadatas=metas_list[i:j], embeddings=embs_list[i:j])

    if embeddings is None:
        # nothing to add
        return {"file": filename, "chunks": 0}

    batch_add(collection, ids, chunks, metadatas, embeddings, batch_size=4000)

    return {"file": filename, "chunks": len(chunks)}
