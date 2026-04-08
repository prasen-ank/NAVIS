import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")
CLAUDE_API_URL = os.environ.get("CLAUDE_API_URL", "https://api.anthropic.com/v1/complete")
DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-2")

# Auto-detect embedding endpoint if Claude key is set, or use explicit override
CLAUDE_EMBEDDING_URL = os.environ.get(
    "CLAUDE_EMBEDDING_URL",
    "https://api.anthropic.com/v1/embeddings" if CLAUDE_API_KEY else None
)
FORCE_CLAUDE_EMBEDDINGS = os.environ.get("FORCE_CLAUDE_EMBEDDINGS", "false").lower() == "true"


def claude_query(prompt: str, max_tokens: int = 800, temperature: float = 0.2) -> str:
    """Query Anthropic Claude. If API key not available, return a fallback echo response."""
    if not CLAUDE_API_KEY:
        # fallback: return short echo so app remains functional without key
        return f"(mock answer) I received the prompt of length {len(prompt)} characters.\n\nExcerpt:\n{prompt[:500]}"

    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "Content-Type": "application/json",
    }
    body = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "max_tokens_to_sample": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(CLAUDE_API_URL, headers=headers, json=body, timeout=30)
    try:
        resp.raise_for_status()
        data = resp.json()
        # anthopic response shape may vary; try common fields
        if "completion" in data:
            return data["completion"]
        if "completion" in data.get("completion", {}):
            return data["completion"]
        # older responses: 'text' or 'completion'
        if "text" in data:
            return data["text"]
        # fallback: stringify
        return json.dumps(data)
    except Exception as e:
        return f"(error calling Claude) {e}: {resp.text if resp is not None else ''}"


def claude_embed(texts, timeout: int = 30):
    """Call a Claude embedding endpoint to get embeddings for texts.

    Uses the Anthropic embeddings API (or custom endpoint if CLAUDE_EMBEDDING_URL is set).
    Tries a few common request body shapes since the API may vary.

    Returns a list of embedding vectors (list of lists of floats).
    Raises an exception if embeddings fail.
    """
    if not CLAUDE_EMBEDDING_URL or not CLAUDE_API_KEY:
        return None

    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "Content-Type": "application/json",
    }

    # Try common request shapes for Anthropic embeddings endpoint
    body_variants = [
        {"input": texts, "model": "claude-3-5-sonnet-20241022"},  # Sonnet is fast for embeddings
        {"input": texts},
        {"texts": texts},
        {"documents": texts},
    ]
    last_exc = None
    for i, body in enumerate(body_variants):
        try:
            logger.debug(f"Claude embed attempt {i+1}: body keys={list(body.keys())}, text_count={len(texts)}")
            resp = requests.post(CLAUDE_EMBEDDING_URL, headers=headers, json=body, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            logger.debug(f"Claude embed response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            
            # Try common response shapes
            if isinstance(data, dict):
                if "embeddings" in data and isinstance(data["embeddings"], list):
                    logger.info(f"Claude embeddings succeeded with body variant {i+1}: got {len(data['embeddings'])} embeddings")
                    return data["embeddings"]
                if "data" in data and isinstance(data["data"], list):
                    out = []
                    for item in data["data"]:
                        if isinstance(item, dict) and "embedding" in item:
                            out.append(item["embedding"])
                    if out:
                        logger.info(f"Claude embeddings succeeded with body variant {i+1}: got {len(out)} embeddings from data")
                        return out
                if "embedding" in data and isinstance(data["embedding"], list):
                    logger.info(f"Claude embeddings succeeded with body variant {i+1}: got single embedding")
                    return [data["embedding"]]
            # Fallback: try to interpret top-level list
            if isinstance(data, list) and all(isinstance(el, list) for el in data):
                logger.info(f"Claude embeddings succeeded with body variant {i+1}: got list-of-lists")
                return data
        except Exception as e:
            last_exc = e
            logger.debug(f"Claude embed attempt {i+1} failed: {e}")
            continue
    
    # If we reach here, embedding via Claude failed
    raise RuntimeError(f"Claude embedding failed after {len(body_variants)} attempts. Last error: {last_exc}")
