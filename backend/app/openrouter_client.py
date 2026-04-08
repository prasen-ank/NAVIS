import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "")
OPENROUTER_SITE_NAME = os.environ.get("OPENROUTER_SITE_NAME", "")

OPENROUTER_EMBED_MODEL = os.environ.get("OPENROUTER_EMBED_MODEL", "openrouter/free")
OPENROUTER_EMBED_URL = os.environ.get("OPENROUTER_EMBED_URL", "https://openrouter.ai/api/v1/embeddings")


def openrouter_embed(texts):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment.")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_SITE_NAME:
        headers["X-OpenRouter-Title"] = OPENROUTER_SITE_NAME

    # OpenRouter free router expects a list of strings for text embeddings
    def batch(iterable, n=100):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    all_embeddings = []
    for chunk in batch(texts, 100):
        # Try a couple of payload shapes to maximize compatibility across router/provider responses
        tried = []
        last_exc = None
        # Variant 1: simple list of strings (most common for embeddings endpoints)
        payload_variants = [
            {"model": OPENROUTER_EMBED_MODEL, "input": chunk, "encoding_format": "float"},
            # Variant 2: per-item 'content' structure (useful for multimodal/embed-vl models)
            {"model": OPENROUTER_EMBED_MODEL, "input": [{"content": [{"type": "text", "text": t}]} for t in chunk], "encoding_format": "float"}
        ]

        for i, payload in enumerate(payload_variants):
            try:
                logger.info(f"Attempt {i+1}: Requesting embeddings for {len(chunk)} texts (variant {i+1}) from OpenRouter...")
                resp = requests.post(OPENROUTER_EMBED_URL, headers=headers, data=json.dumps(payload), timeout=60)
                try:
                    resp.raise_for_status()
                except Exception as e:
                    # Log full response body for debugging
                    body = resp.text[:2000]
                    logger.error(f"OpenRouter returned status {resp.status_code}: {body}")
                    raise
                data = resp.json()
                if "data" not in data:
                    logger.error(f"OpenRouter API unexpected response (no data): {json.dumps(data)[:2000]}")
                    raise RuntimeError(f"OpenRouter API unexpected response: {data}")
                all_embeddings.extend([item["embedding"] for item in data["data"]])
                # success for this chunk
                break
            except Exception as e:
                last_exc = e
                tried.append(str(e))
                logger.debug(f"Variant {i+1} failed for chunk: {e}")
                continue
        else:
            # all variants failed for this chunk
            logger.error(f"All payload variants failed for this chunk. Tried: {tried}")
            raise RuntimeError(f"OpenRouter embedding failed for chunk. Last error: {last_exc}")
    logger.info(f"✓ OpenRouter free router embeddings succeeded for {len(all_embeddings)} vectors")
    return all_embeddings
