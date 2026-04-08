import os
import logging
import voyageai

logger = logging.getLogger(__name__)

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
VOYAGE_MODEL = os.environ.get("VOYAGE_MODEL", "voyage-4-large")

voyage_client = None
if VOYAGE_API_KEY:
    try:
        voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Voyage AI client: {e}")


def voyage_embed(texts, input_type="document", timeout=30):
    if not voyage_client:
        raise RuntimeError("Voyage AI client not initialized. Set VOYAGE_API_KEY in your .env file.")
    try:
        logger.info(f"Using Voyage AI for {len(texts)} texts (model={VOYAGE_MODEL}, input_type={input_type})...")
        result = voyage_client.embed(texts, model=VOYAGE_MODEL, input_type=input_type)
        logger.info(f"✓ Voyage AI embeddings succeeded for {len(result.embeddings)} vectors")
        return result.embeddings
    except Exception as e:
        logger.error(f"Voyage AI embedding failed: {e}")
        raise
