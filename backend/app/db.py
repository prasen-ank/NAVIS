import os
import logging
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)
CHROMA_PERSIST_PATH = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

_client = None


def _get_client():
    """Create or return a chromadb client.

    Uses the newer Settings flags for persistent mode. Falls back to a default
    client constructor if the Settings route raises the legacy configuration
    ValueError.
    """
    global _client
    if _client is not None:
        return _client

    os.makedirs(CHROMA_PERSIST_PATH, exist_ok=True)
    try:
        settings = Settings(is_persistent=True, persist_directory=CHROMA_PERSIST_PATH)
        _client = chromadb.Client(settings)
    except ValueError as e:
        # Older/newer mismatch: try the default client constructor
        # and let chroma choose defaults. Re-raise if that fails.
        try:
            _client = chromadb.Client()
        except Exception:
            raise
    return _client


def get_collection(name: str):
    c = _get_client()
    try:
        return c.get_collection(name)
    except Exception:
        return c.create_collection(name)


def clear_collection(name: str):
    c = _get_client()
    try:
        c.delete_collection(name)
        logger.info(f"Cleared Chroma collection: {name}")
    except Exception as e:
        logger.warning(f"Chroma collection delete failed for {name}: {e}")
    try:
        return c.create_collection(name)
    except Exception as e:
        logger.error(f"Failed to recreate Chroma collection {name}: {e}")
        raise