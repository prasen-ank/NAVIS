import os
import logging
import requests
import json

logger = logging.getLogger(__name__)

JINA_API_KEY = os.environ.get("JINA_API_KEY")
JINA_MODEL = os.environ.get("JINA_MODEL", "jina-embeddings-v5-text-small")
JINA_URL = "https://api.jina.ai/v1/embeddings"
JINA_LLM_MODEL = os.environ.get("JINA_LLM_MODEL", "jina-vlm")
JINA_LLM_URL = os.environ.get("JINA_LLM_URL", "https://api.jina.ai/v1/chat/completions")

jina_client = None
if JINA_API_KEY:
    jina_client = True  # Just a flag since we use requests directly

def jina_embed(texts, task="retrieval.query", normalized=True, timeout=30):
    if not jina_client or not JINA_API_KEY:
        raise RuntimeError("Jina AI client not initialized. Set JINA_API_KEY in your .env file.")
    
    try:
        logger.info(f"Using Jina AI for {len(texts)} texts (model={JINA_MODEL}, task={task})...")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}"
        }
        
        data = {
            "model": JINA_MODEL,
            "task": task,
            "normalized": normalized,
            "input": texts
        }
        
        response = requests.post(JINA_URL, headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        embeddings = result.get("data", [])
        
        if not embeddings:
            raise RuntimeError("No embeddings returned from Jina AI")
        
        # Extract embeddings from the response
        embedding_vectors = [item["embedding"] for item in embeddings]
        
        logger.info(f"✓ Jina AI embeddings succeeded for {len(embedding_vectors)} vectors")
        return embedding_vectors
    except requests.exceptions.HTTPError as http_err:
        response = http_err.response
        body = response.text if response is not None else "<no body>"
        logger.error(f"Jina AI embedding HTTP error: {http_err}. response={body}")
        raise RuntimeError(f"Jina AI embedding failed: {body}") from http_err
    except Exception as e:
        logger.error(f"Jina AI embedding failed: {e}")
        raise


def _normalize_jina_chat_messages(messages):
    if not isinstance(messages, list):
        raise ValueError("Messages must be a list for Jina AI chat.")

    normalized = []
    system_buffer = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            system_buffer.append(content)
            continue
        if role not in {"user", "assistant"}:
            continue

        if system_buffer and role == "user":
            content = "\n\n".join(system_buffer + [content])
            system_buffer = []

        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] = normalized[-1]["content"].strip() + "\n\n" + content.strip()
        else:
            normalized.append({"role": role, "content": content})

    if system_buffer:
        if normalized and normalized[0]["role"] == "user":
            normalized[0]["content"] = "\n\n".join(system_buffer + [normalized[0]["content"]])
        else:
            normalized.insert(0, {"role": "user", "content": "\n\n".join(system_buffer)})

    collapsed = []
    for msg in normalized:
        if collapsed and collapsed[-1]["role"] == msg["role"]:
            collapsed[-1]["content"] = collapsed[-1]["content"].strip() + "\n\n" + msg["content"].strip()
        else:
            collapsed.append(msg)

    if collapsed and collapsed[0]["role"] == "assistant":
        raise ValueError("Jina chat requires the conversation to start with a user message.")

    return collapsed


def jina_chat(messages, temperature=0.2, timeout=60):
    if not jina_client or not JINA_API_KEY:
        raise RuntimeError("Jina AI client not initialized. Set JINA_API_KEY in your .env file.")

    try:
        normalized_messages = _normalize_jina_chat_messages(messages)
        logger.info(f"Using Jina AI chat for {len(normalized_messages)} messages (model={JINA_LLM_MODEL})...")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}"
        }

        payload = {
            "model": JINA_LLM_MODEL,
            "messages": normalized_messages,
            "temperature": temperature
        }

        response = requests.post(JINA_LLM_URL, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()

        choices = result.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message")
                if isinstance(message, dict) and "content" in message:
                    return message
                if "text" in first_choice:
                    return {"role": "assistant", "content": first_choice["text"]}

        if isinstance(result, dict) and "message" in result and isinstance(result["message"], dict):
            return result["message"]
        if isinstance(result, dict) and "output_text" in result:
            return {"role": "assistant", "content": result["output_text"]}

        raise RuntimeError(f"Unexpected Jina AI chat response: {json.dumps(result)[:1000]}")
    except requests.exceptions.HTTPError as http_err:
        response = http_err.response
        body = response.text if response is not None else "<no body>"
        logger.error(f"Jina AI chat HTTP error: {http_err}. response={body}")
        raise RuntimeError(f"Jina AI chat failed: {body}") from http_err
    except Exception as e:
        logger.error(f"Jina AI chat failed: {e}")
        raise