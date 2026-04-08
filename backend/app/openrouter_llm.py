import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "")
OPENROUTER_SITE_NAME = os.environ.get("OPENROUTER_SITE_NAME", "")
OPENROUTER_LLM_MODEL = os.environ.get("OPENROUTER_LLM_MODEL", "openrouter/free")
OPENROUTER_LLM_URL = os.environ.get("OPENROUTER_LLM_URL", "https://openrouter.ai/api/v1/chat/completions")


def openrouter_chat(messages, reasoning_enabled=True):
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

    payload = {
        "model": OPENROUTER_LLM_MODEL,
        "messages": messages,
        "reasoning": {"enabled": reasoning_enabled}
    }
    logger.info(f"Calling OpenRouter LLM ({OPENROUTER_LLM_MODEL}) with {len(messages)} messages...")
    resp = requests.post(OPENROUTER_LLM_URL, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "choices" not in data or not data["choices"]:
        logger.error(f"OpenRouter LLM API error or unexpected response: {json.dumps(data)[:1000]}")
        raise RuntimeError(f"OpenRouter LLM API error or unexpected response: {data}")
    message = data["choices"][0]["message"]
    logger.info(f"✓ OpenRouter LLM succeeded. Response: {message.get('content', '')[:100]}")
    return message
