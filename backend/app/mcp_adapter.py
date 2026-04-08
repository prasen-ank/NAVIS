"""Placeholder adapter for future MCP server data source integration.

When you connect a real-time MCP data source later, implement the fetch_mcp_documents
function to return a list of (id, text, metadata) that can be added to the Chroma collection.
"""
from typing import List, Dict, Tuple


def fetch_mcp_documents(source_config: Dict) -> List[Tuple[str, str, Dict]]:
    """Fetch documents from an MCP data source.

    Returns a list of tuples: (id, text, metadata)

    Implement this to call your MCP server and return documents to ingest.
    """
    # TODO: implement connection to MCP server. For now return empty list.
    return []
