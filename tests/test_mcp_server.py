"""Unit and integration tests for Model Context Protocol (MCP) Server."""

from __future__ import annotations

import json
import math
from pathlib import Path
import pytest

from vector_search_engine.mcp_server import (
    MCPServer,
    handle_jsonrpc_request,
    simple_text_embedder,
    SERVER_NAME,
    SERVER_VERSION,
)


def test_simple_text_embedder():
    # Determinism check
    v1 = simple_text_embedder("Deep learning architectures", dimension=32)
    v2 = simple_text_embedder("Deep learning architectures", dimension=32)
    assert len(v1) == 32
    assert v1 == v2

    # Vector norm is 1.0 (normalized)
    norm = math.sqrt(sum(x * x for x in v1))
    assert math.isclose(norm, 1.0, abs_tol=1e-3)

    # Empty text returns zero vector
    v_empty = simple_text_embedder("", dimension=16)
    assert len(v_empty) == 16
    assert all(x == 0.0 for x in v_empty)


def test_mcp_server_initialize_and_tools_list(temp_dir: Path):
    server = MCPServer(data_dir=temp_dir)

    # Test initialize request
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }
    resp = handle_jsonrpc_request(init_req, server=server)
    assert resp["id"] == 1
    assert "result" in resp
    assert resp["result"]["serverInfo"]["name"] == SERVER_NAME
    assert resp["result"]["serverInfo"]["version"] == SERVER_VERSION

    # Test tools/list request
    list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    resp_list = handle_jsonrpc_request(list_req, server=server)
    assert resp_list["id"] == 2
    tools = resp_list["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    assert "vector_create_collection" in tool_names
    assert "vector_upsert" in tool_names
    assert "vector_query" in tool_names
    assert "vector_hybrid_search" in tool_names
    assert "vector_collection_stats" in tool_names


def test_mcp_server_collection_tools_lifecycle(temp_dir: Path):
    server = MCPServer(data_dir=temp_dir)

    # 1. Create collection
    create_call = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "vector_create_collection",
            "arguments": {
                "name": "mcp_ai_docs",
                "dimension": 4,
                "metric": "cosine",
                "index_type": "hnsw",
            },
        },
    }
    resp_create = handle_jsonrpc_request(create_call, server=server)
    assert resp_create["id"] == 10
    assert not resp_create["result"].get("isError", False)
    content = json.loads(resp_create["result"]["content"][0]["text"])
    assert content["status"] == "success"

    # 2. Insert documents
    insert_call = {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {
            "name": "vector_upsert",
            "arguments": {
                "collection": "mcp_ai_docs",
                "id": "doc_1",
                "vector": [1.0, 0.0, 0.0, 0.0],
                "document": "Transformer architecture for NLP",
                "metadata": {"topic": "ai"},
            },
        },
    }
    resp_insert = handle_jsonrpc_request(insert_call, server=server)
    assert resp_insert["id"] == 11
    insert_data = json.loads(resp_insert["result"]["content"][0]["text"])
    assert insert_data["status"] == "success"

    # Insert second doc
    insert_call2 = {
        "jsonrpc": "2.0",
        "id": 12,
        "method": "tools/call",
        "params": {
            "name": "vector_upsert",
            "arguments": {
                "collection": "mcp_ai_docs",
                "id": "doc_2",
                "vector": [0.0, 1.0, 0.0, 0.0],
                "document": "Quantum entanglement in physics",
                "metadata": {"topic": "physics"},
            },
        },
    }
    handle_jsonrpc_request(insert_call2, server=server)

    # 3. Search vectors
    search_call = {
        "jsonrpc": "2.0",
        "id": 13,
        "method": "tools/call",
        "params": {
            "name": "vector_query",
            "arguments": {
                "collection": "mcp_ai_docs",
                "vector": [1.0, 0.0, 0.0, 0.0],
                "k": 2,
            },
        },
    }
    resp_search = handle_jsonrpc_request(search_call, server=server)
    search_data = json.loads(resp_search["result"]["content"][0]["text"])
    assert len(search_data["results"]) == 2
    assert search_data["results"][0]["id"] == "doc_1"

    # 4. Hybrid search
    hybrid_call = {
        "jsonrpc": "2.0",
        "id": 14,
        "method": "tools/call",
        "params": {
            "name": "vector_hybrid_search",
            "arguments": {
                "collection": "mcp_ai_docs",
                "query_text": "transformer NLP",
                "alpha": 0.5,
                "k": 2,
            },
        },
    }
    resp_hybrid = handle_jsonrpc_request(hybrid_call, server=server)
    hybrid_data = json.loads(resp_hybrid["result"]["content"][0]["text"])
    assert len(hybrid_data["results"]) >= 1

    # 5. Get stats
    stats_call = {
        "jsonrpc": "2.0",
        "id": 15,
        "method": "tools/call",
        "params": {
            "name": "vector_collection_stats",
            "arguments": {"collection": "mcp_ai_docs"},
        },
    }
    resp_stats = handle_jsonrpc_request(stats_call, server=server)
    stats_data = json.loads(resp_stats["result"]["content"][0]["text"])
    assert stats_data["count"] == 2


def test_mcp_server_error_handling(temp_dir: Path):
    server = MCPServer(data_dir=temp_dir)

    # Unknown method
    bad_method = {"jsonrpc": "2.0", "id": 99, "method": "unknown_function"}
    resp = handle_jsonrpc_request(bad_method, server=server)
    assert "error" in resp
    assert resp["error"]["code"] == -32601

    # Invalid tool call
    bad_tool = {
        "jsonrpc": "2.0",
        "id": 100,
        "method": "tools/call",
        "params": {"name": "non_existent_tool", "arguments": {}},
    }
    resp_tool = handle_jsonrpc_request(bad_tool, server=server)
    assert "error" in resp_tool or resp_tool.get("result", {}).get("isError") is True
