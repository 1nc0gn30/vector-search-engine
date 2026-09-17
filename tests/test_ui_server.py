"""Integration tests for Studio HTTP server and REST API endpoints."""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from pathlib import Path
import pytest

from vector_search_engine.models import CollectionConfig, DistanceMetric, IndexType, VectorItem
from vector_search_engine.collection import VectorCollection
from vector_search_engine.ui_server import start_ui_server, _COLLECTIONS_REGISTRY, _REGISTRY_LOCK


@pytest.fixture(scope="module")
def ui_server():
    """Start background HTTP server on dynamic port for testing."""
    # Seed initial test collection
    test_cfg = CollectionConfig(
        name="test_ui_col",
        dimension=4,
        metric=DistanceMetric.COSINE,
        index_type=IndexType.FLAT,
    )
    col = VectorCollection(config=test_cfg)
    col.insert(VectorItem("doc1", [1.0, 0.0, 0.0, 0.0], {"category": "ai"}, "Attention is all you need"))
    col.insert(VectorItem("doc2", [0.0, 1.0, 0.0, 0.0], {"category": "database"}, "PostgreSQL ACID database"))

    with _REGISTRY_LOCK:
        _COLLECTIONS_REGISTRY.clear()
        _COLLECTIONS_REGISTRY["test_ui_col"] = col

    server = start_ui_server(host="127.0.0.1", port=0, background=True)
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    yield base_url
    server.shutdown()
    server.server_close()


def _request_json(url: str, method: str = "GET", data: dict = None):
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    body_bytes = json.dumps(data).encode("utf-8") if data is not None else None
    with urllib.request.urlopen(req, data=body_bytes, timeout=5) as response:
        status = response.status
        content_type = response.headers.get("Content-Type", "")
        body = response.read().decode("utf-8")
        if "application/json" in content_type:
            return status, json.loads(body)
        return status, body


def test_ui_server_static_and_health(ui_server):
    # Test index.html serving
    status, body = _request_json(f"{ui_server}/")
    assert status == 200
    assert "<!DOCTYPE html>" in body
    assert "Vector Studio" in body

    # Test health endpoint
    status, data = _request_json(f"{ui_server}/api/health")
    assert status == 200
    assert data["status"] == "ok"
    assert data["service"] == "vector-search-engine"

    # Test diagnostics endpoint
    status, data = _request_json(f"{ui_server}/api/diagnostics")
    assert status == 200
    assert "platform" in data
    assert data["collections_count"] >= 1


def test_ui_server_collections_crud(ui_server):
    # List collections
    status, data = _request_json(f"{ui_server}/api/collections")
    assert status == 200
    assert len(data["collections"]) >= 1
    assert any(c["name"] == "test_ui_col" for c in data["collections"])

    # Create new collection
    new_col_payload = {
        "name": "new_created_col",
        "dimension": 4,
        "metric": "euclidean",
        "index_type": "hnsw",
    }
    status, data = _request_json(f"{ui_server}/api/collections", method="POST", data=new_col_payload)
    assert status == 201
    assert data["status"] == "created"
    assert data["name"] == "new_created_col"

    # Insert items
    insert_payload = {
        "documents": [
            {"id": "item1", "vector": [0.5, 0.5, 0.5, 0.5], "document": "Distributed vector indexes", "metadata": {"tag": "fast"}},
            {"id": "item2", "vector": [0.1, 0.2, 0.3, 0.4], "document": "Quantum physics simulation", "metadata": {"tag": "science"}},
        ]
    }
    status, data = _request_json(f"{ui_server}/api/collections/new_created_col/insert", method="POST", data=insert_payload)
    assert status == 200
    assert data["count"] == 2

    # Query items
    query_payload = {"vector": [0.5, 0.5, 0.5, 0.5], "k": 2}
    status, data = _request_json(f"{ui_server}/api/collections/new_created_col/query", method="POST", data=query_payload)
    assert status == 200
    assert len(data["results"]) == 2
    assert data["results"][0]["id"] == "item1"

    # Hybrid Search
    hybrid_payload = {
        "query_text": "quantum physics",
        "query_vector": [0.1, 0.2, 0.3, 0.4],
        "k": 2,
        "alpha": 0.5,
    }
    status, data = _request_json(f"{ui_server}/api/collections/new_created_col/hybrid", method="POST", data=hybrid_payload)
    assert status == 200
    assert len(data["results"]) >= 1

    # Get Stats & Items
    status, stats = _request_json(f"{ui_server}/api/collections/new_created_col/stats")
    assert status == 200
    assert stats["count"] == 2

    status, items_resp = _request_json(f"{ui_server}/api/collections/new_created_col/items")
    assert status == 200
    assert items_resp["count"] == 2

    # Delete Collection
    status, del_resp = _request_json(f"{ui_server}/api/collections/new_created_col", method="DELETE")
    assert status == 200
    assert del_resp["status"] == "deleted"


def test_ui_server_benchmark_runner(ui_server):
    bench_payload = {
        "num_vectors": 100,
        "dimension": 8,
        "queries_count": 10,
        "k": 5,
        "metric": "cosine",
    }
    status, res = _request_json(f"{ui_server}/api/benchmark", method="POST", data=bench_payload)
    assert status == 200
    assert "hnsw_qps" in res
    assert "flat_qps" in res
    assert "recall_at_k" in res
    assert res["recall_at_k"] >= 0.5
    assert res["build_time_ms"] > 0
