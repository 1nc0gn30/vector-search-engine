"""Comprehensive unit tests for vector indexes (Flat, HNSW, IVF) and filter engine."""

from __future__ import annotations

import math
import pytest

from vector_search_engine.models import (
    DistanceMetric,
    HNSWConfig,
    IndexType,
    IVFConfig,
    VectorItem,
)
from vector_search_engine.indexes import (
    BaseIndex,
    FlatIndex,
    HNSWIndex,
    IVFIndex,
    evaluate_filter,
)
from vector_search_engine.metrics import generate_random_vector


def test_evaluate_filter_operators():
    meta = {
        "title": "Attention Is All You Need",
        "author": "Vaswani",
        "year": 2017,
        "citations": 120000,
        "category": "nlp",
        "tags": ["transformer", "deep-learning", "attention"],
        "active": True,
    }

    # Equality & operators
    assert evaluate_filter(meta, {"year": 2017})
    assert evaluate_filter(meta, {"year": {"$eq": 2017}})
    assert evaluate_filter(meta, {"year": {"$gt": 2015}})
    assert evaluate_filter(meta, {"year": {"$gte": 2017}})
    assert evaluate_filter(meta, {"year": {"$lt": 2020}})
    assert evaluate_filter(meta, {"year": {"$lte": 2017}})
    assert evaluate_filter(meta, {"year": {"$ne": 2018}})

    # Membership
    assert evaluate_filter(meta, {"category": {"$in": ["nlp", "vision", "rl"]}})
    assert evaluate_filter(meta, {"category": {"$nin": ["database", "systems"]}})

    # Arrays
    assert evaluate_filter(meta, {"tags": {"$contains": "transformer"}})
    assert evaluate_filter(meta, {"tags": {"$all": ["transformer", "attention"]}})
    assert not evaluate_filter(meta, {"tags": {"$all": ["transformer", "cnn"]}})

    # Existence & Regex
    assert evaluate_filter(meta, {"citations": {"$exists": True}})
    assert evaluate_filter(meta, {"missing_key": {"$exists": False}})
    assert evaluate_filter(meta, {"title": {"$regex": "^Attention"}})
    assert not evaluate_filter(meta, {"title": {"$regex": "^BERT"}})

    # Logical operators
    assert evaluate_filter(meta, {"$and": [{"year": {"$gte": 2015}}, {"category": "nlp"}]})
    assert evaluate_filter(meta, {"$or": [{"year": 2020}, {"category": "nlp"}]})
    assert not evaluate_filter(meta, {"$and": [{"year": 2020}, {"category": "nlp"}]})
    assert evaluate_filter(meta, {"$not": {"year": 2020}})
    assert evaluate_filter(meta, {"$nor": [{"year": 2020}, {"category": "cv"}]})


def test_flat_index_crud_and_search():
    index = FlatIndex(dimension=4, metric=DistanceMetric.COSINE)
    assert index.size() == 0

    items = [
        VectorItem("id1", [1.0, 0.0, 0.0, 0.0], {"cat": "a", "val": 10}),
        VectorItem("id2", [0.0, 1.0, 0.0, 0.0], {"cat": "b", "val": 20}),
        VectorItem("id3", [0.7071, 0.7071, 0.0, 0.0], {"cat": "a", "val": 30}),
    ]
    index.add_batch(items)
    assert index.size() == 3
    assert index.get("id1") is not None
    assert index.get("non_existent") is None

    # Search closest to [1.0, 0.0, 0.0, 0.0]
    res = index.search([1.0, 0.0, 0.0, 0.0], k=2, include_vector=True)
    assert len(res) == 2
    assert res[0].id == "id1"
    assert math.isclose(res[0].score, 1.0, abs_tol=1e-3)
    assert res[0].vector == [1.0, 0.0, 0.0, 0.0]
    assert res[1].id == "id3"

    # Search with filter
    filtered_res = index.search(
        [1.0, 0.0, 0.0, 0.0],
        k=5,
        filter_expr={"cat": "b"},
    )
    assert len(filtered_res) == 1
    assert filtered_res[0].id == "id2"

    # Delete
    assert index.delete("id1") is True
    assert index.size() == 2
    assert index.delete("id1") is False

    # Stats
    stats = index.get_stats()
    assert stats["count"] == 2
    assert stats["dimension"] == 4

    # Clear
    index.clear()
    assert index.size() == 0


def test_flat_index_dimension_mismatch():
    index = FlatIndex(dimension=3)
    with pytest.raises(ValueError, match="Item vector dimension"):
        index.add(VectorItem("bad", [1.0, 2.0]))
    with pytest.raises(ValueError, match="Query vector dimension"):
        index.search([1.0, 2.0])


def test_hnsw_index_crud_and_search():
    cfg = HNSWConfig(m=8, ef_construction=32, ef_search=16, max_layers=4)
    index = HNSWIndex(dimension=4, metric=DistanceMetric.EUCLIDEAN, config=cfg, seed=42)

    items = [
        VectorItem(f"v_{i}", generate_random_vector(4, seed=i, normalized=True), {"idx": i, "even": i % 2 == 0})
        for i in range(50)
    ]
    index.add_batch(items)
    assert index.size() == 50

    # Search
    query = items[0].vector
    results = index.search(query, k=5, include_vector=True)
    assert len(results) == 5
    # The first result should be the query vector itself (distance ~0)
    assert results[0].id == "v_0"
    assert math.isclose(results[0].distance, 0.0, abs_tol=1e-4)

    # Filtered search
    even_results = index.search(query, k=5, filter_expr={"even": False})
    for r in even_results:
        assert r.metadata["even"] is False

    # Stats
    stats = index.get_stats()
    assert stats["count"] == 50
    assert stats["index_type"] == IndexType.HNSW.value
    assert stats["layers_count"] >= 1

    # Deletion and graph repair
    root_id = stats["enter_point"]
    assert index.delete(root_id) is True
    assert index.size() == 49
    # New search still works after enter point deletion
    res_after = index.search(query, k=3)
    assert len(res_after) == 3

    index.clear()
    assert index.size() == 0


def test_ivf_index_crud_and_search():
    cfg = IVFConfig(nlist=4, nprobe=2, max_iterations=10, seed=42)
    index = IVFIndex(dimension=4, metric=DistanceMetric.COSINE, config=cfg)

    items = [
        VectorItem(f"item_{i}", generate_random_vector(4, seed=100 + i, normalized=True), {"group": i % 3})
        for i in range(30)
    ]
    index.add_batch(items)
    assert index.size() == 30

    query = items[5].vector
    res = index.search(query, k=5)
    assert len(res) == 5
    assert res[0].id == "item_5"

    # Filtered search
    filtered = index.search(query, k=5, filter_expr={"group": 2})
    for r in filtered:
        assert r.metadata["group"] == 2

    # Stats
    stats = index.get_stats()
    assert stats["is_trained"] is True
    assert stats["clusters_count"] == 4

    # Delete
    assert index.delete("item_5") is True
    assert index.size() == 29
    assert index.get("item_5") is None

    index.clear()
    assert index.size() == 0


def test_hnsw_vs_flat_recall():
    dim = 8
    num_items = 100
    flat_idx = FlatIndex(dimension=dim, metric=DistanceMetric.COSINE)
    hnsw_idx = HNSWIndex(
        dimension=dim,
        metric=DistanceMetric.COSINE,
        config=HNSWConfig(m=16, ef_construction=64, ef_search=32),
        seed=42,
    )

    items = [
        VectorItem(f"doc_{i}", generate_random_vector(dim, seed=200 + i, normalized=True))
        for i in range(num_items)
    ]
    flat_idx.add_batch(items)
    hnsw_idx.add_batch(items)

    query = generate_random_vector(dim, seed=999, normalized=True)
    flat_res = flat_idx.search(query, k=10)
    hnsw_res = hnsw_idx.search(query, k=10)

    flat_ids = {r.id for r in flat_res}
    hnsw_ids = {r.id for r in hnsw_res}

    # Recall@10 should be high (>= 80%)
    overlap = len(flat_ids & hnsw_ids)
    recall = overlap / 10.0
    assert recall >= 0.8, f"HNSW Recall@10 was {recall * 100}%"
