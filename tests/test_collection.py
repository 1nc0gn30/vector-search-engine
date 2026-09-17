"""Tests for VectorCollection storage, indexing, persistence, and querying."""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from vector_search_engine.models import (
    CollectionConfig,
    DistanceMetric,
    HNSWConfig,
    IndexType,
    VectorItem,
)
from vector_search_engine.collection import VectorCollection


def test_collection_crud_and_query(sample_documents):
    config = CollectionConfig(
        name="docs_col",
        dimension=4,
        metric=DistanceMetric.COSINE,
        index_type=IndexType.HNSW,
        hnsw_config=HNSWConfig(m=8, ef_construction=32, ef_search=16),
    )
    col = VectorCollection(config=config)
    assert len(col) == 0
    assert col.count == 0

    # Insert items
    for doc in sample_documents:
        col.insert(
            VectorItem(
                id=doc["id"],
                vector=doc["vector"],
                metadata=doc["metadata"],
                document=doc["text"],
            )
        )

    assert len(col) == len(sample_documents)

    # Test duplicate insert raises error
    with pytest.raises(ValueError, match="already exists"):
        col.insert(VectorItem("doc1", [0.9, 0.1, 0.2, 0.05]))

    # Test dimension mismatch error
    with pytest.raises(ValueError, match="does not match collection dimension"):
        col.insert(VectorItem("bad_dim", [0.1, 0.2]))

    # Get & get_batch
    item1 = col.get("doc1")
    assert item1 is not None
    assert item1.document == sample_documents[0]["text"]

    batch_items = col.get_batch(["doc1", "doc2", "non_existent"])
    assert len(batch_items) == 3
    assert batch_items[0].id == "doc1"
    assert batch_items[1].id == "doc2"
    assert batch_items[2] is None

    # Dense Query
    query_vec = [0.9, 0.1, 0.2, 0.05]
    results = col.query(query_vec, k=3, include_vector=True)
    assert len(results) == 3
    assert results[0].id == "doc1"
    assert results[0].vector == query_vec

    # Filtered Query
    db_results = col.query(query_vec, k=5, filter_expr={"category": "database"})
    assert len(db_results) >= 1
    for r in db_results:
        assert r.metadata["category"] == "database"


def test_collection_sparse_and_hybrid_search(sample_documents):
    config = CollectionConfig(
        name="hybrid_col",
        dimension=4,
        metric=DistanceMetric.COSINE,
        index_type=IndexType.FLAT,
    )
    col = VectorCollection(config=config)
    col.upsert_batch([
        VectorItem(
            id=d["id"],
            vector=d["vector"],
            metadata=d["metadata"],
            document=d["text"],
        )
        for d in sample_documents
    ])

    # Text search
    text_res = col.text_search("quantum computing", k=2)
    assert len(text_res) >= 1
    assert text_res[0].id == "doc4"

    # Hybrid search RRF
    hybrid_rrf = col.hybrid_search(
        query_vector=[0.05, 0.1, 0.9, 0.85],
        query_text="quantum computing",
        k=3,
        fusion_method="rrf",
    )
    assert len(hybrid_rrf) >= 1
    assert hybrid_rrf[0].id == "doc4"

    # Hybrid search Linear
    hybrid_linear = col.hybrid_search(
        query_vector=[0.05, 0.1, 0.9, 0.85],
        query_text="quantum computing",
        k=3,
        alpha=0.6,
        fusion_method="linear",
    )
    assert len(hybrid_linear) >= 1
    assert hybrid_linear[0].id == "doc4"


def test_collection_filter_and_rebuild(sample_documents):
    config = CollectionConfig(
        name="rebuild_col",
        dimension=4,
        metric=DistanceMetric.EUCLIDEAN,
        index_type=IndexType.FLAT,
    )
    col = VectorCollection(config=config)
    col.insert_batch([
        VectorItem(id=d["id"], vector=d["vector"], metadata=d["metadata"], document=d["text"])
        for d in sample_documents
    ])

    filtered = col.filter({"year": {"$gte": 2023}})
    assert len(filtered) == 3
    assert all(item.metadata["year"] >= 2023 for item in filtered)

    # Rebuild index from Flat to HNSW
    col.rebuild_index(index_type=IndexType.HNSW)
    assert col.config.index_type == IndexType.HNSW
    stats = col.stats()
    assert stats.index_type == IndexType.HNSW.value
    assert stats.count == len(sample_documents)


def test_collection_json_persistence(sample_documents, temp_dir: Path):
    json_path = temp_dir / "collection.json"
    config = CollectionConfig(name="persist_json", dimension=4, metric=DistanceMetric.COSINE)
    col = VectorCollection(config=config, persist_path=json_path)

    for d in sample_documents:
        col.insert(VectorItem(id=d["id"], vector=d["vector"], metadata=d["metadata"], document=d["text"]))

    col.save_to_disk(json_path, format="json")
    assert json_path.exists()

    # Reload from disk
    restored = VectorCollection.load_from_disk(json_path)
    assert restored.config.name == "persist_json"
    assert restored.count == len(sample_documents)
    assert restored.get("doc1") is not None
    assert restored.get("doc1").document == sample_documents[0]["text"]

    # Verify query on restored collection
    res = restored.query([0.9, 0.1, 0.2, 0.05], k=1)
    assert res[0].id == "doc1"


def test_collection_binary_persistence(sample_documents, temp_dir: Path):
    bin_path = temp_dir / "collection.vdb"
    config = CollectionConfig(name="persist_bin", dimension=4, metric=DistanceMetric.COSINE)
    col = VectorCollection(config=config, persist_path=bin_path)

    for d in sample_documents:
        col.insert(VectorItem(id=d["id"], vector=d["vector"], metadata=d["metadata"], document=d["text"]))

    col.save_to_disk(bin_path, format="binary")
    assert bin_path.exists()

    # Reload from binary disk
    restored = VectorCollection.load_from_disk(bin_path)
    assert restored.config.name == "persist_bin"
    assert restored.count == len(sample_documents)
    assert restored.get("doc3").metadata["category"] == "database"

    # Query
    res = restored.query([0.1, 0.8, 0.1, 0.7], k=1)
    assert res[0].id == "doc3"
