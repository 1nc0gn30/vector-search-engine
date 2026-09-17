"""Tests for domain data models and configuration classes."""

from __future__ import annotations

import math
import pytest

from vector_search_engine.models import (
    DistanceMetric,
    IndexType,
    VectorItem,
    SearchResult,
    HybridSearchResult,
    HNSWConfig,
    IVFConfig,
    CollectionConfig,
    CollectionStats,
)


def test_distance_metric_parsing():
    assert DistanceMetric.from_str("cosine") == DistanceMetric.COSINE
    assert DistanceMetric.from_str("COSINE") == DistanceMetric.COSINE
    assert DistanceMetric.from_str("euclidean") == DistanceMetric.EUCLIDEAN
    assert DistanceMetric.from_str("dot_product") == DistanceMetric.DOT_PRODUCT
    assert DistanceMetric.from_str("dot-product") == DistanceMetric.DOT_PRODUCT
    assert DistanceMetric.from_str("manhattan") == DistanceMetric.MANHATTAN
    assert DistanceMetric.from_str("hamming") == DistanceMetric.HAMMING
    assert DistanceMetric.from_str("jaccard") == DistanceMetric.JACCARD

    with pytest.raises(ValueError, match="Unknown distance metric"):
        DistanceMetric.from_str("invalid_metric")


def test_index_type_parsing():
    assert IndexType.from_str("flat") == IndexType.FLAT
    assert IndexType.from_str("hnsw") == IndexType.HNSW
    assert IndexType.from_str("ivf") == IndexType.IVF

    with pytest.raises(ValueError, match="Unknown index type"):
        IndexType.from_str("unknown_type")


def test_vector_item_serialization():
    item = VectorItem(
        id="vec1",
        vector=[1.0, 2.0, 3.5],
        metadata={"category": "ai", "count": 10},
        document="A sample document text",
    )

    assert item.id == "vec1"
    assert item.vector == [1.0, 2.0, 3.5]
    assert item.metadata["category"] == "ai"
    assert item.document == "A sample document text"
    assert item.created_at > 0

    d = item.to_dict()
    assert d["id"] == "vec1"
    assert d["vector"] == [1.0, 2.0, 3.5]

    restored = VectorItem.from_dict(d)
    assert restored.id == item.id
    assert restored.vector == item.vector
    assert restored.metadata == item.metadata
    assert restored.document == item.document
    assert restored.created_at == item.created_at


def test_search_result_serialization():
    res = SearchResult(
        id="res1",
        score=0.985,
        distance=0.015,
        vector=[0.1, 0.2],
        metadata={"tag": "fast"},
        document="Result document",
    )

    d = res.to_dict()
    assert d["id"] == "res1"
    assert d["score"] == 0.985
    assert d["distance"] == 0.015
    assert d["vector"] == [0.1, 0.2]

    restored = SearchResult.from_dict(d)
    assert restored.id == res.id
    assert restored.score == res.score
    assert restored.distance == res.distance
    assert restored.vector == res.vector
    assert restored.metadata == res.metadata


def test_hybrid_search_result_serialization():
    hybrid_res = HybridSearchResult(
        id="doc42",
        combined_score=0.88,
        dense_score=0.92,
        sparse_score=0.80,
        dense_rank=1,
        sparse_rank=3,
        metadata={"source": "wikipedia"},
        document="Wikipedia article content",
        vector=[0.5, 0.5],
    )

    d = hybrid_res.to_dict()
    assert d["id"] == "doc42"
    assert d["combined_score"] == 0.88
    assert d["dense_score"] == 0.92
    assert d["sparse_score"] == 0.80
    assert d["dense_rank"] == 1
    assert d["sparse_rank"] == 3

    restored = HybridSearchResult.from_dict(d)
    assert restored.id == hybrid_res.id
    assert restored.combined_score == hybrid_res.combined_score
    assert restored.dense_score == hybrid_res.dense_score
    assert restored.sparse_score == hybrid_res.sparse_score
    assert restored.dense_rank == hybrid_res.dense_rank
    assert restored.sparse_rank == hybrid_res.sparse_rank


def test_hnsw_and_ivf_configs():
    hnsw = HNSWConfig(m=32, ef_construction=128, ef_search=64)
    assert hnsw.m == 32
    assert hnsw.m0 == 64
    assert math.isclose(hnsw.level_mult, 1.0 / math.log(32))

    hnsw_dict = hnsw.to_dict()
    restored_hnsw = HNSWConfig.from_dict(hnsw_dict)
    assert restored_hnsw.m == 32
    assert restored_hnsw.ef_construction == 128
    assert restored_hnsw.ef_search == 64

    ivf = IVFConfig(nlist=32, nprobe=8, max_iterations=50)
    assert ivf.nlist == 32
    assert ivf.nprobe == 8
    ivf_dict = ivf.to_dict()
    restored_ivf = IVFConfig.from_dict(ivf_dict)
    assert restored_ivf.nlist == 32
    assert restored_ivf.nprobe == 8


def test_collection_config_and_stats():
    cfg = CollectionConfig(
        name="test_col",
        dimension=64,
        metric="euclidean",
        index_type="hnsw",
    )
    assert cfg.metric == DistanceMetric.EUCLIDEAN
    assert cfg.index_type == IndexType.HNSW
    assert isinstance(cfg.hnsw_config, HNSWConfig)

    cfg_dict = cfg.to_dict()
    restored_cfg = CollectionConfig.from_dict(cfg_dict)
    assert restored_cfg.name == "test_col"
    assert restored_cfg.dimension == 64
    assert restored_cfg.metric == DistanceMetric.EUCLIDEAN

    stats = CollectionStats(
        name="test_col",
        count=1000,
        dimension=64,
        metric="euclidean",
        index_type="hnsw",
        memory_bytes=102400,
        layers_count=3,
        avg_links_per_node=14.5,
        has_sparse_index=True,
    )
    s_dict = stats.to_dict()
    assert s_dict["count"] == 1000
    assert s_dict["layers_count"] == 3
    assert s_dict["avg_links_per_node"] == 14.5
