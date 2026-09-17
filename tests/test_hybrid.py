"""Tests for sparse BM25 indexing and dense+sparse hybrid search fusion."""

from __future__ import annotations

import math
import pytest

from vector_search_engine.models import SearchResult, HybridSearchResult
from vector_search_engine.hybrid_engine import (
    simple_tokenize,
    BM25Index,
    HybridSearchEngine,
)


def test_simple_tokenize():
    text = "Hello, Vector Search Engine 2026! It's fast & accurate."
    tokens = simple_tokenize(text)
    assert tokens == ["hello", "vector", "search", "engine", "2026", "it", "s", "fast", "accurate"]
    assert simple_tokenize("") == []


def test_bm25_index_crud_and_search():
    bm25 = BM25Index(k1=1.5, b=0.75)
    assert bm25.size() == 0

    docs = [
        ("d1", "The quick brown fox jumps over the lazy dog", {"category": "animals"}),
        ("d2", "Vector search with HNSW and dense embeddings", {"category": "tech"}),
        ("d3", "Hybrid search blends BM25 text retrieval with dense vectors", {"category": "tech"}),
        ("d4", "Brown dogs and brown cats are friendly animals", {"category": "animals"}),
    ]

    for doc_id, text, meta in docs:
        bm25.add_document(doc_id, text, meta)

    assert bm25.size() == 4

    # Search for "brown"
    results = bm25.search("brown", k=3)
    assert len(results) >= 2
    matched_ids = {r.id for r in results}
    assert "d1" in matched_ids
    assert "d4" in matched_ids
    assert all(r.score > 0 for r in results)

    # Search for multi-term query "vector search"
    tech_results = bm25.search("vector search", k=2)
    assert len(tech_results) == 2
    assert {r.id for r in tech_results} == {"d2", "d3"}

    # Search with filter
    filtered_results = bm25.search("brown", k=5, filter_expr={"category": "animals"})
    for r in filtered_results:
        assert r.metadata["category"] == "animals"

    # Delete
    assert bm25.delete_document("d1") is True
    assert bm25.size() == 3
    assert bm25.delete_document("d1") is False

    # Search after deletion
    res_after = bm25.search("fox", k=5)
    assert len(res_after) == 0

    bm25.clear()
    assert bm25.size() == 0


def test_hybrid_search_rrf_fusion():
    dense_results = [
        SearchResult(id="doc_a", score=0.95, distance=0.05, document="Doc A text"),
        SearchResult(id="doc_b", score=0.85, distance=0.15, document="Doc B text"),
        SearchResult(id="doc_c", score=0.75, distance=0.25, document="Doc C text"),
    ]
    sparse_results = [
        SearchResult(id="doc_b", score=4.2, distance=0.19, document="Doc B text"),
        SearchResult(id="doc_d", score=3.8, distance=0.20, document="Doc D text"),
        SearchResult(id="doc_a", score=1.5, distance=0.40, document="Doc A text"),
    ]

    rrf_results = HybridSearchEngine.fuse_rrf(dense_results, sparse_results, k=60, top_k=4)
    assert len(rrf_results) == 4

    # doc_a rank: dense=1, sparse=3 -> 1/(60+1) + 1/(60+3) = 1/61 + 1/63 ≈ 0.01639 + 0.01587 = 0.03226
    # doc_b rank: dense=2, sparse=1 -> 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.01612 + 0.01639 = 0.03251
    # doc_b should rank higher than doc_a
    assert rrf_results[0].id == "doc_b"
    assert rrf_results[1].id == "doc_a"

    assert rrf_results[0].dense_rank == 2
    assert rrf_results[0].sparse_rank == 1


def test_hybrid_search_linear_fusion():
    dense_results = [
        SearchResult(id="doc_dense_top", score=1.0, distance=0.0),
        SearchResult(id="doc_dense_mid", score=0.5, distance=0.5),
        SearchResult(id="doc_dense_low", score=0.0, distance=1.0),
    ]
    sparse_results = [
        SearchResult(id="doc_sparse_top", score=10.0, distance=0.1),
        SearchResult(id="doc_dense_mid", score=5.0, distance=0.2),
        SearchResult(id="doc_sparse_low", score=0.0, distance=0.5),
    ]

    # Pure dense (alpha = 1.0)
    pure_dense = HybridSearchEngine.fuse_linear(dense_results, sparse_results, alpha=1.0, top_k=3)
    assert pure_dense[0].id == "doc_dense_top"

    # Pure sparse (alpha = 0.0)
    pure_sparse = HybridSearchEngine.fuse_linear(dense_results, sparse_results, alpha=0.0, top_k=3)
    assert pure_sparse[0].id == "doc_sparse_top"

    # Balanced (alpha = 0.5)
    balanced = HybridSearchEngine.fuse_linear(dense_results, sparse_results, alpha=0.5, top_k=5)
    assert len(balanced) == 5
    # doc_dense_mid has normalized dense=0.5 and normalized sparse=0.5 -> combined = 0.5
    mid = next(r for r in balanced if r.id == "doc_dense_mid")
    assert math.isclose(mid.combined_score, 0.5, abs_tol=1e-3)
