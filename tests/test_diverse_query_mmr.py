"""Tests for Maximal Marginal Relevance (MMR) diverse search in vector_search_engine."""

from __future__ import annotations

import pytest
from typing import List

from vector_search_engine.models import (
    CollectionConfig,
    DistanceMetric,
    IndexType,
    SearchResult,
    VectorItem,
)
from vector_search_engine.hybrid_engine import HybridSearchEngine
from vector_search_engine.collection import VectorCollection
from vector_search_engine.mcp_server import MCPServer


class TestMaximalMarginalRelevance:
    """Test MMR algorithm and diversity re-ranking."""

    def test_mmr_pure_relevance_vs_diversity(self) -> None:
        """Verify lambda=1.0 matches similarity order while lambda=0.0 promotes diversity."""
        # Query along X axis
        query = [1.0, 0.0, 0.0]

        # Candidates:
        # doc1: [1.0, 0.0, 0.0] (exact match, sim = 1.0)
        # doc2: [0.99, 0.01, 0.0] (almost identical duplicate to doc1, sim ~= 0.999)
        # doc3: [0.7, 0.7, 0.0] (diverse diagonal, sim ~= 0.7)
        c1 = VectorItem(id="doc1", vector=[1.0, 0.0, 0.0], document="target one")
        c2 = VectorItem(id="doc2", vector=[0.99, 0.01, 0.0], document="target duplicate")
        c3 = VectorItem(id="doc3", vector=[0.7071, 0.7071, 0.0], document="diagonal diverse")

        candidates = [c1, c2, c3]

        # With pure relevance (lambda = 1.0), order should be doc1, doc2, doc3
        rel_results = HybridSearchEngine.maximal_marginal_relevance(
            query_vector=query,
            candidates=candidates,
            k=2,
            lambda_mult=1.0,
            metric=DistanceMetric.COSINE,
        )
        assert len(rel_results) == 2
        assert rel_results[0].id == "doc1"
        assert rel_results[1].id == "doc2"

        # With balanced or diversity-heavy lambda (e.g. 0.3), after picking doc1,
        # doc2 has very high similarity (~1.0) to doc1, so penalty is high.
        # doc3 has lower similarity (~0.7) to doc1, so doc3 should be chosen over doc2!
        div_results = HybridSearchEngine.maximal_marginal_relevance(
            query_vector=query,
            candidates=candidates,
            k=2,
            lambda_mult=0.3,
            metric=DistanceMetric.COSINE,
        )
        assert len(div_results) == 2
        assert div_results[0].id == "doc1"
        assert div_results[1].id == "doc3"  # Promoted due to diversity!

    def test_mmr_edge_cases(self) -> None:
        """Test MMR with empty candidates, zero k, and missing vectors."""
        query = [1.0, 0.0]
        assert HybridSearchEngine.maximal_marginal_relevance(query, [], k=5) == []
        assert HybridSearchEngine.maximal_marginal_relevance(query, [VectorItem(id="1", vector=[1.0, 0.0])], k=0) == []

        # SearchResult missing vector should be skipped
        sr_no_vec = SearchResult(id="no_vec", score=0.9, distance=0.1, vector=None)
        sr_with_vec = SearchResult(id="with_vec", score=0.8, distance=0.2, vector=[1.0, 0.0])
        res = HybridSearchEngine.maximal_marginal_relevance(query, [sr_no_vec, sr_with_vec], k=2)
        assert len(res) == 1
        assert res[0].id == "with_vec"


class TestCollectionDiverseQuery:
    """Test VectorCollection.diverse_query method."""

    @pytest.fixture
    def sample_collection(self) -> VectorCollection:
        config = CollectionConfig(
            name="diverse_test",
            dimension=3,
            metric=DistanceMetric.COSINE,
            index_type=IndexType.FLAT,
        )
        col = VectorCollection(config)
        # Cluster A: near [1, 0, 0]
        col.insert(VectorItem(id="a1", vector=[1.0, 0.0, 0.0], metadata={"category": "tech"}))
        col.insert(VectorItem(id="a2", vector=[0.98, 0.05, 0.0], metadata={"category": "tech"}))
        col.insert(VectorItem(id="a3", vector=[0.97, 0.08, 0.0], metadata={"category": "tech"}))
        # Cluster B: near [0.7, 0.7, 0]
        col.insert(VectorItem(id="b1", vector=[0.7071, 0.7071, 0.0], metadata={"category": "finance"}))
        col.insert(VectorItem(id="b2", vector=[0.70, 0.72, 0.0], metadata={"category": "finance"}))
        return col

    def test_diverse_query_selection(self, sample_collection: VectorCollection) -> None:
        query = [1.0, 0.0, 0.0]

        # Standard query returns top 3 in cluster A
        standard = sample_collection.query(query, k=3)
        assert [r.id for r in standard] == ["a1", "a2", "a3"]

        # Diverse query with lambda=0.4 brings in cluster B item (b1 or b2)
        diverse = sample_collection.diverse_query(query, k=3, fetch_k=5, lambda_mult=0.4)
        assert len(diverse) == 3
        assert diverse[0].id == "a1"
        # Second item should be from cluster B for diversity
        assert diverse[1].id in ("b1", "b2")

    def test_diverse_query_dimension_mismatch(self, sample_collection: VectorCollection) -> None:
        with pytest.raises(ValueError, match="dimension"):
            sample_collection.diverse_query([1.0, 2.0], k=2)

    def test_diverse_query_include_vector(self, sample_collection: VectorCollection) -> None:
        results_with_vec = sample_collection.diverse_query([1.0, 0.0, 0.0], k=2, include_vector=True)
        assert results_with_vec[0].vector is not None

        results_no_vec = sample_collection.diverse_query([1.0, 0.0, 0.0], k=2, include_vector=False)
        assert results_no_vec[0].vector is None

    def test_diverse_query_with_filter(self, sample_collection: VectorCollection) -> None:
        # Filter only finance category
        results = sample_collection.diverse_query(
            [1.0, 0.0, 0.0],
            k=2,
            filter_expr={"category": "finance"},
        )
        assert len(results) == 2
        assert all(r.metadata["category"] == "finance" for r in results)


class TestMCPServerDiverseQuery:
    """Test MCP server vector_query tool with diversity_lambda."""

    def test_mcp_vector_query_with_diversity(self) -> None:
        server = MCPServer()
        # Create collection
        server._tool_create_collection({
            "name": "mcp_diverse_col",
            "dimension": 2,
            "metric": "cosine",
        })
        # Insert items
        col = server.get_collection("mcp_diverse_col")
        col.insert(VectorItem(id="v1", vector=[1.0, 0.0]))
        col.insert(VectorItem(id="v2", vector=[0.99, 0.02]))
        col.insert(VectorItem(id="v3", vector=[0.0, 1.0]))

        # Query with diversity_lambda
        res = server._tool_query({
            "collection": "mcp_diverse_col",
            "vector": [1.0, 0.0],
            "top_k": 2,
            "diversity_lambda": 0.2,
        })
        assert res["query_type"] == "diverse_vector_mmr"
        assert len(res["results"]) == 2
        assert res["results"][0]["id"] == "v1"
        assert res["results"][1]["id"] == "v3"  # Diverse orthogonal vector chosen over near duplicate v2!
