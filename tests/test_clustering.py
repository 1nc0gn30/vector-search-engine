"""Unit and integration tests for Vector Clustering & Silhouette Cohesion Engine."""

from __future__ import annotations

import json
import random
import urllib.request
from typing import List

import pytest

from vector_search_engine.clustering import (
    ClusterAnalysisResult,
    ClusterInfo,
    HierarchicalKMeansClusterer,
    KMeansClusterer,
    SilhouetteAnalyzer,
    kmeans_plus_plus_init,
)
from vector_search_engine.collection import VectorCollection
from vector_search_engine.mcp_server import MCPServer
from vector_search_engine.models import (
    CollectionConfig,
    DistanceMetric,
    VectorItem,
)
from vector_search_engine.ui_server import (
    _COLLECTIONS_REGISTRY,
    _REGISTRY_LOCK,
    start_ui_server,
)
from vector_search_engine.cli import main as cli_main


def _generate_clustered_items() -> List[VectorItem]:
    """Generate 3 synthetic well-separated 4D vector clusters."""
    items: List[VectorItem] = []
    # Cluster A around [1.0, 0.0, 0.0, 0.0]
    for i in range(10):
        v = [1.0 + random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0.0, 0.0]
        items.append(
            VectorItem(
                id=f"c1_{i}",
                vector=v,
                document=f"AI Neural Document {i}",
                metadata={"category": "ai", "domain": "tech"},
            )
        )

    # Cluster B around [0.0, 1.0, 0.0, 0.0]
    for i in range(10):
        v = [0.0, 1.0 + random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0.0]
        items.append(
            VectorItem(
                id=f"c2_{i}",
                vector=v,
                document=f"Database SQL Document {i}",
                metadata={"category": "database", "domain": "backend"},
            )
        )

    # Cluster C around [0.0, 0.0, 1.0, 0.0]
    for i in range(10):
        v = [0.0, 0.0, 1.0 + random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]
        items.append(
            VectorItem(
                id=f"c3_{i}",
                vector=v,
                document=f"Physics Quantum Document {i}",
                metadata={"category": "physics", "domain": "science"},
            )
        )

    return items


class TestKMeansClusterer:
    """Test suite for K-Means++ clustering."""

    def test_kmeans_plus_plus_seeding(self) -> None:
        vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
        rng = random.Random(42)
        centroids = kmeans_plus_plus_init(vectors, k=2, metric=DistanceMetric.COSINE, rng=rng)
        assert len(centroids) == 2
        assert len(centroids[0]) == 2
        assert len(centroids[1]) == 2

    def test_kmeans_fit_separated_clusters(self) -> None:
        items = _generate_clustered_items()
        clusterer = KMeansClusterer(k=3, max_iter=30, metric=DistanceMetric.COSINE, seed=42)
        result = clusterer.fit(items)

        assert isinstance(result, ClusterAnalysisResult)
        assert result.k == 3
        assert len(result.clusters) == 3
        assert result.converged is True
        assert result.silhouette_score > 0.4  # Distinct clusters should have high silhouette
        assert result.inertia >= 0.0

        # Verify cluster info fields
        for c in result.clusters:
            assert isinstance(c, ClusterInfo)
            assert c.size > 0
            assert c.medoid_id != ""
            assert len(c.centroid) == 4
            assert c.dispersion >= 0.0
            assert c.diameter >= 0.0
            assert "category" in c.top_metadata or "domain" in c.top_metadata
            d = c.to_dict()
            assert "cluster_id" in d
            assert "dispersion" in d

        # Check serialization
        res_dict = result.to_dict()
        assert res_dict["k"] == 3
        assert len(res_dict["clusters"]) == 3
        assert len(res_dict["item_cluster_map"]) == 30

    def test_kmeans_empty_items(self) -> None:
        clusterer = KMeansClusterer(k=3, metric=DistanceMetric.EUCLIDEAN)
        res = clusterer.fit([])
        assert res.k == 3
        assert len(res.clusters) == 0
        assert res.silhouette_score == 0.0


class TestHierarchicalClusterer:
    """Test suite for Hierarchical Vector Tree Partitioning."""

    def test_hierarchical_clustering_fit(self) -> None:
        items = _generate_clustered_items()
        clusterer = HierarchicalKMeansClusterer(
            branch_factor=2,
            max_depth=3,
            min_leaf_size=5,
            metric=DistanceMetric.COSINE,
            seed=42,
        )
        result = clusterer.fit(items)

        assert result.method == "hierarchical"
        assert result.k >= 2
        assert len(result.clusters) == result.k
        assert result.silhouette_score > 0.3
        for c in result.clusters:
            assert c.size >= 1
            assert c.medoid_id != ""

    def test_hierarchical_empty(self) -> None:
        clusterer = HierarchicalKMeansClusterer(branch_factor=2)
        res = clusterer.fit([])
        assert res.k == 0
        assert len(res.clusters) == 0


class TestSilhouetteAnalyzer:
    """Test suite for Silhouette Cohesion metric."""

    def test_silhouette_computation_edge_cases(self) -> None:
        # 1 item: score is 0.0
        item = VectorItem(id="a", vector=[1.0, 0.0])
        score = SilhouetteAnalyzer.compute([item], {"a": 0})
        assert score == 0.0

        # All items in single cluster: score is 0.0
        item2 = VectorItem(id="b", vector=[0.0, 1.0])
        score2 = SilhouetteAnalyzer.compute([item, item2], {"a": 0, "b": 0})
        assert score2 == 0.0


class TestVectorCollectionClustering:
    """Test collection-level clustering interface."""

    def test_collection_cluster_methods(self) -> None:
        config = CollectionConfig(name="test_clustering_col", dimension=4, metric=DistanceMetric.COSINE)
        col = VectorCollection(config=config)
        items = _generate_clustered_items()
        col.upsert_batch(items)

        # K-Means clustering
        res_km = col.cluster(k=3, method="kmeans", seed=42)
        assert res_km.k == 3
        assert res_km.silhouette_score > 0.4

        # Hierarchical clustering
        res_hier = col.cluster(k=2, method="hierarchical", seed=42)
        assert res_hier.k >= 2

        # Filtered clustering: only AI category
        res_filtered = col.cluster(
            k=2,
            method="kmeans",
            filter_expr={"category": "ai"},
            seed=42,
        )
        assert len(res_filtered.item_cluster_map) == 10
        for item_id in res_filtered.item_cluster_map.keys():
            assert item_id.startswith("c1_")


class TestMCPClusteringTool:
    """Test MCP tool registration and invocation for clustering."""

    def test_mcp_vector_cluster_analysis(self, tmp_path) -> None:
        server = MCPServer(data_dir=tmp_path)
        server.call_tool(
            "vector_create_collection",
            {"name": "mcp_clustered_col", "dimension": 4, "metric": "cosine"},
        )
        server.call_tool(
            "vector_upsert",
            {
                "collection": "mcp_clustered_col",
                "documents": [it.to_dict() for it in _generate_clustered_items()],
            },
        )

        # Check tool appears in list
        tools = server.get_tools_schema()
        tool_names = [t["name"] for t in tools]
        assert "vector_cluster_analysis" in tool_names

        # Call tool
        res = server.call_tool(
            "vector_cluster_analysis",
            {"collection": "mcp_clustered_col", "k": 3, "method": "kmeans", "seed": 42},
        )
        assert res["collection"] == "mcp_clustered_col"
        assert res["k"] == 3
        assert len(res["clusters"]) == 3
        assert res["silhouette_score"] > 0.4


class TestUIAndCLIClustering:
    """Test UI endpoints and CLI command for clustering."""

    def test_ui_server_clustering_endpoints(self) -> None:
        col = VectorCollection(CollectionConfig("ui_cluster_test", dimension=4, metric=DistanceMetric.COSINE))
        col.upsert_batch(_generate_clustered_items())
        with _REGISTRY_LOCK:
            _COLLECTIONS_REGISTRY["ui_cluster_test"] = col

        server = start_ui_server(host="127.0.0.1", port=0, background=True)
        host, port = server.server_address
        base_url = f"http://{host}:{port}"

        try:
            # Test GET /api/collections/<name>/clusters
            req_get = urllib.request.Request(f"{base_url}/api/collections/ui_cluster_test/clusters?k=3&method=kmeans")
            with urllib.request.urlopen(req_get, timeout=5) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert data["k"] == 3
                assert len(data["clusters"]) == 3

            # Test POST /api/collections/<name>/clusters
            post_body = json.dumps({"k": 3, "method": "kmeans", "seed": 42}).encode("utf-8")
            req_post = urllib.request.Request(
                f"{base_url}/api/collections/ui_cluster_test/clusters",
                data=post_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req_post, timeout=5) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert data["k"] == 3
                assert len(data["clusters"]) == 3
                assert data["silhouette_score"] > 0.4
        finally:
            server.shutdown()
            server.server_close()

    def test_cli_cluster_command(self, tmp_path) -> None:
        col_file = tmp_path / "cli_test_col.json"
        col = VectorCollection(CollectionConfig("cli_test_col", dimension=4, metric=DistanceMetric.COSINE))
        col.upsert_batch(_generate_clustered_items())
        col.save_to_disk(col_file)

        # Run CLI cluster subcommand with json output
        ret = cli_main(["cluster", "cli_test_col", "--data-dir", str(tmp_path), "--json", "-k", "3"])
        assert ret == 0
