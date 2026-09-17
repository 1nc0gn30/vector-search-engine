"""Comprehensive test suite for Subagent 1 core vector engine modules."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vector_search_engine.compat import (
    PlatformInfo,
    atomic_write_bytes,
    atomic_write_text,
    get_platform_info,
    safe_delete_dir,
    safe_delete_file,
    safe_ensure_dir,
    safe_json_read,
    safe_json_write,
    safe_path_normalization,
)
from vector_search_engine.models import (
    CollectionConfig,
    CollectionStats,
    DistanceMetric,
    HNSWConfig,
    HybridSearchResult,
    IndexType,
    IVFConfig,
    SearchResult,
    VectorItem,
)
from vector_search_engine.metrics import (
    binary_quantize,
    compute_centroid,
    compute_distance,
    compute_similarity,
    cosine_distance,
    cosine_similarity,
    dequantize_binary,
    dequantize_scalar,
    distance_to_score,
    dot_product,
    dot_product_distance,
    euclidean_distance,
    euclidean_similarity,
    generate_random_vector,
    hamming_distance,
    hamming_distance_bytes,
    hamming_similarity,
    jaccard_distance,
    jaccard_similarity,
    manhattan_distance,
    manhattan_similarity,
    normalize_vector,
    random_projection,
    scalar_quantize,
)
from vector_search_engine.indexes import (
    FlatIndex,
    HNSWIndex,
    IVFIndex,
    evaluate_filter,
)
from vector_search_engine.hybrid_engine import (
    BM25Index,
    HybridSearchEngine,
)
from vector_search_engine.collection import (
    VectorCollection,
)


class TestCompat(unittest.TestCase):
    """Test cross-platform compatibility utilities."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_platform_info(self) -> None:
        info = get_platform_info()
        self.assertIsInstance(info, PlatformInfo)
        self.assertIn("os_name", info.to_dict())
        self.assertIsInstance(info.is_64bit, bool)

    def test_safe_path_normalization(self) -> None:
        p = safe_path_normalization(str(self.base_path / "sub" / "file.txt"))
        self.assertTrue(p.is_absolute())

    def test_atomic_write_and_read_text(self) -> None:
        file_path = self.base_path / "test.txt"
        atomic_write_text(file_path, "Hello Vector Search Engine!")
        self.assertTrue(file_path.exists())
        self.assertEqual(file_path.read_text(encoding="utf-8"), "Hello Vector Search Engine!")

    def test_atomic_write_and_read_bytes(self) -> None:
        file_path = self.base_path / "test.bin"
        data = b"\x00\x01\x02\x03\xff\xfe"
        atomic_write_bytes(file_path, data)
        self.assertTrue(file_path.exists())
        self.assertEqual(file_path.read_bytes(), data)

    def test_safe_json_read_write(self) -> None:
        file_path = self.base_path / "data.json"
        payload = {"name": "test_collection", "dimension": 128, "tags": ["a", "b"]}
        safe_json_write(file_path, payload)
        read_back = safe_json_read(file_path)
        self.assertEqual(read_back, payload)

    def test_safe_delete(self) -> None:
        f = self.base_path / "to_delete.txt"
        atomic_write_text(f, "temp")
        self.assertTrue(safe_delete_file(f))
        self.assertFalse(safe_delete_file(f))


class TestModels(unittest.TestCase):
    """Test domain data models and enums."""

    def test_enums(self) -> None:
        self.assertEqual(DistanceMetric.from_str("cosine"), DistanceMetric.COSINE)
        self.assertEqual(DistanceMetric.from_str("DOT-PRODUCT"), DistanceMetric.DOT_PRODUCT)
        self.assertEqual(IndexType.from_str("hnsw"), IndexType.HNSW)
        self.assertEqual(IndexType.from_str("FLAT"), IndexType.FLAT)

    def test_vector_item_serialization(self) -> None:
        item = VectorItem(
            id="doc_1",
            vector=[0.1, 0.2, 0.3],
            metadata={"category": "ai", "score": 98},
            document="Artificial Intelligence paper",
        )
        d = item.to_dict()
        reconstructed = VectorItem.from_dict(d)
        self.assertEqual(item.id, reconstructed.id)
        self.assertEqual(item.vector, reconstructed.vector)
        self.assertEqual(item.metadata, reconstructed.metadata)
        self.assertEqual(item.document, reconstructed.document)

    def test_config_serialization(self) -> None:
        config = CollectionConfig(
            name="articles",
            dimension=4,
            metric=DistanceMetric.COSINE,
            index_type=IndexType.HNSW,
            hnsw_config=HNSWConfig(m=32, ef_construction=128),
        )
        d = config.to_dict()
        reconstructed = CollectionConfig.from_dict(d)
        self.assertEqual(config.name, reconstructed.name)
        self.assertEqual(config.dimension, reconstructed.dimension)
        self.assertEqual(config.metric, reconstructed.metric)
        self.assertEqual(config.hnsw_config.m, reconstructed.hnsw_config.m)


class TestMetrics(unittest.TestCase):
    """Test pure Python vector arithmetic and distance metrics."""

    def test_cosine_similarity_and_distance(self) -> None:
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        v4 = [-1.0, 0.0, 0.0]

        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0)
        self.assertAlmostEqual(cosine_distance(v1, v2), 0.0)

        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0)
        self.assertAlmostEqual(cosine_distance(v1, v3), 1.0)

        self.assertAlmostEqual(cosine_similarity(v1, v4), -1.0)
        self.assertAlmostEqual(cosine_distance(v1, v4), 2.0)

    def test_euclidean_distance(self) -> None:
        v1 = [0.0, 0.0]
        v2 = [3.0, 4.0]
        self.assertAlmostEqual(euclidean_distance(v1, v2), 5.0)
        self.assertAlmostEqual(euclidean_similarity(v1, v2), 1.0 / 6.0)

    def test_manhattan_distance(self) -> None:
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 0.0, -1.0]
        # |1-4| + |2-0| + |3 - (-1)| = 3 + 2 + 4 = 9
        self.assertAlmostEqual(manhattan_distance(v1, v2), 9.0)

    def test_hamming_distance(self) -> None:
        v1 = [1, 0, 1, 1]
        v2 = [1, 1, 0, 1]
        self.assertAlmostEqual(hamming_distance(v1, v2), 2 / 4)

    def test_jaccard_similarity(self) -> None:
        s1 = {"apple", "banana", "cherry"}
        s2 = {"banana", "cherry", "date"}
        # Intersection: 2, Union: 4 => 0.5
        self.assertAlmostEqual(jaccard_similarity(s1, s2), 0.5)
        self.assertAlmostEqual(jaccard_distance(s1, s2), 0.5)

    def test_normalize_vector(self) -> None:
        v = [3.0, 4.0]
        norm = normalize_vector(v)
        self.assertAlmostEqual(norm[0], 0.6)
        self.assertAlmostEqual(norm[1], 0.8)
        self.assertAlmostEqual(cosine_similarity(norm, norm), 1.0)

    def test_centroid(self) -> None:
        vectors = [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
        ]
        c = compute_centroid(vectors)
        self.assertEqual(c, [2.0, 3.0, 4.0])

    def test_scalar_quantization_roundtrip(self) -> None:
        vec = [0.1, 0.5, 0.9, -0.4, 1.2]
        q_bytes, min_v, max_v = scalar_quantize(vec, num_bits=8)
        dequant = dequantize_scalar(q_bytes, min_v, max_v, num_bits=8)
        self.assertEqual(len(dequant), len(vec))
        for orig, dq in zip(vec, dequant):
            self.assertAlmostEqual(orig, dq, delta=0.02)

    def test_binary_quantization_roundtrip(self) -> None:
        vec = [1.5, -0.5, 0.2, -3.1, 0.0, 1.1, -2.0, 3.0, 0.8]
        b_bytes = binary_quantize(vec)
        dq = dequantize_binary(b_bytes, dimension=len(vec))
        self.assertEqual(len(dq), len(vec))
        for orig, sign in zip(vec, dq):
            if orig > 0:
                self.assertEqual(sign, 1.0)
            else:
                self.assertEqual(sign, -1.0)

    def test_random_projection(self) -> None:
        vecs = [
            generate_random_vector(128, seed=i)
            for i in range(10)
        ]
        proj = random_projection(vecs, target_dim=2, seed=42)
        self.assertEqual(len(proj), 10)
        self.assertEqual(len(proj[0]), 2)


class TestFilterEvaluation(unittest.TestCase):
    """Test metadata filter expression evaluation."""

    def test_filter_expressions(self) -> None:
        meta = {
            "category": "electronics",
            "price": 89.99,
            "rating": 4.5,
            "tags": ["audio", "wireless", "sale"],
            "in_stock": True,
            "title": "Wireless Bluetooth Headphones",
        }

        # Exact
        self.assertTrue(evaluate_filter(meta, {"category": "electronics"}))
        self.assertFalse(evaluate_filter(meta, {"category": "books"}))

        # Comparison operators
        self.assertTrue(evaluate_filter(meta, {"price": {"$lte": 100, "$gte": 50}}))
        self.assertFalse(evaluate_filter(meta, {"price": {"$gt": 100}}))
        self.assertTrue(evaluate_filter(meta, {"rating": {"$gt": 4.0}}))

        # Membership
        self.assertTrue(evaluate_filter(meta, {"category": {"$in": ["electronics", "clothing"]}}))
        self.assertFalse(evaluate_filter(meta, {"category": {"$nin": ["electronics"]}}))

        # Arrays
        self.assertTrue(evaluate_filter(meta, {"tags": {"$contains": "wireless"}}))
        self.assertTrue(evaluate_filter(meta, {"tags": {"$all": ["audio", "wireless"]}}))
        self.assertFalse(evaluate_filter(meta, {"tags": {"$all": ["audio", "gaming"]}}))

        # Regex & Exists
        self.assertTrue(evaluate_filter(meta, {"title": {"$regex": r"Bluetooth"}}))
        self.assertTrue(evaluate_filter(meta, {"in_stock": {"$exists": True}}))

        # Logical $and, $or
        self.assertTrue(evaluate_filter(meta, {
            "$and": [
                {"price": {"$lt": 100}},
                {"rating": {"$gte": 4.0}},
            ]
        }))
        self.assertTrue(evaluate_filter(meta, {
            "$or": [
                {"category": "books"},
                {"tags": {"$contains": "audio"}},
            ]
        }))


class TestIndexes(unittest.TestCase):
    """Test FlatIndex, HNSWIndex, and IVFIndex."""

    def setUp(self) -> None:
        self.dim = 8
        self.items = [
            VectorItem(
                id=f"item_{i}",
                vector=generate_random_vector(self.dim, seed=i, normalized=True),
                metadata={"group": i % 3, "val": i * 10},
                document=f"Document number {i} about search engine techniques",
            )
            for i in range(50)
        ]
        self.query_vec = generate_random_vector(self.dim, seed=999, normalized=True)

    def test_flat_index(self) -> None:
        flat = FlatIndex(dimension=self.dim, metric=DistanceMetric.COSINE)
        flat.add_batch(self.items)
        self.assertEqual(flat.size(), 50)

        results = flat.search(self.query_vec, k=5)
        self.assertEqual(len(results), 5)
        # Check ascending distance
        for j in range(len(results) - 1):
            self.assertLessEqual(results[j].distance, results[j + 1].distance)

        # Test filtering
        filtered = flat.search(self.query_vec, k=5, filter_expr={"group": 1})
        for r in filtered:
            self.assertEqual(r.metadata["group"], 1)

        # Test deletion
        self.assertTrue(flat.delete("item_0"))
        self.assertEqual(flat.size(), 49)

    def test_hnsw_index(self) -> None:
        flat = FlatIndex(dimension=self.dim, metric=DistanceMetric.COSINE)
        flat.add_batch(self.items)
        exact_top5 = flat.search(self.query_vec, k=5)

        hnsw = HNSWIndex(
            dimension=self.dim,
            metric=DistanceMetric.COSINE,
            config=HNSWConfig(m=16, ef_construction=64, ef_search=32),
            seed=42,
        )
        hnsw.add_batch(self.items)
        self.assertEqual(hnsw.size(), 50)

        hnsw_top5 = hnsw.search(self.query_vec, k=5)
        self.assertEqual(len(hnsw_top5), 5)

        # Verify high recall vs exact flat search
        exact_ids = {r.id for r in exact_top5}
        hnsw_ids = {r.id for r in hnsw_top5}
        intersection = exact_ids & hnsw_ids
        # Should have high overlap
        self.assertGreaterEqual(len(intersection), 4)

        # Test filtering
        filtered = hnsw.search(self.query_vec, k=5, filter_expr={"group": 0})
        for r in filtered:
            self.assertEqual(r.metadata["group"], 0)

        # Test deletion
        self.assertTrue(hnsw.delete("item_5"))
        self.assertEqual(hnsw.size(), 49)

    def test_ivf_index(self) -> None:
        ivf = IVFIndex(
            dimension=self.dim,
            metric=DistanceMetric.COSINE,
            config=IVFConfig(nlist=8, nprobe=4, seed=42),
        )
        ivf.add_batch(self.items)
        self.assertEqual(ivf.size(), 50)

        results = ivf.search(self.query_vec, k=5)
        self.assertEqual(len(results), 5)

        stats = ivf.get_stats()
        self.assertTrue(stats["is_trained"])
        self.assertGreater(stats["clusters_count"], 0)


class TestHybridEngine(unittest.TestCase):
    """Test BM25 and Hybrid Search Fusion."""

    def test_bm25_index(self) -> None:
        bm25 = BM25Index()
        bm25.add_document("doc1", "Fast vector database in pure Python")
        bm25.add_document("doc2", "Deep learning embeddings for natural language processing")
        bm25.add_document("doc3", "Hierarchical Navigable Small World graphs for ANN vector search")

        results = bm25.search("vector search", k=2)
        self.assertGreater(len(results), 0)
        self.assertIn(results[0].id, ("doc1", "doc3"))

    def test_rrf_and_linear_fusion(self) -> None:
        dense_results = [
            SearchResult(id="docA", score=0.95, distance=0.05),
            SearchResult(id="docB", score=0.85, distance=0.15),
            SearchResult(id="docC", score=0.70, distance=0.30),
        ]
        sparse_results = [
            SearchResult(id="docB", score=4.5, distance=0.18),
            SearchResult(id="docA", score=3.2, distance=0.23),
            SearchResult(id="docD", score=2.1, distance=0.32),
        ]

        rrf_res = HybridSearchEngine.fuse_rrf(dense_results, sparse_results, k=60, top_k=3)
        self.assertEqual(len(rrf_res), 3)
        self.assertIn(rrf_res[0].id, ("docA", "docB"))

        linear_res = HybridSearchEngine.fuse_linear(dense_results, sparse_results, alpha=0.5, top_k=3)
        self.assertEqual(len(linear_res), 3)


class TestVectorCollection(unittest.TestCase):
    """Test VectorCollection full integration and disk persistence."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.dim = 4

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_crud_and_query(self) -> None:
        config = CollectionConfig(
            name="test_collection",
            dimension=self.dim,
            metric=DistanceMetric.COSINE,
            index_type=IndexType.HNSW,
        )
        coll = VectorCollection(config=config)

        # Insert items
        coll.insert({
            "id": "item1",
            "vector": [1.0, 0.0, 0.0, 0.0],
            "metadata": {"type": "A", "val": 10},
            "document": "First document on search engines",
        })
        coll.insert({
            "id": "item2",
            "vector": [0.0, 1.0, 0.0, 0.0],
            "metadata": {"type": "B", "val": 20},
            "document": "Second document on database architecture",
        })

        self.assertEqual(coll.count, 2)
        self.assertIsNotNone(coll.get("item1"))

        # Query
        res = coll.query([0.9, 0.1, 0.0, 0.0], k=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].id, "item1")

        # Hybrid Search
        hybrid_res = coll.hybrid_search(
            query_vector=[0.0, 1.0, 0.0, 0.0],
            query_text="database architecture",
            k=1,
        )
        self.assertEqual(len(hybrid_res), 1)
        self.assertEqual(hybrid_res[0].id, "item2")

        # Upsert
        coll.upsert({
            "id": "item1",
            "vector": [0.0, 0.0, 1.0, 0.0],
            "metadata": {"type": "A_updated"},
        })
        updated = coll.get("item1")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.metadata["type"], "A_updated")

        # Delete
        self.assertTrue(coll.delete("item2"))
        self.assertEqual(coll.count, 1)

    def test_json_persistence_roundtrip(self) -> None:
        config = CollectionConfig(
            name="json_store",
            dimension=self.dim,
            metric=DistanceMetric.EUCLIDEAN,
            index_type=IndexType.FLAT,
        )
        save_path = self.base_path / "coll.json"
        coll = VectorCollection(config=config, persist_path=save_path)

        for i in range(10):
            coll.insert(
                VectorItem(
                    id=f"v_{i}",
                    vector=[float(i), float(i * 2), float(i * 3), float(i * 4)],
                    metadata={"index": i},
                    document=f"Document text {i}",
                )
            )

        coll.save_to_disk()
        self.assertTrue(save_path.exists())

        loaded = VectorCollection.load_from_disk(save_path)
        self.assertEqual(loaded.count, 10)
        self.assertEqual(loaded.config.name, "json_store")
        self.assertEqual(loaded.config.metric, DistanceMetric.EUCLIDEAN)

        res = loaded.query([0.0, 0.0, 0.0, 0.0], k=1)
        self.assertEqual(res[0].id, "v_0")

    def test_binary_vdb_persistence_roundtrip(self) -> None:
        config = CollectionConfig(
            name="bin_store",
            dimension=self.dim,
            metric=DistanceMetric.COSINE,
            index_type=IndexType.HNSW,
            hnsw_config=HNSWConfig(m=16, ef_construction=32),
        )
        save_path = self.base_path / "coll.vdb"
        coll = VectorCollection(config=config)

        for i in range(25):
            coll.insert(
                VectorItem(
                    id=f"bin_item_{i}",
                    vector=generate_random_vector(self.dim, seed=i, normalized=True),
                    metadata={"tag": f"group_{i % 4}", "val": i * 5},
                    document=f"Binary vector storage document {i} with metadata",
                )
            )

        coll.save_to_disk(save_path, format="binary")
        self.assertTrue(save_path.exists())

        loaded = VectorCollection.load_from_disk(save_path)
        self.assertEqual(loaded.count, 25)
        self.assertEqual(loaded.config.name, "bin_store")
        self.assertEqual(loaded.config.dimension, self.dim)

        item = loaded.get("bin_item_10")
        self.assertIsNotNone(item)
        self.assertEqual(item.metadata["tag"], "group_2")
        self.assertEqual(item.metadata["val"], 50)

        # Query loaded binary collection
        q = generate_random_vector(self.dim, seed=10, normalized=True)
        res = loaded.query(q, k=1)
        self.assertEqual(res[0].id, "bin_item_10")


if __name__ == "__main__":
    unittest.main()
