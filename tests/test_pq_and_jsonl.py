"""Tests for Product Quantization (PQ) and JSONL import/export in vector_search_engine."""

import json
import tempfile
from pathlib import Path
import pytest

from vector_search_engine.models import (
    CollectionConfig,
    DistanceMetric,
    IndexType,
    VectorItem,
)
from vector_search_engine.collection import VectorCollection
from vector_search_engine.metrics import (
    ProductQuantizer,
    generate_random_vector,
)


class TestProductQuantizer:
    """Test suite for ProductQuantizer compression and ADC search."""

    def test_init_validation(self):
        # Dimension must be divisible by num_subspaces
        with pytest.raises(ValueError, match="must be divisible"):
            ProductQuantizer(dimension=10, num_subspaces=3)

        with pytest.raises(ValueError, match="num_subspaces must be positive"):
            ProductQuantizer(dimension=10, num_subspaces=0)

        with pytest.raises(ValueError, match="num_centroids must be between"):
            ProductQuantizer(dimension=8, num_subspaces=2, num_centroids=300)

    def test_train_encode_decode(self):
        dim = 16
        num_subspaces = 4
        num_centroids = 16
        pq = ProductQuantizer(dimension=dim, num_subspaces=num_subspaces, num_centroids=num_centroids)
        assert not pq.is_trained()

        # Generate synthetic training vectors
        train_vecs = [generate_random_vector(dim, seed=i) for i in range(40)]
        pq.train(train_vecs, max_iters=5, seed=123)
        assert pq.is_trained()

        test_vec = generate_random_vector(dim, seed=999)
        code = pq.encode(test_vec)
        assert isinstance(code, bytes)
        assert len(code) == num_subspaces

        # Check decode reconstruction
        recon = pq.decode(code)
        assert len(recon) == dim
        assert isinstance(recon[0], float)

    def test_asymmetric_distance(self):
        dim = 8
        num_subspaces = 2
        num_centroids = 8
        pq = ProductQuantizer(dimension=dim, num_subspaces=num_subspaces, num_centroids=num_centroids)

        train_vecs = [generate_random_vector(dim, seed=i) for i in range(20)]
        pq.train(train_vecs, max_iters=5, seed=42)

        query = generate_random_vector(dim, seed=777)
        target = generate_random_vector(dim, seed=888)
        code = pq.encode(target)

        # Precomputed distance table vs direct
        table = pq.compute_distance_table(query, metric=DistanceMetric.EUCLIDEAN)
        assert len(table) == num_subspaces
        assert len(table[0]) == num_centroids

        dist1 = pq.asymmetric_distance_with_table(table, code)
        dist2 = pq.asymmetric_distance(query, code, metric=DistanceMetric.EUCLIDEAN)
        assert pytest.approx(dist1) == dist2
        assert dist1 >= 0.0

    def test_serialization(self):
        dim = 8
        pq = ProductQuantizer(dimension=dim, num_subspaces=2, num_centroids=4)
        train_vecs = [generate_random_vector(dim, seed=i) for i in range(10)]
        pq.train(train_vecs, max_iters=2, seed=1)

        d = pq.to_dict()
        pq2 = ProductQuantizer.from_dict(d)
        assert pq2.is_trained()
        assert pq2.dimension == pq.dimension
        assert pq2.num_subspaces == pq.num_subspaces

        vec = generate_random_vector(dim, seed=5)
        assert pq.encode(vec) == pq2.encode(vec)


class TestCollectionJSONL:
    """Test suite for VectorCollection JSONL export and import streaming."""

    def test_export_and_import_jsonl(self):
        config = CollectionConfig(
            name="test_jsonl",
            dimension=4,
            metric=DistanceMetric.COSINE,
            index_type=IndexType.FLAT,
        )
        coll = VectorCollection(config=config)

        items = [
            VectorItem(id="doc-1", vector=[1.0, 0.0, 0.0, 0.0], document="Hello world", metadata={"tag": "greet"}),
            VectorItem(id="doc-2", vector=[0.0, 1.0, 0.0, 0.0], document="Vector search", metadata={"tag": "tech"}),
            VectorItem(id="doc-3", vector=[0.0, 0.0, 1.0, 0.0], document="HNSW graphs", metadata={"tag": "algo"}),
        ]
        coll.upsert_batch(items)
        assert coll.count == 3

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "embeddings.jsonl"
            exported_count = coll.export_embeddings_jsonl(jsonl_path)
            assert exported_count == 3
            assert jsonl_path.exists()

            # Verify file structure
            lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 3
            record = json.loads(lines[0])
            assert "id" in record
            assert "vector" in record
            assert record["document"] == "Hello world"
            assert record["metadata"] == {"tag": "greet"}

            # Import into fresh collection
            coll2 = VectorCollection(config=config)
            imported_count = coll2.import_embeddings_jsonl(jsonl_path, batch_size=2)
            assert imported_count == 3
            assert coll2.count == 3

            retrieved = coll2.get("doc-2")
            assert retrieved is not None
            assert retrieved.document == "Vector search"
            assert retrieved.metadata["tag"] == "tech"
