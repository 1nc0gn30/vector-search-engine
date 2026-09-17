"""Vector Search Engine - Pure Python, Zero-Dependency Vector Database.

A high-performance, in-memory and persistent vector search engine featuring
Hierarchical Navigable Small World (HNSW) graph indexing, Flat brute-force baseline,
Inverted File (IVF) clustering, hybrid search (dense vectors + sparse BM25 with
Reciprocal Rank Fusion and linear alpha blending), complex metadata filtering,
scalar/binary quantization, and Model Context Protocol (MCP) server support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

__version__ = "0.1.0"
__author__ = "Vector Search Engine Architecture Team"

# Models and domain structures
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

# Metrics and vector arithmetic
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
    hamming_distance,
    hamming_similarity,
    jaccard_distance,
    jaccard_similarity,
    manhattan_distance,
    manhattan_similarity,
    normalize_vector,
    random_projection,
    scalar_quantize,
)

# Indexes and filtering
from vector_search_engine.indexes import (
    BaseIndex,
    FlatIndex,
    HNSWIndex,
    IVFIndex,
    evaluate_filter,
)

# Hybrid search engine
from vector_search_engine.hybrid_engine import (
    BM25Index,
    HybridSearchEngine,
)

# Alias HybridEngine to HybridSearchEngine for public API consistency
HybridEngine = HybridSearchEngine

# VectorCollection and persistence
from vector_search_engine.collection import (
    VectorCollection,
)


def export_collection(
    collection: VectorCollection,
    filepath: Union[str, Path],
    format: str = "json",
) -> Path:
    """Export a VectorCollection instance to disk in JSON or binary format."""
    return collection.save_to_disk(path=filepath, format=format)


def load_collection(
    filepath: Union[str, Path],
) -> VectorCollection:
    """Load and instantiate a VectorCollection from a serialized file on disk."""
    return VectorCollection.load_from_disk(path=filepath)


__all__ = [
    "__version__",
    "VectorCollection",
    "HNSWIndex",
    "FlatIndex",
    "IVFIndex",
    "HybridEngine",
    "HybridSearchEngine",
    "BM25Index",
    "VectorItem",
    "SearchResult",
    "HybridSearchResult",
    "DistanceMetric",
    "IndexType",
    "CollectionConfig",
    "CollectionStats",
    "HNSWConfig",
    "IVFConfig",
    "BaseIndex",
    "evaluate_filter",
    "cosine_similarity",
    "cosine_distance",
    "euclidean_distance",
    "euclidean_similarity",
    "dot_product",
    "dot_product_distance",
    "manhattan_distance",
    "manhattan_similarity",
    "hamming_distance",
    "hamming_similarity",
    "jaccard_similarity",
    "jaccard_distance",
    "compute_distance",
    "compute_similarity",
    "distance_to_score",
    "normalize_vector",
    "compute_centroid",
    "scalar_quantize",
    "dequantize_scalar",
    "binary_quantize",
    "dequantize_binary",
    "random_projection",
    "export_collection",
    "load_collection",
]
