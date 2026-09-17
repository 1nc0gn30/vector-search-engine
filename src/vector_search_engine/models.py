"""Core domain data models for vector-search-engine.

Defines enums, dataclasses, and configuration structures for items,
search results, indexes, and collection statistics.
"""

from __future__ import annotations

import time
import math
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Union


class DistanceMetric(str, Enum):
    """Supported distance metrics for vector similarity."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"
    HAMMING = "hamming"
    JACCARD = "jaccard"

    @classmethod
    def from_str(cls, value: Union[str, DistanceMetric]) -> DistanceMetric:
        """Parse string or enum into DistanceMetric."""
        if isinstance(value, DistanceMetric):
            return value
        val = str(value).lower().strip().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == val:
                return member
        raise ValueError(
            f"Unknown distance metric '{value}'. Supported: {[m.value for m in cls]}"
        )


class IndexType(str, Enum):
    """Supported index types."""

    FLAT = "flat"
    HNSW = "hnsw"
    IVF = "ivf"

    @classmethod
    def from_str(cls, value: Union[str, IndexType]) -> IndexType:
        """Parse string or enum into IndexType."""
        if isinstance(value, IndexType):
            return value
        val = str(value).lower().strip()
        for member in cls:
            if member.value == val:
                return member
        raise ValueError(
            f"Unknown index type '{value}'. Supported: {[m.value for m in cls]}"
        )


@dataclass
class VectorItem:
    """A vector item stored in the collection with id, vector, metadata and document."""

    id: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    document: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            self.id = str(self.id)
        if not isinstance(self.vector, list):
            self.vector = [float(v) for v in self.vector]
        else:
            self.vector = [float(v) for v in self.vector]
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert VectorItem to dictionary."""
        return {
            "id": self.id,
            "vector": self.vector,
            "metadata": self.metadata,
            "document": self.document,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VectorItem:
        """Construct VectorItem from dictionary."""
        return cls(
            id=str(data["id"]),
            vector=[float(v) for v in data["vector"]],
            metadata=dict(data.get("metadata") or {}),
            document=data.get("document"),
            created_at=float(data.get("created_at", time.time())),
        )


@dataclass
class SearchResult:
    """Result of a vector similarity or text search query."""

    id: str
    score: float
    distance: float
    vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    document: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert SearchResult to dictionary."""
        res: Dict[str, Any] = {
            "id": self.id,
            "score": self.score,
            "distance": self.distance,
            "metadata": self.metadata,
            "document": self.document,
        }
        if self.vector is not None:
            res["vector"] = self.vector
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SearchResult:
        """Construct SearchResult from dictionary."""
        return cls(
            id=str(data["id"]),
            score=float(data["score"]),
            distance=float(data["distance"]),
            vector=[float(v) for v in data["vector"]] if "vector" in data and data["vector"] is not None else None,
            metadata=dict(data.get("metadata") or {}),
            document=data.get("document"),
        )


@dataclass
class HybridSearchResult:
    """Result of a hybrid dense + sparse keyword search query."""

    id: str
    combined_score: float
    dense_score: float
    sparse_score: float
    dense_rank: int
    sparse_rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    document: Optional[str] = None
    vector: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert HybridSearchResult to dictionary."""
        res: Dict[str, Any] = {
            "id": self.id,
            "combined_score": self.combined_score,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "metadata": self.metadata,
            "document": self.document,
        }
        if self.vector is not None:
            res["vector"] = self.vector
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HybridSearchResult:
        """Construct HybridSearchResult from dictionary."""
        return cls(
            id=str(data["id"]),
            combined_score=float(data["combined_score"]),
            dense_score=float(data["dense_score"]),
            sparse_score=float(data["sparse_score"]),
            dense_rank=int(data["dense_rank"]),
            sparse_rank=int(data["sparse_rank"]),
            metadata=dict(data.get("metadata") or {}),
            document=data.get("document"),
            vector=[float(v) for v in data["vector"]] if "vector" in data and data["vector"] is not None else None,
        )


@dataclass
class HNSWConfig:
    """Configuration for Hierarchical Navigable Small World (HNSW) graph index."""

    m: int = 16
    ef_construction: int = 64
    ef_search: int = 32
    max_layers: int = 6
    heuristic_pruning: bool = True
    m0: Optional[int] = None
    level_mult: Optional[float] = None

    def __post_init__(self) -> None:
        if self.m0 is None:
            self.m0 = self.m * 2
        if self.level_mult is None:
            self.level_mult = 1.0 / math.log(max(self.m, 2))

    def to_dict(self) -> Dict[str, Any]:
        """Convert HNSWConfig to dictionary."""
        return {
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "max_layers": self.max_layers,
            "heuristic_pruning": self.heuristic_pruning,
            "m0": self.m0,
            "level_mult": self.level_mult,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HNSWConfig:
        """Construct HNSWConfig from dictionary."""
        return cls(
            m=int(data.get("m", 16)),
            ef_construction=int(data.get("ef_construction", 64)),
            ef_search=int(data.get("ef_search", 32)),
            max_layers=int(data.get("max_layers", 6)),
            heuristic_pruning=bool(data.get("heuristic_pruning", True)),
            m0=int(data["m0"]) if "m0" in data and data["m0"] is not None else None,
            level_mult=float(data["level_mult"]) if "level_mult" in data and data["level_mult"] is not None else None,
        )


@dataclass
class IVFConfig:
    """Configuration for Inverted File (IVF) index."""

    nlist: int = 16
    nprobe: int = 4
    max_iterations: int = 20
    seed: Optional[int] = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert IVFConfig to dictionary."""
        return {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "max_iterations": self.max_iterations,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IVFConfig:
        """Construct IVFConfig from dictionary."""
        return cls(
            nlist=int(data.get("nlist", 16)),
            nprobe=int(data.get("nprobe", 4)),
            max_iterations=int(data.get("max_iterations", 20)),
            seed=int(data["seed"]) if "seed" in data and data["seed"] is not None else None,
        )


@dataclass
class CollectionConfig:
    """Configuration for a VectorCollection."""

    name: str
    dimension: int
    metric: DistanceMetric = DistanceMetric.COSINE
    index_type: IndexType = IndexType.HNSW
    hnsw_config: HNSWConfig = field(default_factory=HNSWConfig)
    ivf_config: Optional[IVFConfig] = None

    def __post_init__(self) -> None:
        if isinstance(self.metric, str):
            self.metric = DistanceMetric.from_str(self.metric)
        if isinstance(self.index_type, str):
            self.index_type = IndexType.from_str(self.index_type)
        if isinstance(self.hnsw_config, dict):
            self.hnsw_config = HNSWConfig.from_dict(self.hnsw_config)
        if isinstance(self.ivf_config, dict):
            self.ivf_config = IVFConfig.from_dict(self.ivf_config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert CollectionConfig to dictionary."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "metric": self.metric.value,
            "index_type": self.index_type.value,
            "hnsw_config": self.hnsw_config.to_dict() if self.hnsw_config else None,
            "ivf_config": self.ivf_config.to_dict() if self.ivf_config else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CollectionConfig:
        """Construct CollectionConfig from dictionary."""
        metric = DistanceMetric.from_str(data.get("metric", DistanceMetric.COSINE))
        index_type = IndexType.from_str(data.get("index_type", IndexType.HNSW))
        hnsw_config = (
            HNSWConfig.from_dict(data["hnsw_config"])
            if data.get("hnsw_config")
            else HNSWConfig()
        )
        ivf_config = (
            IVFConfig.from_dict(data["ivf_config"])
            if data.get("ivf_config")
            else None
        )
        return cls(
            name=str(data["name"]),
            dimension=int(data["dimension"]),
            metric=metric,
            index_type=index_type,
            hnsw_config=hnsw_config,
            ivf_config=ivf_config,
        )


@dataclass
class CollectionStats:
    """Comprehensive runtime and storage statistics for a VectorCollection."""

    name: str
    count: int
    dimension: int
    metric: str
    index_type: str
    memory_bytes: int
    layers_count: int = 1
    avg_links_per_node: float = 0.0
    has_sparse_index: bool = False
    index_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert CollectionStats to dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "dimension": self.dimension,
            "metric": self.metric,
            "index_type": self.index_type,
            "memory_bytes": self.memory_bytes,
            "layers_count": self.layers_count,
            "avg_links_per_node": self.avg_links_per_node,
            "has_sparse_index": self.has_sparse_index,
            "index_stats": self.index_stats,
        }
