"""VectorCollection: main storage, indexing, search, and persistence engine.

Orchestrates dense vector indexes (Flat, HNSW, IVF), sparse BM25 indexing,
hybrid search, metadata filtering, and cross-platform atomic disk serialization.
"""

from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from vector_search_engine.compat import (
    atomic_write_bytes,
    atomic_write_text,
    safe_ensure_dir,
    safe_json_read,
    safe_json_write,
    safe_path_normalization,
)
from vector_search_engine.indexes import (
    BaseIndex,
    FlatIndex,
    HNSWIndex,
    IVFIndex,
    evaluate_filter,
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
from vector_search_engine.hybrid_engine import (
    BM25Index,
    HybridSearchEngine,
)

VDB_BINARY_MAGIC = b"VDB\x01"
VDB_BINARY_VERSION = 1


class VectorCollection:
    """High-performance vector collection with dense, sparse, and hybrid search."""

    def __init__(
        self,
        config: CollectionConfig,
        persist_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.config = config
        self.persist_path = (
            safe_path_normalization(persist_path) if persist_path else None
        )

        # Primary storage
        self._items: Dict[str, VectorItem] = {}

        # Dense vector index
        self.index: BaseIndex = self._create_index(
            index_type=self.config.index_type,
            dimension=self.config.dimension,
            metric=self.config.metric,
            hnsw_config=self.config.hnsw_config,
            ivf_config=self.config.ivf_config,
        )

        # Sparse keyword index for documents
        self._sparse_index: BM25Index = BM25Index()

    def _create_index(
        self,
        index_type: IndexType,
        dimension: int,
        metric: DistanceMetric,
        hnsw_config: Optional[HNSWConfig] = None,
        ivf_config: Optional[IVFConfig] = None,
    ) -> BaseIndex:
        """Create an index instance according to IndexType."""
        if index_type == IndexType.FLAT:
            return FlatIndex(dimension=dimension, metric=metric)
        elif index_type == IndexType.HNSW:
            return HNSWIndex(
                dimension=dimension,
                metric=metric,
                config=hnsw_config or self.config.hnsw_config,
            )
        elif index_type == IndexType.IVF:
            return IVFIndex(
                dimension=dimension,
                metric=metric,
                config=ivf_config or self.config.ivf_config,
            )
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    def __len__(self) -> int:
        """Return total count of items."""
        return len(self._items)

    @property
    def count(self) -> int:
        """Return total count of items."""
        return len(self._items)

    def _normalize_item(
        self, item: Union[VectorItem, Dict[str, Any]]
    ) -> VectorItem:
        """Normalize dict or VectorItem to VectorItem."""
        if isinstance(item, VectorItem):
            return item
        elif isinstance(item, dict):
            return VectorItem.from_dict(item)
        else:
            raise TypeError(f"Expected VectorItem or dict, got {type(item)}")

    def insert(self, item: Union[VectorItem, Dict[str, Any]]) -> str:
        """Insert a single item into the collection. Raises ValueError if ID exists."""
        v_item = self._normalize_item(item)
        if v_item.id in self._items:
            raise ValueError(f"Item with id '{v_item.id}' already exists in collection.")

        if len(v_item.vector) != self.config.dimension:
            raise ValueError(
                f"Item dimension {len(v_item.vector)} does not match collection dimension {self.config.dimension}"
            )

        self._items[v_item.id] = v_item
        self.index.add(v_item)
        if v_item.document:
            self._sparse_index.add_document(
                v_item.id, v_item.document, v_item.metadata
            )
        return v_item.id

    def insert_batch(
        self, items: Sequence[Union[VectorItem, Dict[str, Any]]]
    ) -> List[str]:
        """Insert multiple items into the collection."""
        inserted_ids: List[str] = []
        for item in items:
            inserted_ids.append(self.insert(item))
        return inserted_ids

    def upsert(self, item: Union[VectorItem, Dict[str, Any]]) -> str:
        """Insert or update an item in the collection."""
        v_item = self._normalize_item(item)
        if len(v_item.vector) != self.config.dimension:
            raise ValueError(
                f"Item dimension {len(v_item.vector)} does not match collection dimension {self.config.dimension}"
            )

        if v_item.id in self._items:
            self.delete(v_item.id)

        self._items[v_item.id] = v_item
        self.index.add(v_item)
        if v_item.document:
            self._sparse_index.add_document(
                v_item.id, v_item.document, v_item.metadata
            )
        return v_item.id

    def upsert_batch(
        self, items: Sequence[Union[VectorItem, Dict[str, Any]]]
    ) -> List[str]:
        """Upsert multiple items in the collection."""
        upserted_ids: List[str] = []
        for item in items:
            upserted_ids.append(self.upsert(item))
        return upserted_ids

    def get(self, item_id: str) -> Optional[VectorItem]:
        """Get an item by ID."""
        return self._items.get(str(item_id))

    def get_batch(self, ids: Sequence[str]) -> List[Optional[VectorItem]]:
        """Get multiple items by IDs."""
        return [self.get(i) for i in ids]

    def delete(self, item_id: str) -> bool:
        """Delete an item by ID."""
        s_id = str(item_id)
        if s_id not in self._items:
            return False

        del self._items[s_id]
        self.index.delete(s_id)
        self._sparse_index.delete_document(s_id)
        return True

    def delete_batch(self, ids: Sequence[str]) -> int:
        """Delete multiple items by IDs. Returns count of deleted items."""
        deleted_count = 0
        for i in ids:
            if self.delete(i):
                deleted_count += 1
        return deleted_count

    def clear(self) -> None:
        """Clear all items and indexes in the collection."""
        self._items.clear()
        self.index.clear()
        self._sparse_index.clear()

    def query(
        self,
        query_vector: Sequence[float],
        k: int = 10,
        filter_expr: Optional[Dict[str, Any]] = None,
        include_vector: bool = False,
        ef_search: Optional[int] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """Execute a dense vector similarity query."""
        if len(query_vector) != self.config.dimension:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} does not match collection dimension {self.config.dimension}"
            )

        return self.index.search(
            query_vector=query_vector,
            k=k,
            filter_expr=filter_expr,
            include_vector=include_vector,
            ef_search=ef_search,
            **kwargs,
        )

    def text_search(
        self,
        query_text: str,
        k: int = 10,
        filter_expr: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Execute a sparse BM25 keyword search query."""
        return self._sparse_index.search(
            query=query_text,
            k=k,
            filter_expr=filter_expr,
        )

    def hybrid_search(
        self,
        query_vector: Sequence[float],
        query_text: str,
        k: int = 10,
        alpha: float = 0.5,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        filter_expr: Optional[Dict[str, Any]] = None,
        include_vector: bool = False,
        **kwargs: Any,
    ) -> List[HybridSearchResult]:
        """Execute hybrid search blending dense vector and sparse BM25 results."""
        candidate_k = max(k * 3, 20)
        dense_results = self.query(
            query_vector=query_vector,
            k=candidate_k,
            filter_expr=filter_expr,
            include_vector=include_vector,
            **kwargs,
        )

        sparse_results = self.text_search(
            query_text=query_text,
            k=candidate_k,
            filter_expr=filter_expr,
        )

        fusion = fusion_method.lower().strip()
        if fusion == "rrf":
            return HybridSearchEngine.fuse_rrf(
                dense_results=dense_results,
                sparse_results=sparse_results,
                k=rrf_k,
                top_k=k,
            )
        elif fusion == "linear":
            return HybridSearchEngine.fuse_linear(
                dense_results=dense_results,
                sparse_results=sparse_results,
                alpha=alpha,
                top_k=k,
            )
        else:
            raise ValueError(
                f"Unknown fusion method '{fusion_method}'. Supported: ['rrf', 'linear']"
            )

    def filter(
        self,
        filter_expr: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> List[VectorItem]:
        """Retrieve items matching metadata filter expression."""
        matched: List[VectorItem] = []
        for item in self._items.values():
            if evaluate_filter(item.metadata, filter_expr):
                matched.append(item)
                if limit is not None and len(matched) >= limit:
                    break
        return matched

    def rebuild_index(
        self,
        index_type: Optional[IndexType] = None,
        hnsw_config: Optional[HNSWConfig] = None,
        ivf_config: Optional[IVFConfig] = None,
    ) -> None:
        """Rebuild dense index, optionally switching index type or updating index configs."""
        if index_type is not None:
            self.config.index_type = index_type
        if hnsw_config is not None:
            self.config.hnsw_config = hnsw_config
        if ivf_config is not None:
            self.config.ivf_config = ivf_config

        new_index = self._create_index(
            index_type=self.config.index_type,
            dimension=self.config.dimension,
            metric=self.config.metric,
            hnsw_config=self.config.hnsw_config,
            ivf_config=self.config.ivf_config,
        )

        for item in self._items.values():
            new_index.add(item)

        self.index = new_index

    def stats(self) -> CollectionStats:
        """Compute comprehensive statistics of the collection and index."""
        index_stats = self.index.get_stats()
        approx_bytes = index_stats.get("memory_bytes", 0)

        # Add sparse index memory estimation
        sparse_docs = self._sparse_index.size()
        has_sparse = sparse_docs > 0

        layers_count = index_stats.get("layers_count", 1)
        avg_links = float(index_stats.get("avg_links_per_node", 0.0))

        return CollectionStats(
            name=self.config.name,
            count=len(self._items),
            dimension=self.config.dimension,
            metric=self.config.metric.value,
            index_type=self.config.index_type.value,
            memory_bytes=approx_bytes,
            layers_count=layers_count,
            avg_links_per_node=avg_links,
            has_sparse_index=has_sparse,
            index_stats=index_stats,
        )

    def save_to_disk(
        self,
        path: Optional[Union[str, Path]] = None,
        format: str = "json",
    ) -> Path:
        """Serialize and atomically save collection to disk in JSON or binary format."""
        target_path = safe_path_normalization(path or self.persist_path)
        if not target_path:
            raise ValueError("No target path provided to save collection.")

        fmt = format.lower().strip()
        if fmt == "json":
            data = {
                "config": self.config.to_dict(),
                "items": [item.to_dict() for item in self._items.values()],
                "saved_at": time.time(),
            }
            safe_json_write(target_path, data, indent=2)
        elif fmt in ("binary", "bin", "vdb"):
            raw_bytes = self._serialize_binary()
            atomic_write_bytes(target_path, raw_bytes)
        else:
            raise ValueError(f"Unsupported storage format '{format}'. Supported: ['json', 'binary']")

        return target_path

    def _serialize_binary(self) -> bytes:
        """Serialize collection into compact binary format."""
        buf = bytearray()
        # Magic + version + flags
        buf.extend(VDB_BINARY_MAGIC)
        buf.extend(struct.pack("<II", VDB_BINARY_VERSION, 0))

        # Config JSON block
        config_json = json.dumps(self.config.to_dict()).encode("utf-8")
        buf.extend(struct.pack("<I", len(config_json)))
        buf.extend(config_json)

        # Items count
        buf.extend(struct.pack("<I", len(self._items)))

        # Pack each item
        dim = self.config.dimension
        vec_format = f"<{dim}f"

        for item in self._items.values():
            id_bytes = item.id.encode("utf-8")
            buf.extend(struct.pack("<H", len(id_bytes)))
            buf.extend(id_bytes)

            doc_bytes = item.document.encode("utf-8") if item.document else b""
            buf.extend(struct.pack("<I", len(doc_bytes)))
            if doc_bytes:
                buf.extend(doc_bytes)

            meta_json = json.dumps(item.metadata, default=str).encode("utf-8")
            buf.extend(struct.pack("<I", len(meta_json)))
            buf.extend(meta_json)

            buf.extend(struct.pack("<d", float(item.created_at)))

            # Pack vector as float32
            buf.extend(struct.pack(vec_format, *item.vector))

        return bytes(buf)

    @classmethod
    def _deserialize_binary(
        cls,
        raw_bytes: bytes,
        persist_path: Optional[Union[str, Path]] = None,
    ) -> VectorCollection:
        """Deserialize collection from binary bytes."""
        if not raw_bytes.startswith(VDB_BINARY_MAGIC):
            raise ValueError("Invalid binary format: missing VDB magic header.")

        offset = len(VDB_BINARY_MAGIC)
        version, flags = struct.unpack_from("<II", raw_bytes, offset)
        offset += 8

        # Config JSON block
        config_len = struct.unpack_from("<I", raw_bytes, offset)[0]
        offset += 4
        config_json = raw_bytes[offset : offset + config_len].decode("utf-8")
        offset += config_len

        config_data = json.loads(config_json)
        config = CollectionConfig.from_dict(config_data)

        collection = cls(config=config, persist_path=persist_path)

        # Items count
        items_count = struct.unpack_from("<I", raw_bytes, offset)[0]
        offset += 4

        dim = config.dimension
        vec_format = f"<{dim}f"
        vec_bytes_len = struct.calcsize(vec_format)

        items_to_add: List[VectorItem] = []
        for _ in range(items_count):
            id_len = struct.unpack_from("<H", raw_bytes, offset)[0]
            offset += 2
            item_id = raw_bytes[offset : offset + id_len].decode("utf-8")
            offset += id_len

            doc_len = struct.unpack_from("<I", raw_bytes, offset)[0]
            offset += 4
            document = raw_bytes[offset : offset + doc_len].decode("utf-8") if doc_len > 0 else None
            offset += doc_len

            meta_len = struct.unpack_from("<I", raw_bytes, offset)[0]
            offset += 4
            meta_json = raw_bytes[offset : offset + meta_len].decode("utf-8")
            offset += meta_len
            metadata = json.loads(meta_json) if meta_json else {}

            created_at = struct.unpack_from("<d", raw_bytes, offset)[0]
            offset += 8

            vector = list(struct.unpack_from(vec_format, raw_bytes, offset))
            offset += vec_bytes_len

            items_to_add.append(
                VectorItem(
                    id=item_id,
                    vector=vector,
                    metadata=metadata,
                    document=document,
                    created_at=created_at,
                )
            )

        collection.upsert_batch(items_to_add)
        return collection

    @classmethod
    def load_from_disk(
        cls,
        path: Union[str, Path],
    ) -> VectorCollection:
        """Load and instantiate a VectorCollection from disk (auto-detects JSON or binary)."""
        target_path = safe_path_normalization(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Collection file not found: {target_path}")

        # Check binary magic
        with open(target_path, "rb") as f:
            header = f.read(len(VDB_BINARY_MAGIC))

        if header == VDB_BINARY_MAGIC:
            with open(target_path, "rb") as f:
                raw_bytes = f.read()
            return cls._deserialize_binary(raw_bytes, persist_path=target_path)
        else:
            data = safe_json_read(target_path)
            config = CollectionConfig.from_dict(data["config"])
            collection = cls(config=config, persist_path=target_path)
            items = [VectorItem.from_dict(item_data) for item_data in data.get("items", [])]
            collection.upsert_batch(items)
            return collection
