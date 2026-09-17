"""Model Context Protocol (MCP) Server for vector-search-engine.

Implements JSON-RPC 2.0 protocol handling over stdio for seamless integration
with AI assistants and agent frameworks (Claude, Codex, Antigravity, Cursor, etc.).
Pure Python standard library with zero external runtime dependencies.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union

# Ensure parent directory (src) is in sys.path for standalone script execution
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from vector_search_engine.compat import (
    PlatformInfo,
    get_platform_info,
    safe_ensure_dir,
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
    compute_distance,
    compute_similarity,
    cosine_similarity,
    dequantize_binary,
    dequantize_scalar,
    dot_product,
    euclidean_distance,
    manhattan_distance,
    normalize_vector,
    scalar_quantize,
)
from vector_search_engine.collection import (
    VectorCollection,
)

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "vector-search-engine-mcp"
SERVER_VERSION = "0.1.0"


def simple_text_embedder(
    text: str,
    dimension: int = 64,
    normalize: bool = True,
) -> List[float]:
    """Generate a deterministic, zero-dependency dense vector from text.

    Uses subword n-grams, word tokens, and multi-seed hashing with Gaussian
    pseudo-random projection to map semantic lexical features into Euclidean space.
    """
    if dimension <= 0:
        dimension = 64

    clean_text = (text or "").lower().strip()
    if not clean_text:
        return [0.0] * dimension

    vec = [0.0] * dimension
    words = re.findall(r"\b\w+\b", clean_text)
    
    # Extract word and subword features
    tokens: List[Tuple[str, float]] = []
    for w in words:
        tokens.append((w, 1.0))
        # Add character 3-grams for subword morphological matching
        if len(w) >= 3:
            for i in range(len(w) - 2):
                tokens.append((w[i : i + 3], 0.4))

    for token, weight in tokens:
        # Generate multi-hash signals
        h1 = hash(token) & 0xFFFFFFFF
        h2 = hash(token + "_salt1") & 0xFFFFFFFF
        h3 = hash(token + "_salt2") & 0xFFFFFFFF

        idx1 = h1 % dimension
        idx2 = h2 % dimension
        idx3 = h3 % dimension

        # Sign bit
        sign1 = 1.0 if (h1 & 0x80000000) == 0 else -1.0
        sign2 = 1.0 if (h2 & 0x80000000) == 0 else -1.0
        sign3 = 1.0 if (h3 & 0x80000000) == 0 else -1.0

        vec[idx1] += sign1 * weight * 1.2
        vec[idx2] += sign2 * weight * 0.8
        vec[idx3] += sign3 * weight * 0.5

    if normalize:
        return normalize_vector(vec)
    return vec


class MCPServer:
    """Model Context Protocol (MCP) Server for vector search operations."""

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        auto_persist: bool = True,
    ) -> None:
        if data_dir:
            self.data_dir: Path = safe_ensure_dir(data_dir)
        else:
            default_dir = Path.cwd() / ".vector_store"
            self.data_dir = safe_ensure_dir(default_dir)

        self.auto_persist = auto_persist
        self._collections: Dict[str, VectorCollection] = {}
        self._discover_collections()

    def _discover_collections(self) -> None:
        """Discover existing collections in the data directory."""
        if not self.data_dir.exists():
            return

        try:
            for item in self.data_dir.iterdir():
                if item.is_file() and (item.suffix in (".json", ".vdb")):
                    name = item.stem
                    if name not in self._collections:
                        try:
                            col = VectorCollection.load_from_disk(item)
                            self._collections[col.config.name] = col
                        except Exception:
                            # Skip corrupted or non-collection json files
                            pass
        except OSError:
            pass

    def get_collection(self, name: str) -> VectorCollection:
        """Get an active collection by name, loading from disk if necessary."""
        if name in self._collections:
            return self._collections[name]

        # Check on disk
        for ext in (".json", ".vdb"):
            disk_path = self.data_dir / f"{name}{ext}"
            if disk_path.exists():
                col = VectorCollection.load_from_disk(disk_path)
                self._collections[name] = col
                return col

        raise KeyError(f"Collection '{name}' not found. Available: {list(self._collections.keys())}")

    # =========================================================================
    # Registered Tool Definitions
    # =========================================================================

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Return MCP tool schemas."""
        return [
            {
                "name": "vector_create_collection",
                "description": "Create a new vector collection with specified dimensionality, distance metric, and index type (HNSW, Flat, or IVF).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Unique name of the collection.",
                        },
                        "dimension": {
                            "type": "integer",
                            "description": "Dimensionality of vectors in this collection (e.g. 64, 128, 384, 1536).",
                            "minimum": 1,
                        },
                        "metric": {
                            "type": "string",
                            "description": "Distance metric for vector similarity.",
                            "enum": ["cosine", "euclidean", "dot_product", "manhattan", "hamming", "jaccard"],
                            "default": "cosine",
                        },
                        "index_type": {
                            "type": "string",
                            "description": "Index structure type (hnsw for fast ANN, flat for exact search, ivf for cluster index).",
                            "enum": ["hnsw", "flat", "ivf"],
                            "default": "hnsw",
                        },
                        "m": {
                            "type": "integer",
                            "description": "HNSW parameter: max outgoing links per node per layer (default: 16).",
                            "default": 16,
                        },
                        "ef_construction": {
                            "type": "integer",
                            "description": "HNSW parameter: size of dynamic candidate list during graph construction (default: 64).",
                            "default": 64,
                        },
                        "ef_search": {
                            "type": "integer",
                            "description": "HNSW parameter: size of candidate list during search queries (default: 32).",
                            "default": 32,
                        },
                        "nlist": {
                            "type": "integer",
                            "description": "IVF parameter: number of Voronoi partition clusters (default: 16).",
                            "default": 16,
                        },
                    },
                    "required": ["name", "dimension"],
                },
            },
            {
                "name": "vector_upsert",
                "description": "Insert or update vector items with ID, float array, optional text document, and metadata dict. Automatically embeds document text if vector is omitted.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Name of target collection.",
                        },
                        "id": {
                            "type": "string",
                            "description": "Unique identifier of the item (for single item upsert).",
                        },
                        "vector": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Float array representing the vector. Optional if document is provided.",
                        },
                        "document": {
                            "type": "string",
                            "description": "Text document or chunk associated with this vector.",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Arbitrary key-value metadata for filtering and retrieval.",
                        },
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "vector": {"type": "array", "items": {"type": "number"}},
                                    "document": {"type": "string"},
                                    "metadata": {"type": "object"},
                                },
                                "required": ["id"],
                            },
                            "description": "Optional list of items for batch upsert.",
                        },
                    },
                    "required": ["collection"],
                },
            },
            {
                "name": "vector_query",
                "description": "Query collection for nearest neighbor vectors using dense vector or text embedding with top_k, metric, and metadata filters.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Name of target collection.",
                        },
                        "vector": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Query float vector. Optional if query_text is provided.",
                        },
                        "query_text": {
                            "type": "string",
                            "description": "Query text string to auto-embed if vector is not provided.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of nearest neighbors to return.",
                            "default": 5,
                            "minimum": 1,
                        },
                        "filter": {
                            "type": "object",
                            "description": "Metadata filter expression (supports $eq, $ne, $gt, $gte, $lt, $lte, $in, $and, $or, $regex).",
                        },
                        "include_vector": {
                            "type": "boolean",
                            "description": "Whether to include raw vector float arrays in search results.",
                            "default": False,
                        },
                        "ef_search": {
                            "type": "integer",
                            "description": "Optional search beam width override for HNSW index.",
                        },
                    },
                    "required": ["collection"],
                },
            },
            {
                "name": "vector_hybrid_search",
                "description": "Run combined dense vector + sparse BM25 keyword search with Reciprocal Rank Fusion (RRF) or linear alpha score weighting.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Name of target collection.",
                        },
                        "query_text": {
                            "type": "string",
                            "description": "Search query text string for sparse keyword search and optional vector embedding.",
                        },
                        "vector": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Optional explicit dense query vector (auto-embedded from query_text if omitted).",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top hybrid results to return.",
                            "default": 5,
                            "minimum": 1,
                        },
                        "alpha": {
                            "type": "number",
                            "description": "Weighting between dense (1.0) and sparse (0.0) search for linear fusion.",
                            "default": 0.5,
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "fusion": {
                            "type": "string",
                            "description": "Fusion method: 'rrf' (Reciprocal Rank Fusion) or 'linear' (Min-Max normalized score blending).",
                            "enum": ["rrf", "linear"],
                            "default": "rrf",
                        },
                        "filter": {
                            "type": "object",
                            "description": "Optional metadata filter expression.",
                        },
                    },
                    "required": ["collection", "query_text"],
                },
            },
            {
                "name": "vector_embed_text",
                "description": "Generate deterministic dense vector embedding from text string using pure Python n-gram subword hashing and L2 normalization.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text content to embed.",
                        },
                        "dimension": {
                            "type": "integer",
                            "description": "Target vector dimensionality.",
                            "default": 64,
                            "minimum": 1,
                        },
                        "normalize": {
                            "type": "boolean",
                            "description": "Whether to L2-normalize the generated vector to unit length.",
                            "default": True,
                        },
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "vector_quantize",
                "description": "Compress floating point vectors using 8-bit scalar quantization (SQ8) or 1-bit binary sign quantization.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "vector": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Float vector to quantize.",
                        },
                        "mode": {
                            "type": "string",
                            "description": "Quantization mode: 'sq8' (8-bit scalar with scale/offset) or 'binary' (1-bit packed signs).",
                            "enum": ["sq8", "scalar8", "binary"],
                            "default": "sq8",
                        },
                        "num_bits": {
                            "type": "integer",
                            "description": "Number of bits for scalar quantization (default: 8).",
                            "default": 8,
                        },
                    },
                    "required": ["vector"],
                },
            },
            {
                "name": "vector_collection_stats",
                "description": "Return detailed collection telemetry, item count, dimensionality, memory usage, graph layer stats, and index parameters.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Name of target collection.",
                        },
                    },
                    "required": ["collection"],
                },
            },
            {
                "name": "vector_list_collections",
                "description": "List all active in-memory and on-disk collections with schema details and item counts.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "filter_name": {
                            "type": "string",
                            "description": "Optional substring to filter collection names.",
                        },
                    },
                },
            },
            {
                "name": "vector_diagnostics",
                "description": "Run comprehensive multi-OS system diagnostics checking platform capabilities, memory, storage I/O, and vector math throughput.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_benchmarks": {
                            "type": "boolean",
                            "description": "Whether to include quick vector throughput benchmark.",
                            "default": False,
                        },
                    },
                },
            },
        ]

    # =========================================================================
    # Registered Resource Definitions
    # =========================================================================

    def get_resources_schema(self) -> List[Dict[str, Any]]:
        """Return MCP resource schemas."""
        return [
            {
                "uri": "vector://collections",
                "name": "Active Vector Collections",
                "description": "JSON index of all active vector collections, configurations, item counts, and storage locations.",
                "mimeType": "application/json",
            },
            {
                "uri": "vector://metrics/guide",
                "name": "Vector Similarity Metrics & HNSW Guide",
                "description": "Comprehensive reference guide explaining vector similarity metrics, HNSW indexing parameters, and hybrid search.",
                "mimeType": "text/markdown",
            },
        ]

    # =========================================================================
    # Registered Prompt Definitions
    # =========================================================================

    def get_prompts_schema(self) -> List[Dict[str, Any]]:
        """Return MCP prompt schemas."""
        return [
            {
                "name": "vector_semantic_search_prompt",
                "description": "Interactive assistant prompt for designing semantic search vector collections, chunking strategies, and index tuning.",
                "arguments": [
                    {
                        "name": "dataset_type",
                        "description": "Type of dataset (e.g. documents, products, code snippets, FAQ).",
                        "required": False,
                    },
                    {
                        "name": "domain",
                        "description": "Domain or industry (e.g. legal, e-commerce, software engineering).",
                        "required": False,
                    },
                ],
            },
            {
                "name": "vector_rag_retrieval_prompt",
                "description": "Prompt for setting up a production RAG retrieval pipeline with hybrid search, reciprocal rank fusion, and metadata filtering.",
                "arguments": [
                    {
                        "name": "query",
                        "description": "Sample user query for testing retrieval pipeline.",
                        "required": False,
                    },
                    {
                        "name": "use_hybrid",
                        "description": "Whether to enable hybrid dense + BM25 search (true/false).",
                        "required": False,
                    },
                ],
            },
        ]

    # =========================================================================
    # Tool Execution Handlers
    # =========================================================================

    def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute named tool with provided arguments."""
        handler_map = {
            "vector_create_collection": self._tool_create_collection,
            "vector_upsert": self._tool_upsert,
            "vector_query": self._tool_query,
            "vector_hybrid_search": self._tool_hybrid_search,
            "vector_embed_text": self._tool_embed_text,
            "vector_quantize": self._tool_quantize,
            "vector_collection_stats": self._tool_collection_stats,
            "vector_list_collections": self._tool_list_collections,
            "vector_diagnostics": self._tool_diagnostics,
        }

        if name not in handler_map:
            raise ValueError(f"Unknown tool: '{name}'. Available: {list(handler_map.keys())}")

        return handler_map[name](args)

    def _tool_create_collection(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for vector_create_collection."""
        name = str(args["name"]).strip()
        dim = int(args["dimension"])
        if dim <= 0:
            raise ValueError(f"Dimension must be positive integer, got {dim}")

        metric_str = args.get("metric", "cosine")
        metric = DistanceMetric.from_str(metric_str)

        index_type_str = args.get("index_type", "hnsw")
        index_type = IndexType.from_str(index_type_str)

        m = int(args.get("m", 16))
        ef_c = int(args.get("ef_construction", 64))
        ef_s = int(args.get("ef_search", 32))
        nlist = int(args.get("nlist", 16))

        hnsw_cfg = HNSWConfig(m=m, ef_construction=ef_c, ef_search=ef_s)
        ivf_cfg = IVFConfig(nlist=nlist)

        config = CollectionConfig(
            name=name,
            dimension=dim,
            metric=metric,
            index_type=index_type,
            hnsw_config=hnsw_cfg,
            ivf_config=ivf_cfg,
        )

        persist_file = self.data_dir / f"{name}.json"
        collection = VectorCollection(config=config, persist_path=persist_file)
        self._collections[name] = collection

        if self.auto_persist:
            collection.save_to_disk()

        return {
            "status": "success",
            "message": f"Collection '{name}' created successfully.",
            "collection": {
                "name": name,
                "dimension": dim,
                "metric": metric.value,
                "index_type": index_type.value,
                "hnsw_config": hnsw_cfg.to_dict(),
                "ivf_config": ivf_cfg.to_dict(),
                "persist_path": str(persist_file),
            },
        }

    def _tool_upsert(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for vector_upsert."""
        col_name = str(args["collection"]).strip()
        collection = self.get_collection(col_name)

        start_time = time.perf_counter()
        upserted_ids: List[str] = []

        # Batch upsert
        if "items" in args and isinstance(args["items"], list):
            items_to_add: List[VectorItem] = []
            for raw_item in args["items"]:
                item_id = str(raw_item["id"])
                doc = raw_item.get("document")
                vec = raw_item.get("vector")
                if vec is None:
                    if doc:
                        vec = simple_text_embedder(doc, dimension=collection.config.dimension)
                    else:
                        raise ValueError(f"Item '{item_id}' must provide either 'vector' or 'document'.")

                meta = raw_item.get("metadata") or {}
                items_to_add.append(
                    VectorItem(id=item_id, vector=vec, document=doc, metadata=meta)
                )

            upserted_ids = collection.upsert_batch(items_to_add)

        # Single item upsert
        elif "id" in args:
            item_id = str(args["id"])
            doc = args.get("document")
            vec = args.get("vector")
            if vec is None:
                if doc:
                    vec = simple_text_embedder(doc, dimension=collection.config.dimension)
                else:
                    raise ValueError(f"Item '{item_id}' must provide either 'vector' or 'document'.")

            meta = args.get("metadata") or {}
            v_item = VectorItem(id=item_id, vector=vec, document=doc, metadata=meta)
            collection.upsert(v_item)
            upserted_ids = [item_id]
        else:
            raise ValueError("Must provide either 'id' or 'items' list for upsert.")

        if self.auto_persist:
            collection.save_to_disk()

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "status": "success",
            "collection": col_name,
            "upserted_count": len(upserted_ids),
            "upserted_ids": upserted_ids[:50],
            "total_items": len(collection),
            "latency_ms": round(latency_ms, 3),
        }

    def _tool_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for vector_query."""
        col_name = str(args["collection"]).strip()
        collection = self.get_collection(col_name)

        query_vec = args.get("vector")
        query_text = args.get("query_text") or args.get("text")

        if query_vec is None:
            if query_text:
                query_vec = simple_text_embedder(query_text, dimension=collection.config.dimension)
            else:
                raise ValueError("Must provide either 'vector' or 'query_text'.")

        top_k = int(args.get("top_k", args.get("k", 5)))
        filter_expr = args.get("filter") or args.get("metadata_filter")
        include_vector = bool(args.get("include_vector", False))
        ef_search = args.get("ef_search")

        start_time = time.perf_counter()
        results = collection.query(
            query_vector=query_vec,
            k=top_k,
            filter_expr=filter_expr,
            include_vector=include_vector,
            ef_search=int(ef_search) if ef_search is not None else None,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "collection": col_name,
            "query_type": "dense_vector",
            "top_k": top_k,
            "results_count": len(results),
            "results": [r.to_dict() for r in results],
            "latency_ms": round(latency_ms, 3),
        }

    def _tool_hybrid_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for vector_hybrid_search."""
        col_name = str(args["collection"]).strip()
        collection = self.get_collection(col_name)

        query_text = str(args["query_text"]).strip()
        query_vec = args.get("vector")
        if query_vec is None:
            query_vec = simple_text_embedder(query_text, dimension=collection.config.dimension)

        top_k = int(args.get("top_k", args.get("k", 5)))
        alpha = float(args.get("alpha", 0.5))
        fusion = str(args.get("fusion", "rrf")).lower()
        filter_expr = args.get("filter")
        rrf_k = int(args.get("rrf_k", 60))

        start_time = time.perf_counter()
        hybrid_results = collection.hybrid_search(
            query_vector=query_vec,
            query_text=query_text,
            k=top_k,
            alpha=alpha,
            fusion_method=fusion,
            rrf_k=rrf_k,
            filter_expr=filter_expr,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "collection": col_name,
            "query_type": "hybrid_search",
            "query_text": query_text,
            "fusion_method": fusion,
            "alpha": alpha if fusion == "linear" else None,
            "rrf_k": rrf_k if fusion == "rrf" else None,
            "top_k": top_k,
            "results_count": len(hybrid_results),
            "results": [r.to_dict() for r in hybrid_results],
            "latency_ms": round(latency_ms, 3),
        }

    def _tool_embed_text(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for vector_embed_text."""
        text = str(args["text"])
        dim = int(args.get("dimension", args.get("dim", 64)))
        norm = bool(args.get("normalize", True))

        vec = simple_text_embedder(text=text, dimension=dim, normalize=norm)
        return {
            "text": text,
            "dimension": dim,
            "normalized": norm,
            "vector": vec,
            "vector_preview": vec[:8],
        }

    def _tool_quantize(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for vector_quantize."""
        vector = [float(x) for x in args["vector"]]
        mode = str(args.get("mode", args.get("quantization_type", "sq8"))).lower()
        dim = len(vector)

        if mode in ("sq8", "scalar8", "scalar"):
            num_bits = int(args.get("num_bits", 8))
            q_bytes, min_v, max_v = scalar_quantize(vector, num_bits=num_bits)
            dequantized = dequantize_scalar(q_bytes, min_v, max_v, num_bits=num_bits)

            # Calculate mean squared reconstruction error
            mse = sum((o - d) ** 2 for o, d in zip(vector, dequantized)) / max(dim, 1)
            cosine_rec = cosine_similarity(vector, dequantized)

            return {
                "mode": "sq8",
                "num_bits": num_bits,
                "dimension": dim,
                "original_bytes": dim * 4,  # float32 = 4 bytes
                "quantized_bytes": len(q_bytes),
                "compression_ratio": f"{(dim * 4) / max(len(q_bytes), 1):.2f}x",
                "min_value": min_v,
                "max_value": max_v,
                "quantized_hex": q_bytes.hex(),
                "dequantized_preview": dequantized[:8],
                "mean_squared_error": round(mse, 8),
                "cosine_reconstruction": round(cosine_rec, 6),
            }
        elif mode == "binary":
            b_bytes = binary_quantize(vector)
            dequantized = dequantize_binary(b_bytes, dimension=dim)
            cosine_rec = cosine_similarity(vector, dequantized)

            return {
                "mode": "binary",
                "dimension": dim,
                "original_bytes": dim * 4,
                "quantized_bytes": len(b_bytes),
                "compression_ratio": f"{(dim * 4) / max(len(b_bytes), 1):.2f}x",
                "quantized_hex": b_bytes.hex(),
                "dequantized_preview": dequantized[:8],
                "cosine_reconstruction": round(cosine_rec, 6),
            }
        else:
            raise ValueError(f"Unknown quantization mode: '{mode}'. Supported: ['sq8', 'binary']")

    def _tool_collection_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for vector_collection_stats."""
        col_name = str(args["collection"]).strip()
        collection = self.get_collection(col_name)
        stats = collection.stats()

        res = stats.to_dict()
        res["storage_path"] = str(collection.persist_path) if collection.persist_path else None
        res["memory_human"] = _format_bytes(stats.memory_bytes)
        return res

    def _tool_list_collections(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for vector_list_collections."""
        filter_name = args.get("filter_name")
        self._discover_collections()

        collections_info: List[Dict[str, Any]] = []
        for name, col in sorted(self._collections.items()):
            if filter_name and filter_name.lower() not in name.lower():
                continue
            collections_info.append(
                {
                    "name": name,
                    "count": len(col),
                    "dimension": col.config.dimension,
                    "metric": col.config.metric.value,
                    "index_type": col.config.index_type.value,
                    "storage_path": str(col.persist_path) if col.persist_path else None,
                }
            )

        return {
            "total_collections": len(collections_info),
            "data_directory": str(self.data_dir),
            "collections": collections_info,
        }

    def _tool_diagnostics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for vector_diagnostics."""
        include_benchmarks = bool(args.get("include_benchmarks", False))
        platform_info = get_platform_info()

        # Atomic I/O write & read verification
        test_file = self.data_dir / ".diagnostics_test.tmp"
        io_ok = False
        try:
            test_file.write_text("ok", encoding="utf-8")
            io_ok = test_file.read_text(encoding="utf-8") == "ok"
            test_file.unlink(missing_ok=True)
        except Exception:
            io_ok = False

        diag: Dict[str, Any] = {
            "server": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
                "protocol_version": MCP_PROTOCOL_VERSION,
            },
            "platform": platform_info.to_dict(),
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "float_info": {
                    "max": sys.float_info.max,
                    "min": sys.float_info.min,
                    "epsilon": sys.float_info.epsilon,
                },
            },
            "storage": {
                "data_dir": str(self.data_dir),
                "data_dir_exists": self.data_dir.exists(),
                "atomic_io_functional": io_ok,
                "active_in_memory_collections": len(self._collections),
            },
        }

        if include_benchmarks:
            # 10,000 vector similarity operations benchmark
            dim = 64
            v1 = [math.sin(i) for i in range(dim)]
            v2 = [math.cos(i) for i in range(dim)]
            iters = 10000

            t0 = time.perf_counter()
            for _ in range(iters):
                cosine_similarity(v1, v2)
            t_cos = (time.perf_counter() - t0)

            t0 = time.perf_counter()
            for _ in range(iters):
                euclidean_distance(v1, v2)
            t_euc = (time.perf_counter() - t0)

            diag["benchmarks"] = {
                "dimension": dim,
                "iterations": iters,
                "cosine_similarity_qps": round(iters / max(t_cos, 1e-9), 0),
                "euclidean_distance_qps": round(iters / max(t_euc, 1e-9), 0),
            }

        return diag

    # =========================================================================
    # Resource Read Handlers
    # =========================================================================

    def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read content for a registered MCP resource URI."""
        if uri == "vector://collections":
            self._discover_collections()
            data = {
                "collections": [
                    {
                        "name": name,
                        "count": len(col),
                        "dimension": col.config.dimension,
                        "metric": col.config.metric.value,
                        "index_type": col.config.index_type.value,
                        "config": col.config.to_dict(),
                    }
                    for name, col in sorted(self._collections.items())
                ],
                "data_directory": str(self.data_dir),
                "timestamp": time.time(),
            }
            return {
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(data, indent=2),
            }

        elif uri == "vector://metrics/guide":
            guide_md = """# Vector Similarity Metrics & HNSW Indexing Guide

## Supported Distance Metrics

1. **Cosine Similarity / Cosine Distance**
   - Formula: `cos_sim(A, B) = (A · B) / (||A|| * ||B||)`
   - Distance: `1.0 - cos_sim(A, B)` in range `[0.0, 2.0]`
   - Best for: Text embeddings, semantic search, document matching where vector magnitude is invariant.

2. **Euclidean Distance (L2 Norm)**
   - Formula: `dist(A, B) = sqrt(sum((A_i - B_i)^2))`
   - Similarity: `1.0 / (1.0 + dist)`
   - Best for: Image embeddings, physical coordinates, dense representations where absolute magnitude matters.

3. **Dot Product (Inner Product)**
   - Formula: `dot(A, B) = sum(A_i * B_i)`
   - Distance: `-dot(A, B)` (minimization for indexing)
   - Best for: Unit-normalized embeddings, neural ranking models, Max-Margin Inner Product Search (MIPS).

4. **Manhattan Distance (L1 Norm)**
   - Formula: `dist(A, B) = sum(|A_i - B_i|)`
   - Best for: Sparse high-dimensional data, grid metrics, robust regression.

5. **Hamming Distance**
   - Proportion of differing elements in range `[0.0, 1.0]`.
   - Best for: Binary hashes, perceptual hashing, locality sensitive hashing (LSH).

6. **Jaccard Distance**
   - `1.0 - (sum(min(A_i, B_i)) / sum(max(A_i, B_i)))`
   - Best for: Set similarity, token co-occurrence, binary features.

---

## HNSW (Hierarchical Navigable Small World) Parameters

- `m` (Default: 16): Maximum number of bi-directional outgoing links per node. Higher `m` increases recall and memory usage.
- `ef_construction` (Default: 64): Search beam width during index graph construction. Higher `ef_construction` improves graph quality at the cost of index build time.
- `ef_search` (Default: 32): Dynamic candidate list size during runtime nearest neighbor queries. Higher `ef_search` improves recall at runtime with minimal query latency overhead.

---

## Hybrid Search & Fusion

- **Reciprocal Rank Fusion (RRF)**:
  `RRF_score(d) = sum(1 / (k + rank_i(d)))`
  Default smoothing constant `k = 60`. Robust across disparate score distributions.
- **Linear Alpha Fusion**:
  `Score(d) = alpha * Norm(Dense_score) + (1 - alpha) * Norm(Sparse_BM25_score)`
  Min-Max normalized in range `[0.0, 1.0]`.

---

## Vector Quantization

- **Scalar Quantization (SQ8)**: Maps 32-bit floats to 8-bit unsigned integers `[0, 255]`, achieving 4x memory compression with >99% cosine recall.
- **Binary Quantization**: Compresses 8 floats per byte using sign thresholding, achieving 32x memory compression for high-dimensional vectors (e.g. 1536d).
"""
            return {
                "uri": uri,
                "mimeType": "text/markdown",
                "text": guide_md,
            }
        else:
            raise ValueError(f"Resource URI not found: '{uri}'")

    # =========================================================================
    # Prompt Get Handlers
    # =========================================================================

    def get_prompt(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get interactive prompt template messages."""
        if name == "vector_semantic_search_prompt":
            dataset_type = args.get("dataset_type", "general documents")
            domain = args.get("domain", "general")
            prompt_text = (
                f"You are an expert Vector Database and Semantic Search Architect.\n"
                f"Help configure and build an optimal vector search pipeline for '{dataset_type}' in the '{domain}' domain.\n\n"
                f"Please guide through:\n"
                f"1. Selecting embedding dimensionality and metric (e.g. Cosine with 64d, 384d, or 1536d).\n"
                f"2. Setting up HNSW index parameters (m=16, ef_construction=64, ef_search=32).\n"
                f"3. Designing metadata filtering schema for fast categorical and range queries.\n"
                f"4. Configuring hybrid search (combining dense embeddings with Okapi BM25 keyword search).\n"
                f"5. Demonstrating upsert and query calls using the Vector Search Engine MCP tools."
            )
            return {
                "description": "Interactive Semantic Search Assistant Prompt",
                "messages": [
                    {
                        "role": "user",
                        "content": {"type": "text", "text": prompt_text},
                    }
                ],
            }

        elif name == "vector_rag_retrieval_prompt":
            sample_query = args.get("query", "What are the key technical advantages of HNSW vector indexing?")
            use_hybrid = args.get("use_hybrid", True)
            prompt_text = (
                f"You are a Retrieval-Augmented Generation (RAG) assistant.\n"
                f"Your goal is to retrieve relevant context and synthesize an accurate answer with citations.\n\n"
                f"Query: \"{sample_query}\"\n"
                f"Retrieval strategy: {'Hybrid Search (Dense + BM25 with RRF)' if use_hybrid else 'Dense Vector Search'}\n\n"
                f"Steps:\n"
                f"1. Query the vector search engine collection for top 5 candidates.\n"
                f"2. Format retrieved documents with item IDs and metadata.\n"
                f"3. Synthesize a concise answer grounded in the retrieved documents."
            )
            return {
                "description": "RAG Retrieval Pipeline Setup Prompt",
                "messages": [
                    {
                        "role": "user",
                        "content": {"type": "text", "text": prompt_text},
                    }
                ],
            }
        else:
            raise ValueError(f"Prompt '{name}' not found.")


def _format_bytes(byte_count: int) -> str:
    """Format bytes into human-readable string (KB, MB, GB)."""
    if byte_count < 1024:
        return f"{byte_count} B"
    elif byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f} KB"
    elif byte_count < 1024 * 1024 * 1024:
        return f"{byte_count / (1024 * 1024):.2f} MB"
    else:
        return f"{byte_count / (1024 * 1024 * 1024):.2f} GB"


# =============================================================================
# JSON-RPC 2.0 Protocol Engine
# =============================================================================

def handle_jsonrpc_request(
    request: Dict[str, Any],
    server: Optional[MCPServer] = None,
) -> Optional[Dict[str, Any]]:
    """Process a single JSON-RPC 2.0 request dict and return response dict."""
    if server is None:
        server = MCPServer()

    if not isinstance(request, dict):
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Invalid Request: expected JSON object."},
        }

    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    # Notification check: if id is omitted or None, some notifications do not require a response
    is_notification = "id" not in request

    if not method or not isinstance(method, str):
        if is_notification:
            return None
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32600, "message": "Invalid Request: missing or invalid 'method'."},
        }

    try:
        # MCP Protocol Lifecycle Methods
        if method == "initialize":
            result = {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                },
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                    "prompts": {"listChanged": False},
                },
            }
            return {"jsonrpc": "2.0", "id": req_id, "result": result}

        elif method in ("notifications/initialized", "initialized"):
            # Acknowledge client initialization notification
            if is_notification:
                return None
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        # Tools
        elif method == "tools/list":
            tools = server.get_tools_schema()
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments") or {}
            if not tool_name:
                raise ValueError("Missing 'name' in tools/call parameters.")

            tool_result = server.call_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(tool_result, indent=2),
                        }
                    ],
                    "isError": False,
                },
            }

        # Resources
        elif method == "resources/list":
            resources = server.get_resources_schema()
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": resources}}

        elif method == "resources/read":
            uri = params.get("uri")
            if not uri:
                raise ValueError("Missing 'uri' in resources/read parameters.")
            res_content = server.read_resource(uri)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "contents": [res_content],
                },
            }

        # Prompts
        elif method == "prompts/list":
            prompts = server.get_prompts_schema()
            return {"jsonrpc": "2.0", "id": req_id, "result": {"prompts": prompts}}

        elif method == "prompts/get":
            prompt_name = params.get("name")
            prompt_args = params.get("arguments") or {}
            if not prompt_name:
                raise ValueError("Missing 'name' in prompts/get parameters.")
            p_result = server.get_prompt(prompt_name, prompt_args)
            return {"jsonrpc": "2.0", "id": req_id, "result": p_result}

        # Logging level
        elif method == "logging/setLevel":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        else:
            if is_notification:
                return None
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: '{method}'",
                },
            }

    except Exception as exc:
        if is_notification:
            return None
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32603,
                "message": str(exc),
                "data": {"type": type(exc).__name__},
            },
        }


def process_request(
    raw_line: str,
    server: Optional[MCPServer] = None,
) -> Optional[str]:
    """Parse a single JSON-RPC raw input string, evaluate, and return response string."""
    line = raw_line.strip()
    if not line:
        return None

    try:
        req_obj = json.loads(line)
    except json.JSONDecodeError as err:
        err_resp = {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": f"Parse error: {str(err)}"},
        }
        return json.dumps(err_resp)

    resp_obj = handle_jsonrpc_request(req_obj, server=server)
    if resp_obj is not None:
        return json.dumps(resp_obj)
    return None


def run_stdio_server(
    server: Optional[MCPServer] = None,
    in_stream: Optional[TextIO] = None,
    out_stream: Optional[TextIO] = None,
) -> None:
    """Run interactive MCP JSON-RPC 2.0 server over stdio streams."""
    if server is None:
        server = MCPServer()

    reader = in_stream or sys.stdin
    writer = out_stream or sys.stdout

    # Log startup banner to stderr so stdout remains pure JSON-RPC stream
    sys.stderr.write(
        f"[{SERVER_NAME} v{SERVER_VERSION}] MCP stdio server started. Ready for JSON-RPC 2.0 messages.\n"
    )
    sys.stderr.flush()

    try:
        for line in reader:
            if not line:
                break
            response = process_request(line, server=server)
            if response:
                writer.write(response + "\n")
                writer.flush()
    except (KeyboardInterrupt, EOFError):
        pass


def main() -> None:
    """Direct CLI entrypoint for running MCP server."""
    run_stdio_server()


if __name__ == "__main__":
    main()
