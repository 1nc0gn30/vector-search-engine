"""Pure Python UI server and REST API for Google Vector Studio.

Serves Google Vector Studio web interface and provides REST endpoints for
collection management, dense vector search, sparse BM25 search, hybrid RRF
fusion, graph visualizer inspection, diagnostics, and synthetic benchmarks.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, unquote, urlparse

from vector_search_engine.compat import (
    get_platform_info,
    safe_ensure_dir,
    safe_path_normalization,
)
from vector_search_engine.models import (
    CollectionConfig,
    CollectionStats,
    DistanceMetric,
    HNSWConfig,
    IndexType,
    IVFConfig,
    SearchResult,
    VectorItem,
)
from vector_search_engine.indexes import FlatIndex, HNSWIndex
from vector_search_engine.metrics import generate_random_vector
from vector_search_engine.collection import VectorCollection

# Global in-memory registry of collections
_COLLECTIONS_REGISTRY: Dict[str, VectorCollection] = {}
_REGISTRY_LOCK = threading.Lock()
_START_TIME = time.time()

# Root directory for public static assets
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
PUBLIC_DIR = PACKAGE_ROOT / "public"

EMBEDDED_HTML_FALLBACK = """<!DOCTYPE html>
<html>
<head><title>Google Vector Studio</title></head>
<body><h1>Google Vector Studio</h1><p>Embedded UI active.</p></body>
</html>"""


class StudioRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler providing REST API and static asset delivery."""

    def _set_headers(
        self,
        status_code: int = 200,
        content_type: str = "application/json",
    ) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"
        )
        self.send_header(
            "Access-Control-Allow-Headers", "Content-Type, Authorization"
        )
        self.end_headers()

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self._set_headers(204)

    def _send_json(self, data: Any, status_code: int = 200) -> None:
        """Serialize and send JSON response with Content-Length header."""
        try:
            body = json.dumps(data, indent=2, default=str).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_error_json(self, message: str, status_code: int = 400) -> None:
        """Send JSON error payload."""
        self._send_json({"error": message, "status": status_code}, status_code=status_code)

    def _parse_json_body(self) -> Dict[str, Any]:
        """Read and parse JSON body from request."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            return {}
        raw_body = self.rfile.read(content_length).decode("utf-8")
        if not raw_body.strip():
            return {}
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError as err:
            raise ValueError(f"Malformed JSON request body: {err}")

    def do_GET(self) -> None:
        """Route GET requests for static files and REST endpoints."""
        parsed_url = urlparse(self.path)
        path = unquote(parsed_url.path)

        # Health endpoint
        if path in ("/api/health", "/health"):
            self._send_json({
                "status": "ok",
                "version": "0.1.0",
                "service": "vector-search-engine",
                "uptime_seconds": round(time.time() - _START_TIME, 2),
            })
            return

        # Diagnostics endpoint
        if path == "/api/diagnostics":
            plat = get_platform_info()
            with _REGISTRY_LOCK:
                total_vectors = sum(len(c) for c in _COLLECTIONS_REGISTRY.values())
                cols_count = len(_COLLECTIONS_REGISTRY)

            self._send_json({
                "platform": plat.to_dict(),
                "uptime_seconds": round(time.time() - _START_TIME, 2),
                "collections_count": cols_count,
                "total_vectors": total_vectors,
            })
            return

        # Collections list
        if path == "/api/collections":
            with _REGISTRY_LOCK:
                collections_list = []
                for name, col in _COLLECTIONS_REGISTRY.items():
                    stats = col.stats()
                    collections_list.append({
                        "name": name,
                        "count": stats.count,
                        "dimension": stats.dimension,
                        "metric": stats.metric,
                        "index_type": stats.index_type,
                        "memory_bytes": stats.memory_bytes,
                    })
            self._send_json({"collections": collections_list})
            return

        # Collection detail / stats / items
        if path.startswith("/api/collections/"):
            sub_path = path[len("/api/collections/"):]
            parts = sub_path.split("/")
            col_name = parts[0]

            with _REGISTRY_LOCK:
                col = _COLLECTIONS_REGISTRY.get(col_name)

            if col is None:
                self._send_error_json(f"Collection '{col_name}' not found", status_code=404)
                return

            if len(parts) == 1:
                # GET /api/collections/<name>
                self._send_json({
                    "name": col_name,
                    "config": col.config.to_dict(),
                    "count": col.count,
                    "stats": col.stats().to_dict(),
                })
                return
            elif len(parts) == 2 and parts[1] == "stats":
                # GET /api/collections/<name>/stats
                self._send_json(col.stats().to_dict())
                return
            elif len(parts) == 2 and parts[1] in ("items", "export"):
                # GET /api/collections/<name>/items
                with _REGISTRY_LOCK:
                    items_data = [item.to_dict() for item in col._items.values()]
                self._send_json({"collection": col_name, "count": len(items_data), "items": items_data})
                return

        # Static Asset Serving
        self._serve_static_file(path)

    def _serve_static_file(self, path: str) -> None:
        """Serve public frontend assets or fallback embedded HTML."""
        if path in ("/", "/index.html", ""):
            index_path = PUBLIC_DIR / "index.html"
            if index_path.exists():
                content = index_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(content)
            else:
                content = EMBEDDED_HTML_FALLBACK.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(content)
            return

        # Attempt to serve relative file from public directory
        rel_path = path.lstrip("/")
        target_file = PUBLIC_DIR / rel_path
        if target_file.exists() and target_file.is_file():
            mime_type, _ = mimetypes.guess_type(str(target_file))
            mime_type = mime_type or "application/octet-stream"
            content = target_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        else:
            self._send_error_json(f"Not Found: {path}", status_code=404)

    def do_POST(self) -> None:
        """Route POST requests for collection creation, inserts, queries, benchmarks."""
        parsed_url = urlparse(self.path)
        path = unquote(parsed_url.path)

        try:
            body = self._parse_json_body()
        except ValueError as e:
            self._send_error_json(str(e), status_code=400)
            return

        # POST /api/collections (Create collection)
        if path == "/api/collections":
            col_name = body.get("name")
            if not col_name:
                self._send_error_json("Missing required field 'name'", status_code=400)
                return

            dim = int(body.get("dimension", 128))
            metric = body.get("metric", "cosine")
            index_type = body.get("index_type", "hnsw")
            hnsw_cfg = body.get("hnsw_config")
            ivf_cfg = body.get("ivf_config")

            config = CollectionConfig(
                name=col_name,
                dimension=dim,
                metric=DistanceMetric.from_str(metric),
                index_type=IndexType.from_str(index_type),
                hnsw_config=HNSWConfig.from_dict(hnsw_cfg) if hnsw_cfg else HNSWConfig(),
                ivf_config=IVFConfig.from_dict(ivf_cfg) if ivf_cfg else None,
            )

            with _REGISTRY_LOCK:
                _COLLECTIONS_REGISTRY[col_name] = VectorCollection(config=config)

            self._send_json({"status": "created", "name": col_name, "config": config.to_dict()}, status_code=201)
            return

        # POST /api/benchmark (Run synthetic performance benchmark)
        if path == "/api/benchmark":
            res = self._run_benchmark(body)
            self._send_json(res)
            return

        # Sub-routes under /api/collections/<name>
        if path.startswith("/api/collections/"):
            sub_path = path[len("/api/collections/"):]
            parts = sub_path.split("/")
            col_name = parts[0]

            with _REGISTRY_LOCK:
                col = _COLLECTIONS_REGISTRY.get(col_name)

            if col is None:
                self._send_error_json(f"Collection '{col_name}' not found", status_code=404)
                return

            if len(parts) == 2 and parts[1] == "insert":
                # POST /api/collections/<name>/insert
                docs = body.get("documents") or body.get("items") or []
                if isinstance(docs, dict):
                    docs = [docs]
                if not isinstance(docs, list):
                    self._send_error_json("Expected 'documents' to be a list", status_code=400)
                    return

                inserted_ids = []
                for doc in docs:
                    doc_id = str(doc.get("id") or f"doc_{len(col) + 1}")
                    vec = doc.get("vector")
                    if not vec:
                        vec = generate_random_vector(col.config.dimension, normalized=True)
                    item = VectorItem(
                        id=doc_id,
                        vector=vec,
                        document=doc.get("document") or doc.get("text"),
                        metadata=doc.get("metadata") or {},
                    )
                    col.upsert(item)
                    inserted_ids.append(doc_id)

                self._send_json({"status": "inserted", "count": len(inserted_ids), "ids": inserted_ids})
                return

            elif len(parts) == 2 and parts[1] in ("query", "search"):
                # POST /api/collections/<name>/query
                vector = body.get("vector")
                if not vector:
                    self._send_error_json("Missing required field 'vector'", status_code=400)
                    return
                k = int(body.get("k") or body.get("top_k") or 10)
                filter_expr = body.get("filter") or body.get("filter_expr")
                include_vector = bool(body.get("include_vector", False))

                t0 = time.perf_counter()
                results = col.query(
                    query_vector=vector,
                    k=k,
                    filter_expr=filter_expr,
                    include_vector=include_vector,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0

                self._send_json({
                    "results": [r.to_dict() for r in results],
                    "count": len(results),
                    "latency_ms": round(latency_ms, 3),
                })
                return

            elif len(parts) == 2 and parts[1] == "hybrid":
                # POST /api/collections/<name>/hybrid
                query_text = body.get("query_text") or body.get("query") or ""
                query_vec = body.get("query_vector") or body.get("vector")
                if not query_vec:
                    if col.count > 0:
                        query_vec = next(iter(col._items.values())).vector
                    else:
                        query_vec = generate_random_vector(col.config.dimension)

                k = int(body.get("k") or body.get("top_k") or 10)
                alpha = float(body.get("alpha", 0.5))
                fusion_method = str(body.get("fusion_method", "rrf"))
                filter_expr = body.get("filter") or body.get("filter_expr")

                t0 = time.perf_counter()
                hybrid_res = col.hybrid_search(
                    query_vector=query_vec,
                    query_text=query_text,
                    k=k,
                    alpha=alpha,
                    fusion_method=fusion_method,
                    filter_expr=filter_expr,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0

                self._send_json({
                    "results": [r.to_dict() for r in hybrid_res],
                    "count": len(hybrid_res),
                    "latency_ms": round(latency_ms, 3),
                })
                return

        self._send_error_json(f"Unknown POST endpoint: {path}", status_code=404)

    def do_DELETE(self) -> None:
        """Route DELETE requests for collection deletion."""
        parsed_url = urlparse(self.path)
        path = unquote(parsed_url.path)

        if path.startswith("/api/collections/"):
            col_name = path[len("/api/collections/"):].split("/")[0]
            with _REGISTRY_LOCK:
                if col_name in _COLLECTIONS_REGISTRY:
                    del _COLLECTIONS_REGISTRY[col_name]
                    self._send_json({"status": "deleted", "name": col_name})
                    return

            self._send_error_json(f"Collection '{col_name}' not found", status_code=404)
            return

        self._send_error_json(f"Unknown DELETE endpoint: {path}", status_code=404)

    def _run_benchmark(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Execute in-memory synthetic benchmark comparing Flat vs HNSW."""
        num_vecs = int(body.get("num_vectors", 300))
        dim = int(body.get("dimension", 32))
        queries_count = int(body.get("queries_count", 30))
        k = int(body.get("k", 10))
        metric = DistanceMetric.from_str(body.get("metric", "cosine"))

        # Generate vectors
        vectors = [generate_random_vector(dim, seed=i, normalized=True) for i in range(num_vecs)]
        query_vectors = [
            generate_random_vector(dim, seed=1000 + j, normalized=True)
            for j in range(queries_count)
        ]

        # Flat Index build & query
        flat = FlatIndex(dimension=dim, metric=metric)
        flat.add_batch([VectorItem(f"v_{i}", v) for i, v in enumerate(vectors)])

        # HNSW Index build & query
        t_build_start = time.perf_counter()
        hnsw = HNSWIndex(dimension=dim, metric=metric, config=HNSWConfig(m=16, ef_construction=64, ef_search=32), seed=42)
        hnsw.add_batch([VectorItem(f"v_{i}", v) for i, v in enumerate(vectors)])
        build_time_ms = (time.perf_counter() - t_build_start) * 1000.0

        # Benchmark Flat queries
        flat_latencies = []
        flat_ground_truth = []
        for q in query_vectors:
            t0 = time.perf_counter()
            res = flat.search(q, k=k)
            flat_latencies.append((time.perf_counter() - t0) * 1000.0)
            flat_ground_truth.append({r.id for r in res})

        # Benchmark HNSW queries
        hnsw_latencies = []
        hnsw_matches = 0
        total_k = queries_count * k

        for idx, q in enumerate(query_vectors):
            t0 = time.perf_counter()
            res = hnsw.search(q, k=k)
            hnsw_latencies.append((time.perf_counter() - t0) * 1000.0)
            hnsw_ids = {r.id for r in res}
            overlap = len(hnsw_ids & flat_ground_truth[idx])
            hnsw_matches += overlap

        recall = hnsw_matches / max(total_k, 1)
        flat_avg_lat = sum(flat_latencies) / len(flat_latencies)
        hnsw_avg_lat = sum(hnsw_latencies) / len(hnsw_latencies)

        hnsw_latencies.sort()
        p50 = hnsw_latencies[int(len(hnsw_latencies) * 0.50)]
        p95 = hnsw_latencies[int(len(hnsw_latencies) * 0.95)]

        hnsw_qps = (1000.0 / hnsw_avg_lat) if hnsw_avg_lat > 0 else 0
        flat_qps = (1000.0 / flat_avg_lat) if flat_avg_lat > 0 else 0
        speedup = flat_avg_lat / max(hnsw_avg_lat, 1e-6)

        return {
            "num_vectors": num_vecs,
            "dimension": dim,
            "queries_count": queries_count,
            "k": k,
            "build_time_ms": round(build_time_ms, 2),
            "hnsw_avg_latency_ms": round(hnsw_avg_lat, 3),
            "hnsw_p50_ms": round(p50, 3),
            "hnsw_p95_ms": round(p95, 3),
            "flat_avg_latency_ms": round(flat_avg_lat, 3),
            "hnsw_qps": round(hnsw_qps, 1),
            "flat_qps": round(flat_qps, 1),
            "recall_at_k": round(recall, 4),
            "speedup_factor": round(speedup, 2),
        }

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress noisy default request logging during automated tests."""
        if os.environ.get("VECTOR_DEBUG_HTTP") == "1":
            super().log_message(format, *args)


def start_ui_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    collections: Optional[Dict[str, VectorCollection]] = None,
    background: bool = False,
) -> ThreadingHTTPServer:
    """Start ThreadingHTTPServer serving Google Vector Studio and REST API."""
    if collections:
        with _REGISTRY_LOCK:
            _COLLECTIONS_REGISTRY.update(collections)

    server = ThreadingHTTPServer((host, port), StudioRequestHandler)
    if background:
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
    return server


def run_ui_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run UI server in foreground."""
    server = start_ui_server(host=host, port=port, background=False)
    print(f"🚀 Google Vector Studio UI running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Google Vector Studio UI server...")
    finally:
        server.server_close()


def main() -> None:
    """CLI entry point for running UI server."""
    parser = argparse.ArgumentParser(description="Google Vector Studio UI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()
    run_ui_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
