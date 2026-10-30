#!/usr/bin/env python3
"""
Vector Search Engine Local Workbench Server (:8121)
Serves public/ with /health and /api/stats endpoints.
"""
import sys, os, json, http.server, socketserver, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "public"
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

class VectorSearchRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

    def do_GET(self):
        if self.path in ("/health", "/api/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            payload = {
                "status": "healthy",
                "service": "vector-search-engine",
                "version": "2.5.0",
                "sovereign": True,
                "engine": "HNSW + Okapi BM25 RRF Hybrid"
            }
            self.wfile.write(json.dumps(payload, indent=2).encode("utf-8"))
            return
        elif self.path in ("/api/stats", "/stats"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            payload = {
                "collections": 3,
                "total_vectors": 1250,
                "dimensions": 384,
                "index_type": "HNSW",
                "m_parameter": 16,
                "ef_search": 64
            }
            self.wfile.write(json.dumps(payload, indent=2).encode("utf-8"))
            return
        return super().do_GET()

def main():
    parser = argparse.ArgumentParser(description="Vector Search Engine Local Server")
    parser.add_argument("--port", type=int, default=8121)
    parser.add_argument("--host", default="127.0.0.1")
    args, _ = parser.parse_known_args()

    class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    print(f"Launching Vector Search Engine on http://{args.host}:{args.port} ...")
    with ThreadedServer((args.host, args.port), VectorSearchRequestHandler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    main()
