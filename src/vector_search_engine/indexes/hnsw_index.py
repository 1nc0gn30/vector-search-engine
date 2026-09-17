"""Pure Python Hierarchical Navigable Small World (HNSW) graph index.

Implements multi-layer skip-graph indexing with greedy top-layer routing,
beam-search layer traversal, heuristic neighbor selection, dynamic ef tuning,
metadata filtering, and deletion support without third-party dependencies.
"""

from __future__ import annotations

import heapq
import math
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from vector_search_engine.models import (
    DistanceMetric,
    HNSWConfig,
    IndexType,
    SearchResult,
    VectorItem,
)
from vector_search_engine.metrics import (
    compute_distance,
    distance_to_score,
)
from vector_search_engine.indexes import BaseIndex, evaluate_filter


class HNSWIndex(BaseIndex):
    """Hierarchical Navigable Small World (HNSW) Graph Index."""

    def __init__(
        self,
        dimension: int,
        metric: Union[DistanceMetric, str] = DistanceMetric.COSINE,
        config: Optional[HNSWConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.dimension = int(dimension)
        self.metric = DistanceMetric.from_str(metric)
        self.config = config or HNSWConfig()
        self._rng = random.Random(seed) if seed is not None else random.Random()

        # Storage
        self._nodes: Dict[str, VectorItem] = {}
        self._levels: Dict[str, int] = {}
        # Multi-layer adjacency lists: _graph[layer][node_id] -> List[neighbor_id]
        self._graph: List[Dict[str, List[str]]] = [
            {} for _ in range(self.config.max_layers)
        ]
        self._enter_point: Optional[str] = None
        self._max_level: int = -1

    def _random_level(self) -> int:
        """Generate random level for a new node according to exponential distribution."""
        u = self._rng.random()
        if u == 0:
            u = 1e-9
        level_mult = self.config.level_mult or (1.0 / math.log(max(self.config.m, 2)))
        level = int(-math.log(u) * level_mult)
        return min(level, self.config.max_layers - 1)

    def size(self) -> int:
        """Return total number of items."""
        return len(self._nodes)

    def get(self, item_id: str) -> Optional[VectorItem]:
        """Get item by ID."""
        return self._nodes.get(item_id)

    def clear(self) -> None:
        """Clear all items and reset graph structure."""
        self._nodes.clear()
        self._levels.clear()
        self._graph = [{} for _ in range(self.config.max_layers)]
        self._enter_point = None
        self._max_level = -1

    def _dist(self, vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        """Compute distance between two raw vectors."""
        return compute_distance(vec_a, vec_b, self.metric)

    def _search_layer(
        self,
        query_vector: Sequence[float],
        enter_points: Sequence[str],
        ef: int,
        layer: int,
    ) -> List[Tuple[float, str]]:
        """Search a single graph layer using priority queues (beam search).

        Returns list of (distance, node_id) sorted by distance ascending.
        """
        visited: Set[str] = set(enter_points)
        # candidates (C): min-heap of (dist, node_id)
        candidates: List[Tuple[float, str]] = []
        # nearest elements (W): max-heap of (-dist, node_id)
        nearest: List[Tuple[float, str]] = []

        for ep in enter_points:
            if ep not in self._nodes:
                continue
            d = self._dist(query_vector, self._nodes[ep].vector)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(nearest, (-d, ep))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_d = -nearest[0][0]

            if c_dist > furthest_d and len(nearest) >= ef:
                break

            neighbors = self._graph[layer].get(c_id, [])
            for nb_id in neighbors:
                if nb_id not in visited:
                    visited.add(nb_id)
                    if nb_id not in self._nodes:
                        continue

                    nb_vec = self._nodes[nb_id].vector
                    nb_dist = self._dist(query_vector, nb_vec)
                    furthest_d = -nearest[0][0]

                    if nb_dist < furthest_d or len(nearest) < ef:
                        heapq.heappush(candidates, (nb_dist, nb_id))
                        heapq.heappush(nearest, (-nb_dist, nb_id))

                        if len(nearest) > ef:
                            heapq.heappop(nearest)

        # Convert nearest max-heap back to positive distances sorted ascending
        result = [(-d, node_id) for d, node_id in nearest]
        result.sort(key=lambda x: x[0])
        return result

    def _select_neighbors(
        self,
        query_vector: Sequence[float],
        candidates: List[Tuple[float, str]],
        m_max: int,
        layer: int,
    ) -> List[str]:
        """Select up to m_max neighbors from candidates using simple or heuristic pruning."""
        if not self.config.heuristic_pruning:
            # Simple selection: closest m_max candidates
            return [node_id for _, node_id in candidates[:m_max]]

        # Heuristic neighbor selection (Algorithm 4 from Malkov & Yashunin)
        # Keeps candidates that are closer to query than to any already chosen neighbor
        result: List[str] = []
        # Sorted candidate queue (dist, node_id)
        candidate_queue = list(candidates)
        candidate_queue.sort(key=lambda x: x[0])

        discarded: List[str] = []

        for c_dist, c_id in candidate_queue:
            if len(result) >= m_max:
                break
            if c_id not in self._nodes:
                continue
            c_vec = self._nodes[c_id].vector

            # Check if closer to query than to any selected neighbor
            is_closer = True
            for r_id in result:
                r_vec = self._nodes[r_id].vector
                dist_to_r = self._dist(c_vec, r_vec)
                if dist_to_r < c_dist:
                    is_closer = False
                    break

            if is_closer:
                result.append(c_id)
            else:
                discarded.append(c_id)

        # If result is smaller than m_max, fill from discarded to maintain connectivity
        for d_id in discarded:
            if len(result) >= m_max:
                break
            if d_id not in result:
                result.append(d_id)

        return result

    def add(self, item: VectorItem) -> None:
        """Add an item into the HNSW index."""
        if len(item.vector) != self.dimension:
            raise ValueError(
                f"Item vector dimension {len(item.vector)} does not match index dimension {self.dimension}"
            )

        # If item already exists, delete it first to maintain clean graph
        if item.id in self._nodes:
            self.delete(item.id)

        self._nodes[item.id] = item
        node_level = self._random_level()
        self._levels[item.id] = node_level

        # First node in the index
        if self._enter_point is None:
            self._enter_point = item.id
            self._max_level = node_level
            for lc in range(node_level + 1):
                self._graph[lc][item.id] = []
            return

        curr_ep = self._enter_point
        max_l = self._max_level
        q_vec = item.vector

        # Phase 1: Greedily traverse from top layer down to node_level + 1
        for lc in range(max_l, node_level, -1):
            changed = True
            curr_dist = self._dist(q_vec, self._nodes[curr_ep].vector)
            while changed:
                changed = False
                for nb_id in self._graph[lc].get(curr_ep, []):
                    if nb_id not in self._nodes:
                        continue
                    nb_dist = self._dist(q_vec, self._nodes[nb_id].vector)
                    if nb_dist < curr_dist:
                        curr_dist = nb_dist
                        curr_ep = nb_id
                        changed = True

        # Phase 2: From min(max_l, node_level) down to 0, insert and connect
        top_insert_level = min(max_l, node_level)
        enter_points = [curr_ep]

        for lc in range(top_insert_level, -1, -1):
            ef_con = max(self.config.ef_construction, self.config.m)
            candidates = self._search_layer(q_vec, enter_points, ef=ef_con, layer=lc)

            m_cur = self.config.m0 if lc == 0 else self.config.m
            neighbors = self._select_neighbors(q_vec, candidates, m_cur, lc)

            self._graph[lc][item.id] = list(neighbors)

            # Add reverse connections
            for nb_id in neighbors:
                if nb_id not in self._graph[lc]:
                    self._graph[lc][nb_id] = []
                self._graph[lc][nb_id].append(item.id)

                # Shrink neighbor connections if exceeding m_cur
                if len(self._graph[lc][nb_id]) > m_cur:
                    nb_candidates = [
                        (self._dist(self._nodes[nb_id].vector, self._nodes[n_id].vector), n_id)
                        for n_id in self._graph[lc][nb_id]
                        if n_id in self._nodes
                    ]
                    nb_candidates.sort(key=lambda x: x[0])
                    new_nb = self._select_neighbors(
                        self._nodes[nb_id].vector, nb_candidates, m_cur, lc
                    )
                    self._graph[lc][nb_id] = new_nb

            enter_points = [c[1] for c in candidates[:self.config.m]]

        # Ensure graph dict entry exists for all levels up to node_level
        for lc in range(top_insert_level + 1, node_level + 1):
            if item.id not in self._graph[lc]:
                self._graph[lc][item.id] = []

        # Update entry point if new node level is strictly higher
        if node_level > self._max_level:
            self._max_level = node_level
            self._enter_point = item.id

    def add_batch(self, items: Sequence[VectorItem]) -> None:
        """Add multiple items into the HNSW index."""
        for item in items:
            self.add(item)

    def delete(self, item_id: str) -> bool:
        """Delete an item by ID and repair graph connectivity."""
        if item_id not in self._nodes:
            return False

        item_level = self._levels.get(item_id, 0)

        # Remove from graph layers and disconnect from neighbors
        for lc in range(item_level + 1):
            if lc < len(self._graph):
                neighbors = self._graph[lc].pop(item_id, [])
                for nb_id in neighbors:
                    if nb_id in self._graph[lc] and item_id in self._graph[lc][nb_id]:
                        self._graph[lc][nb_id].remove(item_id)

        del self._nodes[item_id]
        if item_id in self._levels:
            del self._levels[item_id]

        # Repair enter point if deleted node was the enter point
        if self._enter_point == item_id:
            self._enter_point = None
            self._max_level = -1

            # Find new enter point from highest populated level
            for lc in range(len(self._graph) - 1, -1, -1):
                if self._graph[lc]:
                    self._enter_point = next(iter(self._graph[lc].keys()))
                    self._max_level = lc
                    break

        return True

    def search(
        self,
        query_vector: Sequence[float],
        k: int = 10,
        filter_expr: Optional[Dict[str, Any]] = None,
        include_vector: bool = False,
        ef_search: Optional[int] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """Search HNSW graph for top k nearest neighbors matching optional filter."""
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} does not match index dimension {self.dimension}"
            )
        if k <= 0 or not self._nodes or self._enter_point is None:
            return []

        curr_ep = self._enter_point
        max_l = self._max_level

        # Phase 1: Greedily navigate down from top layer to layer 1
        for lc in range(max_l, 0, -1):
            changed = True
            curr_dist = self._dist(query_vector, self._nodes[curr_ep].vector)
            while changed:
                changed = False
                for nb_id in self._graph[lc].get(curr_ep, []):
                    if nb_id not in self._nodes:
                        continue
                    nb_dist = self._dist(query_vector, self._nodes[nb_id].vector)
                    if nb_dist < curr_dist:
                        curr_dist = nb_dist
                        curr_ep = nb_id
                        changed = True

        # Phase 2: Beam search on layer 0 with ef_search
        ef = max(k, ef_search or self.config.ef_search)
        candidates = self._search_layer(query_vector, [curr_ep], ef=ef, layer=0)

        # Filter and construct results
        results: List[SearchResult] = []
        for dist, node_id in candidates:
            if node_id not in self._nodes:
                continue
            item = self._nodes[node_id]

            if filter_expr and not evaluate_filter(item.metadata, filter_expr):
                continue

            score = distance_to_score(dist, self.metric)
            results.append(
                SearchResult(
                    id=item.id,
                    score=score,
                    distance=dist,
                    vector=item.vector if include_vector else None,
                    metadata=item.metadata,
                    document=item.document,
                )
            )
            if len(results) >= k and not filter_expr:
                break

        # If filtering reduced candidate count below k, fallback scan remaining filtered nodes
        if filter_expr and len(results) < k and len(self._nodes) > len(candidates):
            seen_ids = {r.id for r in results}
            for node_id, item in self._nodes.items():
                if node_id in seen_ids:
                    continue
                if evaluate_filter(item.metadata, filter_expr):
                    dist = self._dist(query_vector, item.vector)
                    score = distance_to_score(dist, self.metric)
                    results.append(
                        SearchResult(
                            id=item.id,
                            score=score,
                            distance=dist,
                            vector=item.vector if include_vector else None,
                            metadata=item.metadata,
                            document=item.document,
                        )
                    )

        results.sort(key=lambda r: (r.distance, -r.score))
        return results[:k]

    def get_stats(self) -> Dict[str, Any]:
        """Return graph index statistics."""
        num_nodes = len(self._nodes)
        total_links = 0
        layer_counts = {}

        for lc, layer_dict in enumerate(self._graph):
            layer_counts[f"layer_{lc}"] = len(layer_dict)
            for node_id, neighbors in layer_dict.items():
                total_links += len(neighbors)

        avg_links = total_links / max(num_nodes, 1)

        # Estimate memory
        approx_bytes = sys.getsizeof(self._nodes) + sys.getsizeof(self._levels)
        for item in self._nodes.values():
            approx_bytes += sys.getsizeof(item.id)
            approx_bytes += sys.getsizeof(item.vector) + len(item.vector) * 8
            approx_bytes += sys.getsizeof(item.metadata)
            if item.document:
                approx_bytes += sys.getsizeof(item.document)
        for layer_dict in self._graph:
            approx_bytes += sys.getsizeof(layer_dict)
            for n_id, n_list in layer_dict.items():
                approx_bytes += sys.getsizeof(n_id) + sys.getsizeof(n_list) + len(n_list) * 8

        return {
            "index_type": IndexType.HNSW.value,
            "count": num_nodes,
            "dimension": self.dimension,
            "metric": self.metric.value,
            "layers_count": max(0, self._max_level + 1),
            "avg_links_per_node": round(avg_links, 2),
            "memory_bytes": approx_bytes,
            "enter_point": self._enter_point,
            "max_level": self._max_level,
            "layer_counts": layer_counts,
            "hnsw_config": self.config.to_dict(),
        }
