"""Inverted File (IVF) vector index with pure Python k-means clustering.

Partitions vector space into Voronoi cells (inverted lists) around centroids
and queries only the top nprobe nearest posting lists for fast approximate search.
"""

from __future__ import annotations

import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from vector_search_engine.models import (
    DistanceMetric,
    IndexType,
    IVFConfig,
    SearchResult,
    VectorItem,
)
from vector_search_engine.metrics import (
    compute_centroid,
    compute_distance,
    distance_to_score,
)
from vector_search_engine.indexes import BaseIndex, evaluate_filter


class IVFIndex(BaseIndex):
    """Inverted File (IVF) Index using k-means centroid partitioning."""

    def __init__(
        self,
        dimension: int,
        metric: Union[DistanceMetric, str] = DistanceMetric.COSINE,
        config: Optional[IVFConfig] = None,
    ) -> None:
        self.dimension = int(dimension)
        self.metric = DistanceMetric.from_str(metric)
        self.config = config or IVFConfig()
        self._rng = random.Random(self.config.seed)

        self._items: Dict[str, VectorItem] = {}
        self._item_to_centroid: Dict[str, int] = {}
        # Inverted lists: centroid_idx -> list of item_ids
        self._inverted_lists: Dict[int, List[str]] = {}
        self._centroids: List[List[float]] = []
        self._is_trained: bool = False

    def size(self) -> int:
        """Return total number of items."""
        return len(self._items)

    def get(self, item_id: str) -> Optional[VectorItem]:
        """Get an item by ID."""
        return self._items.get(item_id)

    def clear(self) -> None:
        """Clear all items and centroids."""
        self._items.clear()
        self._item_to_centroid.clear()
        self._inverted_lists.clear()
        self._centroids.clear()
        self._is_trained = False

    def _train_kmeans(self, vectors: List[List[float]]) -> None:
        """Train k-means centroids on current vectors using Lloyd's algorithm."""
        k = min(self.config.nlist, len(vectors))
        if k <= 0:
            return

        # Initialize centroids using k-means++
        centroids: List[List[float]] = [list(self._rng.choice(vectors))]
        for _ in range(1, k):
            # Compute squared distances to nearest existing centroid
            dist_sqs: List[float] = []
            for vec in vectors:
                min_d = min(
                    compute_distance(vec, c, DistanceMetric.EUCLIDEAN)
                    for c in centroids
                )
                dist_sqs.append(min_d * min_d)

            total_dist = sum(dist_sqs)
            if total_dist <= 0:
                # Random choice fallback if all points identical
                centroids.append(list(self._rng.choice(vectors)))
                continue

            r = self._rng.random() * total_dist
            accum = 0.0
            chosen_idx = len(vectors) - 1
            for idx, d_sq in enumerate(dist_sqs):
                accum += d_sq
                if accum >= r:
                    chosen_idx = idx
                    break
            centroids.append(list(vectors[chosen_idx]))

        # Iterate Lloyd's k-means
        for _ in range(self.config.max_iterations):
            clusters: List[List[List[float]]] = [[] for _ in range(k)]
            for vec in vectors:
                best_c = 0
                best_dist = float("inf")
                for c_idx, c in enumerate(centroids):
                    d = compute_distance(vec, c, self.metric)
                    if d < best_dist:
                        best_dist = d
                        best_c = c_idx
                clusters[best_c].append(vec)

            # Recompute centroids
            new_centroids: List[List[float]] = []
            for c_idx in range(k):
                if clusters[c_idx]:
                    new_centroids.append(compute_centroid(clusters[c_idx]))
                else:
                    new_centroids.append(centroids[c_idx])

            centroids = new_centroids

        self._centroids = centroids
        self._inverted_lists = {i: [] for i in range(len(self._centroids))}
        self._is_trained = True

        # Reassign all existing items
        self._item_to_centroid.clear()
        for item_id, item in self._items.items():
            best_c = self._find_nearest_centroid(item.vector)
            self._inverted_lists[best_c].append(item_id)
            self._item_to_centroid[item_id] = best_c

    def _find_nearest_centroid(self, vector: Sequence[float]) -> int:
        """Find the index of the nearest centroid to a vector."""
        if not self._centroids:
            return 0
        best_idx = 0
        best_dist = float("inf")
        for i, c in enumerate(self._centroids):
            d = compute_distance(vector, c, self.metric)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def train(self) -> None:
        """Explicitly train centroids on all stored vectors."""
        if len(self._items) >= max(2, self.config.nlist):
            vectors = [item.vector for item in self._items.values()]
            self._train_kmeans(vectors)

    def add(self, item: VectorItem) -> None:
        """Add an item into the IVF index."""
        if len(item.vector) != self.dimension:
            raise ValueError(
                f"Item vector dimension {len(item.vector)} does not match index dimension {self.dimension}"
            )

        if item.id in self._items:
            self.delete(item.id)

        self._items[item.id] = item

        # If trained, assign to nearest centroid
        if self._is_trained and self._centroids:
            c_idx = self._find_nearest_centroid(item.vector)
            self._inverted_lists[c_idx].append(item.id)
            self._item_to_centroid[item.id] = c_idx
        elif len(self._items) >= self.config.nlist * 2:
            # Auto-train when enough items accumulate
            self.train()

    def add_batch(self, items: Sequence[VectorItem]) -> None:
        """Add multiple items into the IVF index."""
        for item in items:
            self.add(item)
        if not self._is_trained and len(self._items) >= max(4, self.config.nlist):
            self.train()

    def delete(self, item_id: str) -> bool:
        """Delete an item by ID from IVF index."""
        if item_id not in self._items:
            return False

        if item_id in self._item_to_centroid:
            c_idx = self._item_to_centroid.pop(item_id)
            if c_idx in self._inverted_lists and item_id in self._inverted_lists[c_idx]:
                self._inverted_lists[c_idx].remove(item_id)

        del self._items[item_id]
        return True

    def search(
        self,
        query_vector: Sequence[float],
        k: int = 10,
        filter_expr: Optional[Dict[str, Any]] = None,
        include_vector: bool = False,
        nprobe: Optional[int] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """Search IVF inverted lists for top k nearest neighbors."""
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} does not match index dimension {self.dimension}"
            )
        if k <= 0 or not self._items:
            return []

        # If not trained or very few centroids, fallback to flat scan
        if not self._is_trained or not self._centroids:
            candidates: List[SearchResult] = []
            for item in self._items.values():
                if filter_expr and not evaluate_filter(item.metadata, filter_expr):
                    continue
                dist = compute_distance(query_vector, item.vector, self.metric)
                score = distance_to_score(dist, self.metric)
                candidates.append(
                    SearchResult(
                        id=item.id,
                        score=score,
                        distance=dist,
                        vector=item.vector if include_vector else None,
                        metadata=item.metadata,
                        document=item.document,
                    )
                )
            candidates.sort(key=lambda r: (r.distance, -r.score))
            return candidates[:k]

        # Find top nprobe centroids
        actual_nprobe = min(nprobe or self.config.nprobe, len(self._centroids))
        centroid_dists: List[Tuple[float, int]] = []
        for c_idx, c in enumerate(self._centroids):
            d = compute_distance(query_vector, c, self.metric)
            centroid_dists.append((d, c_idx))

        centroid_dists.sort(key=lambda x: x[0])
        probed_clusters = [c_idx for _, c_idx in centroid_dists[:actual_nprobe]]

        # Gather candidates from probed clusters
        results: List[SearchResult] = []
        seen_ids: Set[str] = set()

        for c_idx in probed_clusters:
            for item_id in self._inverted_lists.get(c_idx, []):
                if item_id in seen_ids or item_id not in self._items:
                    continue
                seen_ids.add(item_id)
                item = self._items[item_id]

                if filter_expr and not evaluate_filter(item.metadata, filter_expr):
                    continue

                dist = compute_distance(query_vector, item.vector, self.metric)
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

        # Fallback if filtered results are fewer than k
        if filter_expr and len(results) < k and len(self._items) > len(seen_ids):
            for item_id, item in self._items.items():
                if item_id in seen_ids:
                    continue
                if evaluate_filter(item.metadata, filter_expr):
                    dist = compute_distance(query_vector, item.vector, self.metric)
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
        """Return IVF index statistics."""
        cluster_sizes = {
            f"cluster_{k}": len(v) for k, v in self._inverted_lists.items()
        }
        approx_bytes = sys.getsizeof(self._items) + sys.getsizeof(self._centroids)
        for item in self._items.values():
            approx_bytes += sys.getsizeof(item.id)
            approx_bytes += sys.getsizeof(item.vector) + len(item.vector) * 8
            approx_bytes += sys.getsizeof(item.metadata)

        return {
            "index_type": IndexType.IVF.value,
            "count": len(self._items),
            "dimension": self.dimension,
            "metric": self.metric.value,
            "is_trained": self._is_trained,
            "nlist": self.config.nlist,
            "nprobe": self.config.nprobe,
            "clusters_count": len(self._centroids),
            "cluster_sizes": cluster_sizes,
            "memory_bytes": approx_bytes,
        }
