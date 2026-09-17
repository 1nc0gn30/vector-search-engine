"""Exact brute-force flat linear scan vector index.

Provides 100% recall baseline for all supported distance metrics and
evaluates complex metadata filtering expressions.
"""

from __future__ import annotations

import heapq
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

from vector_search_engine.models import (
    DistanceMetric,
    IndexType,
    SearchResult,
    VectorItem,
)
from vector_search_engine.metrics import (
    compute_distance,
    distance_to_score,
)
from vector_search_engine.indexes import BaseIndex, evaluate_filter


class FlatIndex(BaseIndex):
    """Brute-force exact search index over vector items."""

    def __init__(
        self,
        dimension: int,
        metric: Union[DistanceMetric, str] = DistanceMetric.COSINE,
    ) -> None:
        self.dimension = int(dimension)
        self.metric = DistanceMetric.from_str(metric)
        self._items: Dict[str, VectorItem] = {}

    def add(self, item: VectorItem) -> None:
        """Add or update an item in the index."""
        if len(item.vector) != self.dimension:
            raise ValueError(
                f"Item vector dimension {len(item.vector)} does not match index dimension {self.dimension}"
            )
        self._items[item.id] = item

    def add_batch(self, items: Sequence[VectorItem]) -> None:
        """Add multiple items to the index."""
        for item in items:
            self.add(item)

    def delete(self, item_id: str) -> bool:
        """Delete an item by ID from the index."""
        if item_id in self._items:
            del self._items[item_id]
            return True
        return False

    def get(self, item_id: str) -> Optional[VectorItem]:
        """Get an item by ID."""
        return self._items.get(item_id)

    def size(self) -> int:
        """Return total number of items."""
        return len(self._items)

    def clear(self) -> None:
        """Clear all items."""
        self._items.clear()

    def search(
        self,
        query_vector: Sequence[float],
        k: int = 10,
        filter_expr: Optional[Dict[str, Any]] = None,
        include_vector: bool = False,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """Exhaustively scan all indexed vectors, filter matching items, and return top k."""
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} does not match index dimension {self.dimension}"
            )
        if k <= 0:
            return []

        candidates: List[SearchResult] = []

        for item_id, item in self._items.items():
            if filter_expr and not evaluate_filter(item.metadata, filter_expr):
                continue

            dist = compute_distance(query_vector, item.vector, self.metric)
            score = distance_to_score(dist, self.metric)

            result = SearchResult(
                id=item.id,
                score=score,
                distance=dist,
                vector=item.vector if include_vector else None,
                metadata=item.metadata,
                document=item.document,
            )
            candidates.append(result)

        # Sort candidates: primary key distance ascending, secondary score descending
        candidates.sort(key=lambda r: (r.distance, -r.score))
        return candidates[:k]

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        # Estimate memory usage
        approx_bytes = sys.getsizeof(self._items)
        for item in self._items.values():
            approx_bytes += sys.getsizeof(item.id)
            approx_bytes += sys.getsizeof(item.vector) + len(item.vector) * 8
            approx_bytes += sys.getsizeof(item.metadata)
            if item.document:
                approx_bytes += sys.getsizeof(item.document)

        return {
            "index_type": IndexType.FLAT.value,
            "count": len(self._items),
            "dimension": self.dimension,
            "metric": self.metric.value,
            "memory_bytes": approx_bytes,
        }
