"""Index implementations and metadata filter evaluation engine."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from vector_search_engine.models import (
    DistanceMetric,
    IndexType,
    SearchResult,
    VectorItem,
)


def evaluate_filter(
    metadata: Optional[Dict[str, Any]],
    filter_expr: Optional[Dict[str, Any]],
) -> bool:
    """Evaluate whether a metadata dictionary satisfies a filter expression.

    Supports exact matching, comparison operators ($eq, $ne, $gt, $gte, $lt, $lte),
    membership operators ($in, $nin), array operators ($contains, $all),
    string operators ($regex, $exists), and logical operators ($and, $or, $not, $nor).
    """
    if not filter_expr:
        return True

    meta = metadata or {}

    for key, condition in filter_expr.items():
        if key == "$and":
            if not isinstance(condition, list):
                return False
            if not all(evaluate_filter(meta, sub_cond) for sub_cond in condition):
                return False
        elif key == "$or":
            if not isinstance(condition, list):
                return False
            if not any(evaluate_filter(meta, sub_cond) for sub_cond in condition):
                return False
        elif key == "$not":
            if not isinstance(condition, dict):
                return False
            if evaluate_filter(meta, condition):
                return False
        elif key == "$nor":
            if not isinstance(condition, list):
                return False
            if any(evaluate_filter(meta, sub_cond) for sub_cond in condition):
                return False
        else:
            # Field-level condition
            field_val = meta.get(key)
            if not _evaluate_field_condition(field_val, condition, key in meta):
                return False

    return True


def _evaluate_field_condition(
    field_val: Any,
    condition: Any,
    field_exists: bool,
) -> bool:
    """Evaluate condition on a specific field value."""
    if isinstance(condition, dict):
        # Operators dictionary
        for op, op_val in condition.items():
            if op == "$eq":
                if field_val != op_val:
                    return False
            elif op == "$ne":
                if field_val == op_val:
                    return False
            elif op == "$gt":
                if field_val is None or field_val <= op_val:
                    return False
            elif op == "$gte":
                if field_val is None or field_val < op_val:
                    return False
            elif op == "$lt":
                if field_val is None or field_val >= op_val:
                    return False
            elif op == "$lte":
                if field_val is None or field_val > op_val:
                    return False
            elif op == "$in":
                if not isinstance(op_val, (list, tuple, set)):
                    return False
                if field_val not in op_val:
                    return False
            elif op == "$nin":
                if not isinstance(op_val, (list, tuple, set)):
                    return False
                if field_val in op_val:
                    return False
            elif op == "$exists":
                if bool(op_val) != field_exists:
                    return False
            elif op == "$contains":
                if field_val is None:
                    return False
                if isinstance(field_val, (list, tuple, set, str)):
                    if op_val not in field_val:
                        return False
                else:
                    return False
            elif op == "$all":
                if not isinstance(op_val, (list, tuple, set)):
                    return False
                if not isinstance(field_val, (list, tuple, set)):
                    return False
                val_set = set(field_val)
                if not all(item in val_set for item in op_val):
                    return False
            elif op == "$regex":
                if field_val is None or not isinstance(field_val, str):
                    return False
                try:
                    if not re.search(str(op_val), field_val):
                        return False
                except re.error:
                    return False
            else:
                # Unknown operator, treat as field mismatch
                return False
        return True
    else:
        # Direct equality match
        return field_val == condition


class BaseIndex(ABC):
    """Abstract base class for all vector index implementations."""

    @abstractmethod
    def add(self, item: VectorItem) -> None:
        """Add a single item to the index."""
        pass

    @abstractmethod
    def add_batch(self, items: Sequence[VectorItem]) -> None:
        """Add multiple items to the index."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        k: int = 10,
        filter_expr: Optional[Dict[str, Any]] = None,
        include_vector: bool = False,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """Search for top k nearest neighbors matching optional filter."""
        pass

    @abstractmethod
    def delete(self, item_id: str) -> bool:
        """Delete an item by ID from the index."""
        pass

    @abstractmethod
    def get(self, item_id: str) -> Optional[VectorItem]:
        """Get an item by ID."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return total number of items in the index."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all items from the index."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return index-specific statistics."""
        pass


from vector_search_engine.indexes.flat_index import FlatIndex
from vector_search_engine.indexes.hnsw_index import HNSWIndex
from vector_search_engine.indexes.ivf_index import IVFIndex

__all__ = [
    "BaseIndex",
    "FlatIndex",
    "HNSWIndex",
    "IVFIndex",
    "evaluate_filter",
]
