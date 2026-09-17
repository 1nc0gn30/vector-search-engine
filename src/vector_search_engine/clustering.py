"""Vector Clustering and Partitioning Engine.

Provides K-Means++ clustering, Hierarchical Vector Partitioning,
Medoid extraction, and Silhouette Cohesion Analysis for vector collections.
Pure Python, zero-dependency.
"""

from __future__ import annotations

import collections
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from vector_search_engine.metrics import (
    compute_centroid,
    compute_distance,
    normalize_vector,
)
from vector_search_engine.models import DistanceMetric, VectorItem


@dataclass
class ClusterInfo:
    """Detailed summary of an individual vector cluster."""

    cluster_id: int
    centroid: List[float]
    medoid_id: str
    member_ids: List[str]
    size: int
    dispersion: float = 0.0
    diameter: float = 0.0
    top_metadata: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "centroid": [round(x, 6) for x in self.centroid],
            "medoid_id": self.medoid_id,
            "member_ids": self.member_ids,
            "size": self.size,
            "dispersion": round(self.dispersion, 6),
            "diameter": round(self.diameter, 6),
            "top_metadata": self.top_metadata,
        }


@dataclass
class ClusterAnalysisResult:
    """Comprehensive clustering and cohesion report."""

    method: str
    k: int
    metric: str
    clusters: List[ClusterInfo]
    silhouette_score: float
    inertia: float
    iterations: int
    converged: bool
    duration_ms: float
    item_cluster_map: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "k": self.k,
            "metric": self.metric,
            "clusters": [c.to_dict() for c in self.clusters],
            "silhouette_score": round(self.silhouette_score, 4),
            "inertia": round(self.inertia, 4),
            "iterations": self.iterations,
            "converged": self.converged,
            "duration_ms": round(self.duration_ms, 2),
            "item_cluster_map": self.item_cluster_map,
        }


def kmeans_plus_plus_init(
    vectors: Sequence[Sequence[float]],
    k: int,
    metric: DistanceMetric,
    rng: random.Random,
) -> List[List[float]]:
    """K-Means++ initialization algorithm for optimal cluster seeding."""
    n = len(vectors)
    if n == 0 or k <= 0:
        return []

    # First centroid chosen uniformly at random
    first_idx = rng.randrange(n)
    centroids: List[List[float]] = [list(vectors[first_idx])]

    for _ in range(1, k):
        # Compute squared distance of each point to its nearest centroid
        distances_sq: List[float] = []
        for vec in vectors:
            min_dist = float("inf")
            for c in centroids:
                d = compute_distance(vec, c, metric)
                if d < min_dist:
                    min_dist = d
            distances_sq.append(min_dist * min_dist)

        total_weight = sum(distances_sq)
        if total_weight <= 1e-12:
            # All remaining points identical or close to existing centroids
            remaining = [i for i in range(n) if list(vectors[i]) not in centroids]
            if remaining:
                chosen_idx = rng.choice(remaining)
                centroids.append(list(vectors[chosen_idx]))
            else:
                centroids.append(list(vectors[rng.randrange(n)]))
            continue

        # Roulette wheel selection based on D(x)^2
        target = rng.random() * total_weight
        cumulative = 0.0
        chosen_idx = n - 1
        for idx, dist_sq in enumerate(distances_sq):
            cumulative += dist_sq
            if cumulative >= target:
                chosen_idx = idx
                break
        centroids.append(list(vectors[chosen_idx]))

    return centroids


class SilhouetteAnalyzer:
    """Evaluates cluster separation and cohesion."""

    @staticmethod
    def compute(
        items: Sequence[VectorItem],
        assignments: Dict[str, int],
        metric: DistanceMetric = DistanceMetric.COSINE,
        max_samples: int = 500,
        seed: int = 42,
    ) -> float:
        """Compute average silhouette coefficient across clustered items in range [-1.0, 1.0]."""
        if len(items) <= 1:
            return 0.0

        unique_clusters = set(assignments.values())
        if len(unique_clusters) <= 1:
            return 0.0

        # Subsample if dataset is large to prevent O(N^2) latency
        sample_items = list(items)
        if len(sample_items) > max_samples:
            rng = random.Random(seed)
            sample_items = rng.sample(sample_items, max_samples)

        # Group items by cluster
        cluster_items: Dict[int, List[VectorItem]] = collections.defaultdict(list)
        for item in items:
            c_id = assignments.get(item.id)
            if c_id is not None:
                cluster_items[c_id].append(item)

        scores: List[float] = []

        for item in sample_items:
            c_id = assignments.get(item.id)
            if c_id is None:
                continue

            own_cluster = cluster_items[c_id]
            if len(own_cluster) <= 1:
                # Single-element cluster has silhouette = 0 by convention
                scores.append(0.0)
                continue

            # a(i): average distance to points in same cluster
            dist_own_sum = sum(
                compute_distance(item.vector, other.vector, metric)
                for other in own_cluster
                if other.id != item.id
            )
            a_i = dist_own_sum / (len(own_cluster) - 1)

            # b(i): minimum average distance to points in any other cluster
            b_i = float("inf")
            for other_cid, other_items in cluster_items.items():
                if other_cid == c_id or not other_items:
                    continue
                dist_other_sum = sum(
                    compute_distance(item.vector, other.vector, metric)
                    for other in other_items
                )
                mean_dist = dist_other_sum / len(other_items)
                if mean_dist < b_i:
                    b_i = mean_dist

            denom = max(a_i, b_i)
            if denom <= 1e-12:
                s_i = 0.0
            else:
                s_i = (b_i - a_i) / denom
            scores.append(s_i)

        return sum(scores) / len(scores) if scores else 0.0


class KMeansClusterer:
    """K-Means vector clustering with K-Means++ initialization and medoid extraction."""

    def __init__(
        self,
        k: int = 3,
        max_iter: int = 50,
        metric: Union[DistanceMetric, str] = DistanceMetric.COSINE,
        tolerance: float = 1e-4,
        seed: Optional[int] = 42,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.k = k
        self.max_iter = max_iter
        if isinstance(metric, str):
            metric = DistanceMetric.from_str(metric)
        self.metric = metric
        self.tolerance = tolerance
        self.seed = seed

    def fit(self, items: Sequence[VectorItem]) -> ClusterAnalysisResult:
        """Run K-Means clustering on the provided items."""
        start_time = time.perf_counter()
        n = len(items)
        if n == 0:
            return ClusterAnalysisResult(
                method="kmeans",
                k=self.k,
                metric=self.metric.value,
                clusters=[],
                silhouette_score=0.0,
                inertia=0.0,
                iterations=0,
                converged=True,
                duration_ms=(time.perf_counter() - start_time) * 1000.0,
                item_cluster_map={},
            )

        actual_k = min(self.k, n)
        rng = random.Random(self.seed)
        vectors = [it.vector for it in items]
        dim = len(vectors[0])

        # Step 1: K-Means++ init
        centroids = kmeans_plus_plus_init(vectors, actual_k, self.metric, rng)

        converged = False
        iteration = 0
        assignments: Dict[str, int] = {}
        cluster_groups: Dict[int, List[VectorItem]] = {i: [] for i in range(actual_k)}

        for it_count in range(self.max_iter):
            iteration = it_count + 1
            new_groups: Dict[int, List[VectorItem]] = {i: [] for i in range(actual_k)}
            new_assignments: Dict[str, int] = {}

            # Assignment step
            for item in items:
                best_c = 0
                best_d = float("inf")
                for c_idx, c_vec in enumerate(centroids):
                    d = compute_distance(item.vector, c_vec, self.metric)
                    if d < best_d:
                        best_d = d
                        best_c = c_idx
                new_groups[best_c].append(item)
                new_assignments[item.id] = best_c

            # Update centroids step
            max_shift = 0.0
            new_centroids: List[List[float]] = []

            for c_idx in range(actual_k):
                group = new_groups[c_idx]
                if not group:
                    # Handle empty cluster: reseed with random item
                    fallback_item = rng.choice(items)
                    new_centroids.append(list(fallback_item.vector))
                    continue

                group_vectors = [x.vector for x in group]
                updated = compute_centroid(group_vectors)
                if self.metric == DistanceMetric.COSINE:
                    updated = normalize_vector(updated)

                shift = compute_distance(centroids[c_idx], updated, self.metric)
                if shift > max_shift:
                    max_shift = shift
                new_centroids.append(updated)

            centroids = new_centroids
            cluster_groups = new_groups
            assignments = new_assignments

            if max_shift < self.tolerance:
                converged = True
                break

        # Compute inertia (sum of squared distances to centroid)
        inertia = 0.0
        for c_idx, group in cluster_groups.items():
            c_vec = centroids[c_idx]
            for item in group:
                d = compute_distance(item.vector, c_vec, self.metric)
                inertia += d * d

        # Build ClusterInfo for each cluster
        cluster_infos: List[ClusterInfo] = []
        for c_idx in range(actual_k):
            group = cluster_groups[c_idx]
            c_vec = centroids[c_idx]
            member_ids = [it.id for it in group]
            size = len(group)

            if not group:
                cluster_infos.append(
                    ClusterInfo(
                        cluster_id=c_idx,
                        centroid=c_vec,
                        medoid_id="",
                        member_ids=[],
                        size=0,
                        dispersion=0.0,
                        diameter=0.0,
                        top_metadata={},
                    )
                )
                continue

            # Find medoid: item in cluster with minimum distance to centroid
            best_medoid_id = group[0].id
            min_medoid_d = float("inf")
            dispersion_sum = 0.0
            for it in group:
                d = compute_distance(it.vector, c_vec, self.metric)
                dispersion_sum += d
                if d < min_medoid_d:
                    min_medoid_d = d
                    best_medoid_id = it.id

            dispersion = dispersion_sum / size

            # Cluster diameter: max pairwise distance between items
            # Subsample for diameter if cluster is large
            diameter = 0.0
            eval_members = group if len(group) <= 50 else rng.sample(group, 50)
            for i in range(len(eval_members)):
                for j in range(i + 1, len(eval_members)):
                    pair_d = compute_distance(
                        eval_members[i].vector, eval_members[j].vector, self.metric
                    )
                    if pair_d > diameter:
                        diameter = pair_d

            # Extract top metadata frequencies
            metadata_counts: Dict[str, collections.Counter] = collections.defaultdict(
                collections.Counter
            )
            for it in group:
                for k_meta, v_meta in it.metadata.items():
                    if isinstance(v_meta, (str, int, float, bool)):
                        metadata_counts[k_meta][str(v_meta)] += 1

            top_meta: Dict[str, Dict[str, int]] = {}
            for k_meta, counter in metadata_counts.items():
                top_meta[k_meta] = dict(counter.most_common(5))

            cluster_infos.append(
                ClusterInfo(
                    cluster_id=c_idx,
                    centroid=c_vec,
                    medoid_id=best_medoid_id,
                    member_ids=member_ids,
                    size=size,
                    dispersion=dispersion,
                    diameter=diameter,
                    top_metadata=top_meta,
                )
            )

        # Silhouette score
        sil_score = SilhouetteAnalyzer.compute(
            items=items,
            assignments=assignments,
            metric=self.metric,
            max_samples=min(500, n),
            seed=self.seed or 42,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return ClusterAnalysisResult(
            method="kmeans",
            k=actual_k,
            metric=self.metric.value,
            clusters=cluster_infos,
            silhouette_score=sil_score,
            inertia=inertia,
            iterations=iteration,
            converged=converged,
            duration_ms=duration_ms,
            item_cluster_map=assignments,
        )


class HierarchicalKMeansClusterer:
    """Recursive top-down hierarchical vector clustering tree."""

    def __init__(
        self,
        branch_factor: int = 2,
        max_depth: int = 3,
        min_leaf_size: int = 4,
        metric: Union[DistanceMetric, str] = DistanceMetric.COSINE,
        seed: Optional[int] = 42,
    ) -> None:
        self.branch_factor = max(2, branch_factor)
        self.max_depth = max(1, max_depth)
        self.min_leaf_size = max(1, min_leaf_size)
        if isinstance(metric, str):
            metric = DistanceMetric.from_str(metric)
        self.metric = metric
        self.seed = seed

    def fit(self, items: Sequence[VectorItem]) -> ClusterAnalysisResult:
        """Run top-down recursive clustering to partition the vector collection."""
        start_time = time.perf_counter()
        n = len(items)
        if n == 0:
            return ClusterAnalysisResult(
                method="hierarchical",
                k=0,
                metric=self.metric.value,
                clusters=[],
                silhouette_score=0.0,
                inertia=0.0,
                iterations=0,
                converged=True,
                duration_ms=(time.perf_counter() - start_time) * 1000.0,
                item_cluster_map={},
            )

        leaf_clusters: List[List[VectorItem]] = []
        self._partition_recursive(list(items), depth=0, leaves=leaf_clusters)

        # Build ClusterInfo for each leaf partition
        cluster_infos: List[ClusterInfo] = []
        assignments: Dict[str, int] = {}
        total_inertia = 0.0

        for c_idx, group in enumerate(leaf_clusters):
            size = len(group)
            group_vectors = [x.vector for x in group]
            centroid = compute_centroid(group_vectors)
            if self.metric == DistanceMetric.COSINE:
                centroid = normalize_vector(centroid)

            best_medoid = group[0].id
            min_med_d = float("inf")
            disp_sum = 0.0

            for it in group:
                assignments[it.id] = c_idx
                d = compute_distance(it.vector, centroid, self.metric)
                total_inertia += d * d
                disp_sum += d
                if d < min_med_d:
                    min_med_d = d
                    best_medoid = it.id

            dispersion = disp_sum / size if size > 0 else 0.0
            diameter = 0.0
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    d = compute_distance(group[i].vector, group[j].vector, self.metric)
                    if d > diameter:
                        diameter = d

            meta_counts: Dict[str, collections.Counter] = collections.defaultdict(
                collections.Counter
            )
            for it in group:
                for k_meta, v_meta in it.metadata.items():
                    if isinstance(v_meta, (str, int, float, bool)):
                        meta_counts[k_meta][str(v_meta)] += 1

            top_meta = {k: dict(c.most_common(5)) for k, c in meta_counts.items()}

            cluster_infos.append(
                ClusterInfo(
                    cluster_id=c_idx,
                    centroid=centroid,
                    medoid_id=best_medoid,
                    member_ids=[x.id for x in group],
                    size=size,
                    dispersion=dispersion,
                    diameter=diameter,
                    top_metadata=top_meta,
                )
            )

        sil_score = SilhouetteAnalyzer.compute(
            items=items,
            assignments=assignments,
            metric=self.metric,
            max_samples=min(500, n),
            seed=self.seed or 42,
        )

        return ClusterAnalysisResult(
            method="hierarchical",
            k=len(leaf_clusters),
            metric=self.metric.value,
            clusters=cluster_infos,
            silhouette_score=sil_score,
            inertia=total_inertia,
            iterations=self.max_depth,
            converged=True,
            duration_ms=(time.perf_counter() - start_time) * 1000.0,
            item_cluster_map=assignments,
        )

    def _partition_recursive(
        self,
        items: List[VectorItem],
        depth: int,
        leaves: List[List[VectorItem]],
    ) -> None:
        """Recursively split items until max_depth or min_leaf_size reached."""
        if depth >= self.max_depth or len(items) <= self.min_leaf_size:
            leaves.append(items)
            return

        branch = min(self.branch_factor, len(items))
        if branch <= 1:
            leaves.append(items)
            return

        # Split current group with KMeans(k=branch)
        clusterer = KMeansClusterer(
            k=branch,
            max_iter=15,
            metric=self.metric,
            seed=(self.seed or 42) + depth * 31,
        )
        res = clusterer.fit(items)

        groups: Dict[int, List[VectorItem]] = {i: [] for i in range(branch)}
        for it in items:
            c_id = res.item_cluster_map.get(it.id, 0)
            groups[c_id].append(it)

        # If splitting failed to divide points
        non_empty = [g for g in groups.values() if g]
        if len(non_empty) <= 1:
            leaves.append(items)
            return

        for g in non_empty:
            self._partition_recursive(g, depth + 1, leaves)
