"""Pure Python high-precision vector arithmetic and distance metrics.

Supports Cosine, Euclidean, Dot Product, Manhattan, Hamming, and Jaccard
metrics, L2 normalization, centroid computation, scalar/binary quantization,
and Johnson-Lindenstrauss random projections for visualizers.
"""

from __future__ import annotations

import math
import random
import struct
from typing import Any, List, Optional, Sequence, Set, Tuple, Union

from vector_search_engine.models import DistanceMetric


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute dot product of two equal-length numeric vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions do not match: {len(a)} != {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors in range [-1.0, 1.0].

    Returns 0.0 if either vector has zero magnitude.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions do not match: {len(a)} != {len(b)}")

    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a_sq += x * x
        norm_b_sq += y * y

    if norm_a_sq <= 0.0 or norm_b_sq <= 0.0:
        return 0.0

    denom = math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq)
    sim = dot / denom
    # Clamp to [-1.0, 1.0] to prevent floating point inaccuracies
    return max(-1.0, min(1.0, sim))


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine distance in range [0.0, 2.0].

    distance = 1.0 - cosine_similarity(a, b)
    """
    sim = cosine_similarity(a, b)
    return max(0.0, 1.0 - sim)


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute L2 Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions do not match: {len(a)} != {len(b)}")

    dist_sq = 0.0
    for x, y in zip(a, b):
        diff = x - y
        dist_sq += diff * diff
    return math.sqrt(dist_sq)


def euclidean_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Euclidean similarity in range (0.0, 1.0].

    similarity = 1.0 / (1.0 + euclidean_distance(a, b))
    """
    dist = euclidean_distance(a, b)
    return 1.0 / (1.0 + dist)


def dot_product_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute dot product distance (negative dot product).

    Used for distance minimization in index structures.
    """
    return -dot_product(a, b)


def manhattan_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute L1 Manhattan distance between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions do not match: {len(a)} != {len(b)}")
    return sum(abs(x - y) for x, y in zip(a, b))


def manhattan_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Manhattan similarity in range (0.0, 1.0].

    similarity = 1.0 / (1.0 + manhattan_distance(a, b))
    """
    dist = manhattan_distance(a, b)
    return 1.0 / (1.0 + dist)


def hamming_distance(
    a: Sequence[Union[int, float, bool]],
    b: Sequence[Union[int, float, bool]],
) -> float:
    """Compute normalized Hamming distance (proportion of differing elements).

    Returns value in range [0.0, 1.0].
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions do not match: {len(a)} != {len(b)}")
    if len(a) == 0:
        return 0.0
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    return diffs / len(a)


def hamming_similarity(
    a: Sequence[Union[int, float, bool]],
    b: Sequence[Union[int, float, bool]],
) -> float:
    """Compute Hamming similarity in range [0.0, 1.0]."""
    return 1.0 - hamming_distance(a, b)


def jaccard_similarity(
    a: Union[Sequence[float], Set[Any], Sequence[Any]],
    b: Union[Sequence[float], Set[Any], Sequence[Any]],
) -> float:
    """Compute Jaccard similarity.

    Supports both discrete sets and continuous numeric vectors (generalized Jaccard / Ruzicka similarity).
    """
    # Set-based Jaccard
    if isinstance(a, set) or isinstance(b, set):
        set_a = set(a)
        set_b = set(b)
        if not set_a and not set_b:
            return 1.0
        union_len = len(set_a | set_b)
        if union_len == 0:
            return 0.0
        return len(set_a & set_b) / union_len

    # Continuous numeric generalized Jaccard: sum(min(a_i, b_i)) / sum(max(a_i, b_i))
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions do not match: {len(a)} != {len(b)}")
    if len(a) == 0:
        return 1.0

    min_sum = 0.0
    max_sum = 0.0
    for x, y in zip(a, b):
        min_sum += min(x, y)
        max_sum += max(x, y)

    if max_sum <= 0.0:
        if min_sum <= 0.0 and max_sum == 0.0:
            # Check if elements are identical
            if all(x == y for x, y in zip(a, b)):
                return 1.0
        return 0.0

    sim = min_sum / max_sum
    return max(0.0, min(1.0, sim))


def jaccard_distance(
    a: Union[Sequence[float], Set[Any], Sequence[Any]],
    b: Union[Sequence[float], Set[Any], Sequence[Any]],
) -> float:
    """Compute Jaccard distance in range [0.0, 1.0]."""
    return 1.0 - jaccard_similarity(a, b)


def compute_distance(
    a: Sequence[float],
    b: Sequence[float],
    metric: Union[DistanceMetric, str],
) -> float:
    """Compute distance between vectors according to metric."""
    if isinstance(metric, str):
        metric = DistanceMetric.from_str(metric)

    if metric == DistanceMetric.COSINE:
        return cosine_distance(a, b)
    elif metric == DistanceMetric.EUCLIDEAN:
        return euclidean_distance(a, b)
    elif metric == DistanceMetric.DOT_PRODUCT:
        return dot_product_distance(a, b)
    elif metric == DistanceMetric.MANHATTAN:
        return manhattan_distance(a, b)
    elif metric == DistanceMetric.HAMMING:
        return hamming_distance(a, b)
    elif metric == DistanceMetric.JACCARD:
        return jaccard_distance(a, b)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def compute_similarity(
    a: Sequence[float],
    b: Sequence[float],
    metric: Union[DistanceMetric, str],
) -> float:
    """Compute similarity score (higher is more similar) according to metric."""
    if isinstance(metric, str):
        metric = DistanceMetric.from_str(metric)

    if metric == DistanceMetric.COSINE:
        return cosine_similarity(a, b)
    elif metric == DistanceMetric.EUCLIDEAN:
        return euclidean_similarity(a, b)
    elif metric == DistanceMetric.DOT_PRODUCT:
        return dot_product(a, b)
    elif metric == DistanceMetric.MANHATTAN:
        return manhattan_similarity(a, b)
    elif metric == DistanceMetric.HAMMING:
        return hamming_similarity(a, b)
    elif metric == DistanceMetric.JACCARD:
        return jaccard_similarity(a, b)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def distance_to_score(
    distance: float,
    metric: Union[DistanceMetric, str],
) -> float:
    """Convert distance value to similarity score (higher is better)."""
    if isinstance(metric, str):
        metric = DistanceMetric.from_str(metric)

    if metric == DistanceMetric.COSINE:
        return max(-1.0, min(1.0, 1.0 - distance))
    elif metric == DistanceMetric.EUCLIDEAN:
        return 1.0 / (1.0 + max(0.0, distance))
    elif metric == DistanceMetric.DOT_PRODUCT:
        return -distance
    elif metric == DistanceMetric.MANHATTAN:
        return 1.0 / (1.0 + max(0.0, distance))
    elif metric == DistanceMetric.HAMMING:
        return 1.0 - max(0.0, min(1.0, distance))
    elif metric == DistanceMetric.JACCARD:
        return 1.0 - max(0.0, min(1.0, distance))
    else:
        return 1.0 / (1.0 + max(0.0, distance))


def normalize_vector(vec: Sequence[float]) -> List[float]:
    """Compute L2 unit normal of a vector.

    If vector norm is zero, returns zero vector.
    """
    norm_sq = sum(x * x for x in vec)
    if norm_sq <= 0.0:
        return [0.0] * len(vec)
    norm = math.sqrt(norm_sq)
    return [x / norm for x in vec]


def compute_centroid(vectors: Sequence[Sequence[float]]) -> List[float]:
    """Compute element-wise centroid (mean vector) of a collection of vectors."""
    if not vectors:
        return []
    dim = len(vectors[0])
    n = len(vectors)
    if n == 1:
        return [float(x) for x in vectors[0]]

    sums = [0.0] * dim
    for vec in vectors:
        if len(vec) != dim:
            raise ValueError(f"Inconsistent vector dimension: {len(vec)} != {dim}")
        for i, val in enumerate(vec):
            sums[i] += val

    return [s / n for s in sums]


def scalar_quantize(
    vector: Sequence[float],
    num_bits: int = 8,
) -> Tuple[bytes, float, float]:
    """Scalar quantize a float vector into uint8 bytes with min/max scaling.

    Returns:
        (quantized_bytes, min_val, max_val)
    """
    if not vector:
        return b"", 0.0, 0.0

    min_val = min(vector)
    max_val = max(vector)
    span = max_val - min_val

    max_int = (1 << num_bits) - 1  # 255 for 8 bits

    if span <= 1e-12:
        # All values identical
        byte_vals = [0] * len(vector)
    else:
        byte_vals = [
            int(round(((x - min_val) / span) * max_int))
            for x in vector
        ]
        # Clamp bounds
        byte_vals = [max(0, min(max_int, b)) for b in byte_vals]

    return bytes(byte_vals), float(min_val), float(max_val)


def dequantize_scalar(
    quantized_bytes: bytes,
    min_val: float,
    max_val: float,
    num_bits: int = 8,
) -> List[float]:
    """Dequantize uint8 bytes back into floating point vector."""
    if not quantized_bytes:
        return []

    max_int = (1 << num_bits) - 1
    span = max_val - min_val

    if span <= 1e-12:
        return [min_val] * len(quantized_bytes)

    scale = span / max_int
    return [min_val + b * scale for b in quantized_bytes]


def binary_quantize(vector: Sequence[float]) -> bytes:
    """1-bit binary sign quantization.

    Packs 8 float signs (1 if > 0 else 0) into 1 byte.
    """
    if not vector:
        return b""

    byte_list = bytearray()
    current_byte = 0
    bit_idx = 0

    for val in vector:
        if val > 0.0:
            current_byte |= (1 << (7 - bit_idx))
        bit_idx += 1
        if bit_idx == 8:
            byte_list.append(current_byte)
            current_byte = 0
            bit_idx = 0

    if bit_idx > 0:
        byte_list.append(current_byte)

    return bytes(byte_list)


def dequantize_binary(quantized_bytes: bytes, dimension: int) -> List[float]:
    """Dequantize 1-bit packed binary bytes into signed floats (+1.0 / -1.0)."""
    if dimension <= 0:
        return []

    result = []
    for byte_val in quantized_bytes:
        for bit_idx in range(8):
            if len(result) >= dimension:
                break
            bit = (byte_val >> (7 - bit_idx)) & 1
            result.append(1.0 if bit == 1 else -1.0)

    # Pad with -1.0 if needed
    while len(result) < dimension:
        result.append(-1.0)

    return result


def hamming_distance_bytes(a: bytes, b: bytes) -> int:
    """Compute fast bitwise Hamming distance between two raw byte sequences."""
    min_len = min(len(a), len(b))
    dist = 0
    for byte_a, byte_b in zip(a[:min_len], b[:min_len]):
        dist += (byte_a ^ byte_b).bit_count()
    if len(a) != len(b):
        dist += abs(len(a) - len(b)) * 8
    return dist


def random_projection(
    vectors: Sequence[Sequence[float]],
    target_dim: int = 2,
    seed: int = 42,
) -> List[List[float]]:
    """Project high-dimensional vectors to 2D or 3D using Gaussian random projection.

    Deterministic using seed for consistent visualizer layout.
    """
    if not vectors:
        return []

    num_vectors = len(vectors)
    input_dim = len(vectors[0])

    if input_dim <= target_dim:
        # If input dim is already smaller or equal, pad or truncate
        return [
            list(vec[:target_dim]) + [0.0] * max(0, target_dim - len(vec))
            for vec in vectors
        ]

    # Generate random projection matrix of shape (input_dim, target_dim)
    rng = random.Random(seed)
    scale = 1.0 / math.sqrt(target_dim)
    proj_matrix: List[List[float]] = [
        [rng.gauss(0.0, 1.0) * scale for _ in range(target_dim)]
        for _ in range(input_dim)
    ]

    projected: List[List[float]] = []
    for vec in vectors:
        point = [0.0] * target_dim
        for i, val in enumerate(vec):
            row = proj_matrix[i]
            for j in range(target_dim):
                point[j] += val * row[j]
        projected.append(point)

    return projected


def generate_random_vector(
    dim: int,
    seed: Optional[int] = None,
    normalized: bool = True,
) -> List[float]:
    """Generate a random Gaussian or normalized random vector."""
    if dim <= 0:
        raise ValueError(f"Dimension must be positive, got {dim}")

    rng = random.Random(seed) if seed is not None else random.Random()
    raw = [rng.gauss(0.0, 1.0) for _ in range(dim)]

    if normalized:
        return normalize_vector(raw)
    return raw


class ProductQuantizer:
    """Pure Python Product Quantization (PQ) engine for vector compression.

    Partitions D-dimensional vectors into M orthogonal sub-vectors of size d = D // M.
    Quantizes each subspace into K centroids (default 256 for 1 byte per subspace).
    Enables Asymmetric Distance Computation (ADC) with precomputed query lookup tables.
    """

    def __init__(
        self,
        dimension: int,
        num_subspaces: int = 8,
        num_centroids: int = 256,
    ) -> None:
        if num_subspaces <= 0:
            raise ValueError(f"num_subspaces must be positive, got {num_subspaces}")
        if dimension % num_subspaces != 0:
            raise ValueError(
                f"Dimension {dimension} must be divisible by num_subspaces {num_subspaces}"
            )
        if not (1 <= num_centroids <= 256):
            raise ValueError(
                f"num_centroids must be between 1 and 256, got {num_centroids}"
            )

        self.dimension = dimension
        self.num_subspaces = num_subspaces
        self.sub_dim = dimension // num_subspaces
        self.num_centroids = num_centroids
        # codebook: shape (M, K, sub_dim)
        self.codebook: Optional[List[List[List[float]]]] = None

    def is_trained(self) -> bool:
        """Check whether the quantizer codebook has been trained."""
        return self.codebook is not None and len(self.codebook) == self.num_subspaces

    def train(
        self,
        vectors: Sequence[Sequence[float]],
        max_iters: int = 15,
        seed: int = 42,
    ) -> None:
        """Train codebook centroids across all subspaces using k-means."""
        if not vectors:
            raise ValueError("Training vectors cannot be empty")
        for v in vectors:
            if len(v) != self.dimension:
                raise ValueError(
                    f"Vector dimension {len(v)} does not match quantizer dimension {self.dimension}"
                )

        rng = random.Random(seed)
        codebook: List[List[List[float]]] = []

        for m in range(self.num_subspaces):
            start_idx = m * self.sub_dim
            end_idx = start_idx + self.sub_dim
            sub_vectors = [v[start_idx:end_idx] for v in vectors]

            # Initialize centroids
            k = min(self.num_centroids, len(sub_vectors))
            centroids: List[List[float]] = [list(x) for x in rng.sample(sub_vectors, k)]
            while len(centroids) < self.num_centroids:
                centroids.append(list(rng.choice(sub_vectors)))

            # Iterative k-means
            for _ in range(max_iters):
                clusters: List[List[Sequence[float]]] = [[] for _ in range(self.num_centroids)]
                for sv in sub_vectors:
                    # Find nearest centroid
                    best_c = 0
                    best_d = float("inf")
                    for c_idx, c_vec in enumerate(centroids):
                        d = sum((a - b) * (a - b) for a, b in zip(sv, c_vec))
                        if d < best_dist if (best_dist := d) < float("inf") else True:
                            pass
                    for c_idx, c_vec in enumerate(centroids):
                        d = sum((a - b) * (a - b) for a, b in zip(sv, c_vec))
                        if d < best_d:
                            best_d = d
                            best_c = c_idx
                    clusters[best_c].append(sv)

                # Recompute centroids
                for c_idx in range(self.num_centroids):
                    pts = clusters[c_idx]
                    if pts:
                        new_centroid = [
                            sum(pt[dim_i] for pt in pts) / len(pts)
                            for dim_i in range(self.sub_dim)
                        ]
                        centroids[c_idx] = new_centroid
                    else:
                        centroids[c_idx] = list(rng.choice(sub_vectors))

            codebook.append(centroids)

        self.codebook = codebook

    def encode(self, vector: Sequence[float]) -> bytes:
        """Compress a D-dimensional vector into M bytes."""
        if not self.is_trained():
            raise RuntimeError("ProductQuantizer must be trained before encoding")
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} does not match {self.dimension}"
            )

        code_bytes = bytearray(self.num_subspaces)
        for m in range(self.num_subspaces):
            start_idx = m * self.sub_dim
            end_idx = start_idx + self.sub_dim
            sv = vector[start_idx:end_idx]

            centroids = self.codebook[m]  # type: ignore[index]
            best_idx = 0
            best_dist = float("inf")
            for c_idx, c_vec in enumerate(centroids):
                d = sum((a - b) * (a - b) for a, b in zip(sv, c_vec))
                if d < best_dist:
                    best_dist = d
                    best_idx = c_idx

            code_bytes[m] = best_idx

        return bytes(code_bytes)

    def decode(self, code: bytes) -> List[float]:
        """Reconstruct an approximation of the vector from its M-byte code."""
        if not self.is_trained():
            raise RuntimeError("ProductQuantizer must be trained before decoding")
        if len(code) != self.num_subspaces:
            raise ValueError(
                f"Code length {len(code)} does not match num_subspaces {self.num_subspaces}"
            )

        reconstructed: List[float] = []
        for m in range(self.num_subspaces):
            c_idx = code[m]
            reconstructed.extend(self.codebook[m][c_idx])  # type: ignore[index]

        return reconstructed

    def compute_distance_table(
        self,
        query: Sequence[float],
        metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
    ) -> List[List[float]]:
        """Precompute distance lookup table of shape (M, K) for query sub-vectors."""
        if not self.is_trained():
            raise RuntimeError("ProductQuantizer must be trained before computing distance table")
        if len(query) != self.dimension:
            raise ValueError(f"Query dimension {len(query)} does not match {self.dimension}")

        table: List[List[float]] = []
        for m in range(self.num_subspaces):
            start_idx = m * self.sub_dim
            end_idx = start_idx + self.sub_dim
            q_sub = query[start_idx:end_idx]

            row: List[float] = []
            for c_vec in self.codebook[m]:  # type: ignore[index]
                if metric == DistanceMetric.EUCLIDEAN:
                    row.append(sum((a - b) * (a - b) for a, b in zip(q_sub, c_vec)))
                elif metric == DistanceMetric.DOT_PRODUCT:
                    row.append(-sum(a * b for a, b in zip(q_sub, c_vec)))
                elif metric == DistanceMetric.MANHATTAN:
                    row.append(sum(abs(a - b) for a, b in zip(q_sub, c_vec)))
                else:
                    row.append(sum((a - b) * (a - b) for a, b in zip(q_sub, c_vec)))
            table.append(row)

        return table

    def asymmetric_distance_with_table(
        self,
        distance_table: Sequence[Sequence[float]],
        code: bytes,
    ) -> float:
        """Fast Asymmetric Distance Computation (ADC) using precomputed lookup table."""
        dist = 0.0
        for m in range(self.num_subspaces):
            dist += distance_table[m][code[m]]
        return dist

    def asymmetric_distance(
        self,
        query: Sequence[float],
        code: bytes,
        metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
    ) -> float:
        """Compute ADC distance directly from query to encoded code."""
        table = self.compute_distance_table(query, metric=metric)
        return self.asymmetric_distance_with_table(table, code)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize quantizer parameters and codebook."""
        return {
            "dimension": self.dimension,
            "num_subspaces": self.num_subspaces,
            "num_centroids": self.num_centroids,
            "codebook": self.codebook,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductQuantizer":
        """Reconstruct a ProductQuantizer from dictionary."""
        pq = cls(
            dimension=data["dimension"],
            num_subspaces=data["num_subspaces"],
            num_centroids=data["num_centroids"],
        )
        pq.codebook = data.get("codebook")
        return pq

