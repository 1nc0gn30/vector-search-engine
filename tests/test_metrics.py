"""Tests for vector distance metrics, arithmetic, quantization, and projections."""

from __future__ import annotations

import math
import pytest

from vector_search_engine.models import DistanceMetric
from vector_search_engine.metrics import (
    dot_product,
    cosine_similarity,
    cosine_distance,
    euclidean_distance,
    euclidean_similarity,
    dot_product_distance,
    manhattan_distance,
    manhattan_similarity,
    hamming_distance,
    hamming_similarity,
    jaccard_similarity,
    jaccard_distance,
    compute_distance,
    compute_similarity,
    distance_to_score,
    normalize_vector,
    compute_centroid,
    scalar_quantize,
    dequantize_scalar,
    binary_quantize,
    dequantize_binary,
    hamming_distance_bytes,
    random_projection,
    generate_random_vector,
)


def test_dot_product():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    assert dot_product(a, b) == 32.0  # 4 + 10 + 18

    with pytest.raises(ValueError, match="Vector dimensions do not match"):
        dot_product([1.0], [1.0, 2.0])


def test_cosine_similarity_and_distance():
    a = [1.0, 0.0]
    b = [1.0, 0.0]
    assert math.isclose(cosine_similarity(a, b), 1.0)
    assert math.isclose(cosine_distance(a, b), 0.0)

    # Orthogonal vectors
    c = [0.0, 1.0]
    assert math.isclose(cosine_similarity(a, c), 0.0)
    assert math.isclose(cosine_distance(a, c), 1.0)

    # Opposite vectors
    d = [-1.0, 0.0]
    assert math.isclose(cosine_similarity(a, d), -1.0)
    assert math.isclose(cosine_distance(a, d), 2.0)

    # Zero vector handling
    zero = [0.0, 0.0]
    assert cosine_similarity(zero, a) == 0.0


def test_euclidean_distance_and_similarity():
    a = [0.0, 0.0]
    b = [3.0, 4.0]
    assert math.isclose(euclidean_distance(a, b), 5.0)
    assert math.isclose(euclidean_similarity(a, b), 1.0 / 6.0)

    # Distance to self is 0
    assert math.isclose(euclidean_distance(a, a), 0.0)
    assert math.isclose(euclidean_similarity(a, a), 1.0)


def test_manhattan_distance_and_similarity():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 0.0, -1.0]
    # |1-4| + |2-0| + |3 - (-1)| = 3 + 2 + 4 = 9
    assert math.isclose(manhattan_distance(a, b), 9.0)
    assert math.isclose(manhattan_similarity(a, b), 1.0 / 10.0)


def test_hamming_distance_and_similarity():
    a = [1, 0, 1, 1]
    b = [1, 1, 1, 0]
    # Differ at index 1 and 3 -> 2/4 = 0.5
    assert math.isclose(hamming_distance(a, b), 0.5)
    assert math.isclose(hamming_similarity(a, b), 0.5)

    # Identical
    assert math.isclose(hamming_distance(a, a), 0.0)
    assert math.isclose(hamming_similarity(a, a), 1.0)


def test_jaccard_continuous_and_set():
    # Set-based
    set_a = {"apple", "banana", "cherry"}
    set_b = {"banana", "cherry", "date"}
    assert math.isclose(jaccard_similarity(set_a, set_b), 2.0 / 4.0)
    assert math.isclose(jaccard_distance(set_a, set_b), 0.5)

    # Continuous numeric
    vec_a = [1.0, 2.0, 3.0]
    vec_b = [1.0, 2.0, 3.0]
    assert math.isclose(jaccard_similarity(vec_a, vec_b), 1.0)
    assert math.isclose(jaccard_distance(vec_a, vec_b), 0.0)

    vec_c = [0.0, 2.0, 1.0]
    # min_sum = 0 + 2 + 1 = 3, max_sum = 1 + 2 + 3 = 6 -> sim = 0.5
    assert math.isclose(jaccard_similarity(vec_a, vec_c), 0.5)


def test_compute_distance_and_similarity_dispatches():
    a = [1.0, 2.0]
    b = [2.0, 3.0]

    for metric in DistanceMetric:
        dist = compute_distance(a, b, metric)
        sim = compute_similarity(a, b, metric)
        score = distance_to_score(dist, metric)
        assert isinstance(dist, float)
        assert isinstance(sim, float)
        assert isinstance(score, float)


def test_normalize_vector_and_centroid():
    v = [3.0, 4.0]
    norm = normalize_vector(v)
    assert math.isclose(norm[0], 0.6)
    assert math.isclose(norm[1], 0.8)
    assert math.isclose(math.sqrt(norm[0] ** 2 + norm[1] ** 2), 1.0)

    # Zero vector
    assert normalize_vector([0.0, 0.0]) == [0.0, 0.0]

    # Centroid
    vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    c = compute_centroid(vectors)
    assert math.isclose(c[0], 3.0)
    assert math.isclose(c[1], 4.0)


def test_scalar_quantization():
    vec = [0.0, 0.25, 0.5, 0.75, 1.0]
    q_bytes, min_v, max_v = scalar_quantize(vec, num_bits=8)
    assert len(q_bytes) == len(vec)
    assert min_v == 0.0
    assert max_v == 1.0

    dequant = dequantize_scalar(q_bytes, min_v, max_v, num_bits=8)
    for orig, restored in zip(vec, dequant):
        assert abs(orig - restored) <= 0.01


def test_binary_quantization():
    vec = [1.5, -2.0, 0.5, -0.1, 3.2, 4.0, -1.0, 0.0, 2.0]
    q_bytes = binary_quantize(vec)
    dequant = dequantize_binary(q_bytes, dimension=len(vec))
    assert len(dequant) == len(vec)
    assert dequant[0] == 1.0
    assert dequant[1] == -1.0


def test_random_projection_and_generation():
    vectors = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ]
    proj_2d = random_projection(vectors, target_dim=2, seed=42)
    assert len(proj_2d) == 2
    assert len(proj_2d[0]) == 2
    assert len(proj_2d[1]) == 2

    # Determinism test
    proj_again = random_projection(vectors, target_dim=2, seed=42)
    assert proj_2d == proj_again

    # Random vector generator
    rand_vec = generate_random_vector(16, seed=123, normalized=True)
    assert len(rand_vec) == 16
    norm_len = math.sqrt(sum(x * x for x in rand_vec))
    assert math.isclose(norm_len, 1.0)
