"""Pytest configuration and common fixtures for vector-search-engine test suite."""

from __future__ import annotations

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator, List, Dict, Any

import pytest

# Ensure `src` directory is at the beginning of sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that is cleaned up after the test."""
    temp_path = Path(tempfile.mkdtemp(prefix="vector_test_"))
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_vectors_2d() -> List[List[float]]:
    """Small 2D sample vector dataset for exact geometry / metric tests."""
    return [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [0.70710678, 0.70710678],
        [0.5, 0.5],
    ]


@pytest.fixture
def sample_vectors_4d() -> List[List[float]]:
    """Sample 4D vectors."""
    return [
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [-1.0, 0.0, 1.0, 0.0],
        [2.0, 0.0, -2.0, 1.0],
    ]


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """Sample document records with text, metadata, and vectors."""
    return [
        {
            "id": "doc1",
            "text": "Deep learning architectures for natural language processing",
            "vector": [0.9, 0.1, 0.2, 0.05],
            "metadata": {"category": "ai", "year": 2023, "tags": ["nlp", "deep-learning"]},
        },
        {
            "id": "doc2",
            "text": "Vector search indexing with HNSW and hierarchical graphs",
            "vector": [0.85, 0.15, 0.3, 0.1],
            "metadata": {"category": "database", "year": 2024, "tags": ["vector", "hnsw"]},
        },
        {
            "id": "doc3",
            "text": "Relational databases and SQL query optimization strategies",
            "vector": [0.1, 0.8, 0.1, 0.7],
            "metadata": {"category": "database", "year": 2021, "tags": ["sql", "rdbms"]},
        },
        {
            "id": "doc4",
            "text": "Quantum computing algorithms and superconducting qubits",
            "vector": [0.05, 0.1, 0.9, 0.85],
            "metadata": {"category": "physics", "year": 2022, "tags": ["quantum", "hardware"]},
        },
        {
            "id": "doc5",
            "text": "Information retrieval and BM25 lexical search ranking",
            "vector": [0.6, 0.5, 0.2, 0.3],
            "metadata": {"category": "search", "year": 2023, "tags": ["ir", "bm25"]},
        },
    ]
