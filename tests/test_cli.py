"""Integration and unit tests for command-line interface (CLI)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from vector_search_engine.cli import (
    main,
    parse_vector_input,
    parse_metadata_input,
)


def test_parse_vector_input():
    # JSON list
    v1 = parse_vector_input("[0.1, 0.2, 0.3, 0.4]")
    assert v1 == [0.1, 0.2, 0.3, 0.4]

    # Comma-separated
    v2 = parse_vector_input("1.5, -2.0, 3.25")
    assert v2 == [1.5, -2.0, 3.25]

    with pytest.raises(ValueError, match="Unable to parse vector"):
        parse_vector_input("")


def test_parse_metadata_input():
    # JSON dict
    m1 = parse_metadata_input('{"category": "ai", "year": 2026}')
    assert m1 == {"category": "ai", "year": 2026}

    # Key=value pairs
    m2 = parse_metadata_input("topic=nlp,author=vaswani")
    assert m2 == {"topic": "nlp", "author": "vaswani"}

    assert parse_metadata_input(None) == {}


def test_cli_smoke_commands(temp_dir: Path):
    data_dir_arg = ["--data-dir", str(temp_dir), "--quiet"]

    # 1. Create collection
    create_args = [
        "create",
        "cli_test_col",
        "-d", "4",
        "-m", "cosine",
        "-i", "hnsw",
    ] + data_dir_arg
    ret = main(create_args)
    assert ret == 0

    # 2. Insert items
    insert_args = [
        "insert",
        "cli_test_col",
        "--id", "doc_1",
        "-v", "0.9,0.1,0.2,0.05",
        "-d", "Deep learning transformer architecture",
        "--meta", '{"category": "ai"}',
    ] + data_dir_arg
    ret = main(insert_args)
    assert ret == 0

    # 3. Query
    query_args = [
        "query",
        "cli_test_col",
        "-v", "0.9,0.1,0.2,0.05",
        "-k", "1",
    ] + data_dir_arg
    ret = main(query_args)
    assert ret == 0

    # 4. Hybrid Search
    hybrid_args = [
        "hybrid",
        "cli_test_col",
        "transformer architecture",
        "-v", "0.9,0.1,0.2,0.05",
        "-a", "0.5",
        "-k", "1",
    ] + data_dir_arg
    ret = main(hybrid_args)
    assert ret == 0

    # 5. Stats
    stats_args = ["stats", "cli_test_col"] + data_dir_arg
    ret = main(stats_args)
    assert ret == 0

    # 6. List
    list_args = ["list"] + data_dir_arg
    ret = main(list_args)
    assert ret == 0

    # 7. Embed text
    embed_args = [
        "embed",
        "Hello semantic search!",
        "-d", "8",
    ] + data_dir_arg
    ret = main(embed_args)
    assert ret == 0

    # 8. Benchmark
    bench_args = [
        "benchmark",
        "-n", "50",
        "-d", "4",
        "-c", "10",
        "-k", "5",
    ] + data_dir_arg
    ret = main(bench_args)
    assert ret == 0

    # 9. Diagnostics
    diag_args = ["diagnostics"] + data_dir_arg
    ret = main(diag_args)
    assert ret == 0
