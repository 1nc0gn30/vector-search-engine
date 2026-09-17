"""Tests for cross-platform compatibility utilities."""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from vector_search_engine.compat import (
    PlatformInfo,
    get_platform_info,
    safe_path_normalization,
    safe_ensure_dir,
    atomic_write_bytes,
    atomic_write_text,
    safe_json_read,
    safe_json_write,
    safe_delete_file,
    safe_delete_dir,
)


def test_get_platform_info():
    info = get_platform_info()
    assert isinstance(info, PlatformInfo)
    assert isinstance(info.os_name, str)
    assert isinstance(info.is_windows, bool)
    assert isinstance(info.is_linux, bool)
    assert isinstance(info.is_macos, bool)
    assert isinstance(info.is_termux, bool)
    assert isinstance(info.is_64bit, bool)
    assert isinstance(info.python_version, str)
    assert isinstance(info.architecture, str)

    d = info.to_dict()
    assert d["os_name"] == info.os_name
    assert d["python_version"] == info.python_version


def test_safe_path_normalization(temp_dir: Path):
    # Empty string should normalize to CWD
    cwd_path = safe_path_normalization("")
    assert cwd_path == Path.cwd().resolve()

    # Relative path
    rel_path = safe_path_normalization("./subfolder")
    assert rel_path == (Path.cwd() / "subfolder").resolve()

    # Absolute path
    p = temp_dir / "a" / "b"
    assert safe_path_normalization(p) == p.resolve()


def test_safe_ensure_dir(temp_dir: Path):
    target = temp_dir / "deep" / "nested" / "dir"
    res = safe_ensure_dir(target)
    assert res.exists()
    assert res.is_dir()


def test_atomic_write_bytes_and_text(temp_dir: Path):
    target_bin = temp_dir / "data.bin"
    atomic_write_bytes(target_bin, b"\x00\x01\x02\x03\x04")
    assert target_bin.exists()
    assert target_bin.read_bytes() == b"\x00\x01\x02\x03\x04"

    target_txt = temp_dir / "file.txt"
    atomic_write_text(target_txt, "Hello, Vector Search Engine! 🚀")
    assert target_txt.exists()
    assert target_txt.read_text(encoding="utf-8") == "Hello, Vector Search Engine! 🚀"


def test_safe_json_read_write(temp_dir: Path):
    json_path = temp_dir / "config.json"
    data = {"name": "test_collection", "dimension": 128, "metric": "cosine", "active": True}

    safe_json_write(json_path, data)
    assert json_path.exists()

    loaded = safe_json_read(json_path)
    assert loaded == data

    # Default fallback on non-existent file
    missing_path = temp_dir / "non_existent.json"
    assert safe_json_read(missing_path, default={"fallback": 1}) == {"fallback": 1}

    # Exception raised when default is None and file missing
    with pytest.raises(FileNotFoundError):
        safe_json_read(missing_path)


def test_safe_delete_file_and_dir(temp_dir: Path):
    # Test delete file
    f = temp_dir / "temp.txt"
    f.write_text("delete me")
    assert f.exists()
    assert safe_delete_file(f) is True
    assert not f.exists()
    assert safe_delete_file(f) is False  # Second delete returns False

    # Test delete dir
    d = temp_dir / "sub_to_delete"
    d.mkdir()
    (d / "inner.txt").write_text("inner")
    assert safe_delete_dir(d) is True
    assert not d.exists()
    assert safe_delete_dir(d) is False
