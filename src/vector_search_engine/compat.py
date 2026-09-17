"""Cross-platform compatibility utilities for vector-search-engine.

Provides atomic file I/O, safe path normalization, platform detection,
and robust directory/file manipulation supporting Linux, macOS, Windows,
and Android Termux environments.
"""

from __future__ import annotations

import os
import sys
import json
import tempfile
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass(frozen=True)
class PlatformInfo:
    """System platform information."""

    os_name: str
    is_windows: bool
    is_linux: bool
    is_macos: bool
    is_termux: bool
    is_64bit: bool
    python_version: str
    architecture: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert platform info to dictionary."""
        return {
            "os_name": self.os_name,
            "is_windows": self.is_windows,
            "is_linux": self.is_linux,
            "is_macos": self.is_macos,
            "is_termux": self.is_termux,
            "is_64bit": self.is_64bit,
            "python_version": self.python_version,
            "architecture": self.architecture,
        }


def get_platform_info() -> PlatformInfo:
    """Detect and return current platform information."""
    system_name = platform.system().lower()
    is_windows = system_name == "windows"
    is_macos = system_name == "darwin"
    is_linux = system_name == "linux"
    is_termux = "com.termux" in os.environ.get("PREFIX", "") or "TERMUX_VERSION" in os.environ

    return PlatformInfo(
        os_name=platform.system(),
        is_windows=is_windows,
        is_linux=is_linux,
        is_macos=is_macos,
        is_termux=is_termux,
        is_64bit=sys.maxsize > 2**32,
        python_version=platform.python_version(),
        architecture=platform.machine(),
    )


def safe_path_normalization(path: Union[str, Path]) -> Path:
    """Normalize path across Linux, macOS, Windows, and Termux.

    Expands user directories (~), resolves symlinks/relative paths,
    and handles Windows UNC / drive letters and Termux prefix paths.
    """
    if isinstance(path, str):
        # Handle empty or whitespace-only paths
        stripped = path.strip()
        if not stripped:
            return Path.cwd()
        # Handle Termux path prefix if running in Termux
        if "PREFIX" in os.environ and stripped.startswith("$PREFIX"):
            stripped = stripped.replace("$PREFIX", os.environ["PREFIX"], 1)
        path = Path(stripped)

    expanded = path.expanduser()
    try:
        # Resolve to absolute path
        return expanded.resolve()
    except (RuntimeError, PermissionError, OSError):
        # Fallback to absolute if resolve fails (e.g. symlink loop or restricted permissions)
        return expanded.absolute()


def safe_ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists with safe permissions. Returns resolved Path."""
    p = safe_path_normalization(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_bytes(
    path: Union[str, Path],
    content: bytes,
    sync: bool = True,
) -> Path:
    """Atomically write binary data to a file using temporary file and atomic replace.

    Uses fsync before replace to ensure data durability across crashes.
    """
    target = safe_path_normalization(path)
    parent_dir = target.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same filesystem directory for atomic rename
    temp_fd, temp_path = tempfile.mkstemp(
        dir=str(parent_dir),
        prefix=f".{target.name}.tmp_",
    )

    try:
        with os.fdopen(temp_fd, "wb") as f:
            f.write(content)
            f.flush()
            if sync:
                try:
                    os.fsync(f.fileno())
                except (OSError, AttributeError):
                    pass

        # Atomic replace (works on Windows Python 3.3+ and POSIX)
        os.replace(temp_path, target)

        # Directory sync on POSIX for directory entry durability
        if sync and hasattr(os, "O_DIRECTORY"):
            try:
                dir_fd = os.open(str(parent_dir), os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except (OSError, AttributeError):
                pass

        return target

    except Exception:
        # Clean up temporary file on failure
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise


def atomic_write_text(
    path: Union[str, Path],
    content: str,
    encoding: str = "utf-8",
    sync: bool = True,
) -> Path:
    """Atomically write text data to a file."""
    raw_bytes = content.encode(encoding)
    return atomic_write_bytes(path, raw_bytes, sync=sync)


def safe_json_read(
    path: Union[str, Path],
    encoding: str = "utf-8",
    default: Optional[Any] = None,
) -> Any:
    """Safely read and parse a JSON file.

    If file does not exist and default is provided, returns default.
    """
    target = safe_path_normalization(path)
    if not target.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON file not found: {target}")

    with open(target, "r", encoding=encoding) as f:
        return json.load(f)


def safe_json_write(
    path: Union[str, Path],
    data: Any,
    indent: Optional[int] = 2,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    sync: bool = True,
) -> Path:
    """Safely and atomically serialize and write data as JSON."""
    json_str = json.dumps(
        data,
        indent=indent,
        ensure_ascii=ensure_ascii,
        default=str,
    )
    return atomic_write_text(path, json_str, encoding=encoding, sync=sync)


def safe_delete_file(path: Union[str, Path]) -> bool:
    """Safely delete a file if it exists. Returns True if deleted, False otherwise."""
    try:
        target = safe_path_normalization(path)
        if target.exists() and target.is_file():
            target.unlink()
            return True
        return False
    except OSError:
        return False


def safe_delete_dir(path: Union[str, Path]) -> bool:
    """Safely remove a directory and all its contents if it exists."""
    try:
        target = safe_path_normalization(path)
        if target.exists() and target.is_dir():
            shutil.rmtree(target)
            return True
        return False
    except OSError:
        return False
