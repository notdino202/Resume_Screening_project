"""File and content hashing utilities for deterministic caching."""

from __future__ import annotations
import hashlib
from pathlib import Path


def compute_file_hash(file_path: str | Path, block_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file at the given path.

    Args:
        file_path: Path to the target file.
        block_size: Chunk size in bytes for reading.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_bytes_hash(content: bytes) -> str:
    """Computes SHA-256 hash of raw byte content.

    Args:
        content: Raw bytes.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    return hashlib.sha256(content).hexdigest()
