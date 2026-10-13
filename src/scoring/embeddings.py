"""Embedding management and caching module using sentence-transformers (all-MiniLM-L6-v2)."""

from __future__ import annotations
import hashlib
from typing import Optional, Sequence
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Computes and caches dense semantic embeddings locally on CPU."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self._cache: dict[str, np.ndarray] = {}

    def _hash_text(self, text: str) -> str:
        """Computes deterministic hash for a text string."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def get_embedding(self, text: str) -> np.ndarray:
        """Retrieves or computes normalized embedding vector for single text."""
        cleaned = text.strip()
        if not cleaned:
            # Return zero vector of appropriate dimension (384)
            return np.zeros(self.model.get_sentence_embedding_dimension(), dtype=np.float32)

        key = self._hash_text(cleaned)
        if key in self._cache:
            return self._cache[key]

        import torch
        with torch.inference_mode():
            vec = self.model.encode(cleaned, normalize_embeddings=True, show_progress_bar=False)
        self._cache[key] = np.array(vec, dtype=np.float32)
        return self._cache[key]

    def get_embeddings_batch(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Computes embeddings for a batch of strings, utilizing cache where available."""
        results: list[np.ndarray] = [None] * len(texts)  # type: ignore
        to_encode_indices: list[int] = []
        to_encode_texts: list[str] = []

        for idx, text in enumerate(texts):
            cleaned = text.strip()
            if not cleaned:
                results[idx] = np.zeros(self.model.get_sentence_embedding_dimension(), dtype=np.float32)
                continue
            key = self._hash_text(cleaned)
            if key in self._cache:
                results[idx] = self._cache[key]
            else:
                to_encode_indices.append(idx)
                to_encode_texts.append(cleaned)

        if to_encode_texts:
            import torch
            with torch.inference_mode():
                encoded = self.model.encode(
                    to_encode_texts,
                    batch_size=16,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            for orig_idx, vec in zip(to_encode_indices, encoded):
                key = self._hash_text(to_encode_texts[orig_idx if False else to_encode_indices.index(orig_idx)])
                vec_arr = np.array(vec, dtype=np.float32)
                self._cache[key] = vec_arr
                results[orig_idx] = vec_arr

        return results

    def compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Computes cosine similarity between two normalized vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
