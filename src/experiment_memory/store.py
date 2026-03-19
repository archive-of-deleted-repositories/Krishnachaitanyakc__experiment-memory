"""Experiment store with vector similarity search."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime, timezone
from typing import Any


class BaseExperimentStore(ABC):
    """Abstract base class for experiment stores."""

    @abstractmethod
    def add_experiment(
        self,
        commit: str,
        description: str,
        metric_value: float,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add an experiment record. Returns experiment ID."""

    @abstractmethod
    def query_similar(self, description: str, k: int = 5) -> list[dict[str, Any]]:
        """Find experiments with similar descriptions."""

    @abstractmethod
    def query_by_metric_range(
        self, min_val: float, max_val: float
    ) -> list[dict[str, Any]]:
        """Find experiments with metric values in [min_val, max_val]."""

    @abstractmethod
    def get_history(self, n: int = 50) -> list[dict[str, Any]]:
        """Get the last n experiments."""

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_tfidf_vector(tokens: list[str], vocab: dict[str, int], idf: dict[str, float]) -> list[float]:
    """Build a TF-IDF vector for a token list."""
    tf = Counter(tokens)
    vec = [0.0] * len(vocab)
    for token, count in tf.items():
        if token in vocab:
            idx = vocab[token]
            vec[idx] = (count / max(len(tokens), 1)) * idf.get(token, 1.0)
    return vec


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SimpleMemoryStore(BaseExperimentStore):
    """Experiment store with optional JSONL persistence and TF-IDF similarity."""

    def __init__(self, jsonl_path: str | None = None) -> None:
        self._experiments: list[dict[str, Any]] = []
        self._jsonl_path = jsonl_path
        self._commit_set: set[str] = set()
        if jsonl_path and os.path.exists(jsonl_path):
            self._load_from_jsonl()

    def _load_from_jsonl(self) -> None:
        """Load experiments from JSONL file."""
        with open(self._jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    exp = json.loads(line)
                    self._experiments.append(exp)
                    self._commit_set.add(exp["commit"])

    def _append_to_jsonl(self, experiment: dict[str, Any]) -> None:
        """Append a single experiment to the JSONL file."""
        if self._jsonl_path:
            os.makedirs(os.path.dirname(self._jsonl_path) or ".", exist_ok=True)
            with open(self._jsonl_path, "a") as f:
                f.write(json.dumps(experiment) + "\n")

    def add_experiment(
        self,
        commit: str,
        description: str,
        metric_value: float,
        metadata: dict[str, Any] | None = None,
        dedup: bool = False,
    ) -> str:
        if dedup and commit in self._commit_set:
            # Return the existing experiment's ID
            for exp in self._experiments:
                if exp["commit"] == commit:
                    return exp["id"]
            return ""

        exp_id = str(uuid.uuid4())[:8]
        experiment = {
            "id": exp_id,
            "commit": commit,
            "description": description,
            "metric_value": metric_value,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._experiments.append(experiment)
        self._commit_set.add(commit)
        self._append_to_jsonl(experiment)
        return exp_id

    def query_similar(self, description: str, k: int = 5) -> list[dict[str, Any]]:
        if not self._experiments:
            return []

        all_docs = [e["description"] for e in self._experiments]
        all_docs.append(description)

        # Build vocabulary
        all_tokens = [_tokenize(doc) for doc in all_docs]
        vocab: dict[str, int] = {}
        for tokens in all_tokens:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

        if not vocab:
            return []

        # Compute IDF
        n_docs = len(all_docs)
        doc_freq: Counter[str] = Counter()
        for tokens in all_tokens:
            for token in set(tokens):
                doc_freq[token] += 1
        idf = {token: math.log(n_docs / (1 + freq)) for token, freq in doc_freq.items()}

        # Build vectors
        query_vec = _build_tfidf_vector(all_tokens[-1], vocab, idf)
        similarities = []
        for i, exp in enumerate(self._experiments):
            exp_vec = _build_tfidf_vector(all_tokens[i], vocab, idf)
            sim = _cosine_similarity(query_vec, exp_vec)
            similarities.append((sim, exp))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in similarities[:k]]

    def query_by_metric_range(
        self, min_val: float, max_val: float
    ) -> list[dict[str, Any]]:
        return [
            e
            for e in self._experiments
            if min_val <= e["metric_value"] <= max_val
        ]

    def query_by_time_range(
        self, start: str, end: str
    ) -> list[dict[str, Any]]:
        """Find experiments with timestamps in [start, end].

        Args:
            start: ISO format datetime string.
            end: ISO format datetime string.
        """
        return [
            e
            for e in self._experiments
            if "timestamp" in e and start <= e["timestamp"] <= end
        ]

    def query_by_metadata(
        self, filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Find experiments matching all metadata filters (AND semantics)."""
        results = []
        for e in self._experiments:
            meta = e.get("metadata", {})
            if all(meta.get(k) == v for k, v in filters.items()):
                results.append(e)
        return results

    def get_history(self, n: int = 50) -> list[dict[str, Any]]:
        return self._experiments[-n:]

    def get_stats(self) -> dict[str, Any]:
        if not self._experiments:
            return {"count": 0, "min_metric": None, "max_metric": None, "avg_metric": None}

        metrics = [e["metric_value"] for e in self._experiments]
        return {
            "count": len(self._experiments),
            "min_metric": min(metrics),
            "max_metric": max(metrics),
            "avg_metric": sum(metrics) / len(metrics),
        }

    def export_store(self, path: str) -> None:
        """Export all experiments to a single JSON file."""
        with open(path, "w") as f:
            json.dump({"experiments": self._experiments}, f, indent=2)

    def import_store(self, path: str) -> None:
        """Import experiments from a JSON file (exported by export_store)."""
        with open(path, "r") as f:
            data = json.load(f)
        for exp in data.get("experiments", []):
            if exp["commit"] not in self._commit_set:
                self._experiments.append(exp)
                self._commit_set.add(exp["commit"])
                self._append_to_jsonl(exp)
