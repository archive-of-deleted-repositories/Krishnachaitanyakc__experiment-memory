"""Inspiration tracker for cross-pollination between experiments."""

from __future__ import annotations

from typing import Any


class InspirationTracker:
    """Tracks which experiments inspired others."""

    def __init__(self) -> None:
        # Maps experiment_id -> list of experiment_ids it was inspired by
        self._inspirations: dict[str, list[str]] = {}

    def record_inspiration(
        self, experiment_id: str, inspired_by_ids: list[str]
    ) -> None:
        """Record that an experiment was inspired by other experiments."""
        self._inspirations[experiment_id] = list(inspired_by_ids)

    def get_inspiration_chain(self, experiment_id: str) -> list[str]:
        """Get the full chain of inspirations for an experiment (recursive)."""
        visited: set[str] = set()
        chain: list[str] = []
        self._walk_chain(experiment_id, visited, chain)
        return chain

    def _walk_chain(
        self, experiment_id: str, visited: set[str], chain: list[str]
    ) -> None:
        """Recursively walk the inspiration chain."""
        if experiment_id in visited:
            return
        visited.add(experiment_id)

        parents = self._inspirations.get(experiment_id, [])
        for parent_id in parents:
            if parent_id not in visited:
                chain.append(parent_id)
                self._walk_chain(parent_id, visited, chain)

    def find_related_experiments(self, experiment_id: str) -> list[str]:
        """Find experiments that were directly inspired by this one."""
        related = []
        for exp_id, parents in self._inspirations.items():
            if experiment_id in parents and exp_id != experiment_id:
                related.append(exp_id)
        return related

    def get_all_inspirations(self) -> dict[str, list[str]]:
        """Get all recorded inspirations."""
        return dict(self._inspirations)
