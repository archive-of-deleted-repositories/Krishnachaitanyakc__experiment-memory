"""Journal manager for capturing agent reasoning chains."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class JournalManager:
    """Manages a journal of reasoning entries for experiments."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def append_entry(
        self, experiment_id: str, reasoning: str, outcome: str
    ) -> None:
        """Add a journal entry."""
        self._entries.append(
            {
                "experiment_id": experiment_id,
                "reasoning": reasoning,
                "outcome": outcome,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_journal(self) -> list[dict[str, Any]]:
        """Get all journal entries."""
        return list(self._entries)

    def search_journal(self, query: str) -> list[dict[str, Any]]:
        """Search journal entries by keyword in reasoning or outcome."""
        query_lower = query.lower()
        return [
            entry
            for entry in self._entries
            if query_lower in entry["reasoning"].lower()
            or query_lower in entry["outcome"].lower()
        ]

    def get_entries_for_experiment(
        self, experiment_id: str
    ) -> list[dict[str, Any]]:
        """Get all journal entries for a specific experiment."""
        return [
            entry
            for entry in self._entries
            if entry["experiment_id"] == experiment_id
        ]
