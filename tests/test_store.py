"""Tests for experiment store."""

import pytest
from autoresearch_memory.store import BaseExperimentStore, SimpleMemoryStore


class TestSimpleMemoryStore:
    def setup_method(self):
        self.store = SimpleMemoryStore()

    def test_is_base_store(self):
        assert isinstance(self.store, BaseExperimentStore)

    def test_add_experiment(self):
        self.store.add_experiment(
            commit="abc123",
            description="Increased learning rate to 0.01",
            metric_value=0.85,
            metadata={"epoch": 10},
        )
        history = self.store.get_history(n=10)
        assert len(history) == 1
        assert history[0]["commit"] == "abc123"
        assert history[0]["description"] == "Increased learning rate to 0.01"
        assert history[0]["metric_value"] == 0.85
        assert history[0]["metadata"] == {"epoch": 10}

    def test_add_multiple_experiments(self):
        for i in range(5):
            self.store.add_experiment(
                commit=f"commit_{i}",
                description=f"Experiment {i}",
                metric_value=0.5 + i * 0.1,
            )
        history = self.store.get_history(n=10)
        assert len(history) == 5

    def test_get_history_limits(self):
        for i in range(10):
            self.store.add_experiment(
                commit=f"commit_{i}",
                description=f"Experiment {i}",
                metric_value=float(i),
            )
        history = self.store.get_history(n=3)
        assert len(history) == 3

    def test_query_similar(self):
        self.store.add_experiment("c1", "Increased learning rate to 0.01", 0.85)
        self.store.add_experiment("c2", "Added dropout regularization", 0.82)
        self.store.add_experiment("c3", "Changed learning rate schedule", 0.88)
        self.store.add_experiment("c4", "Switched to Adam optimizer", 0.80)

        results = self.store.query_similar("learning rate", k=2)
        assert len(results) == 2
        descriptions = [r["description"] for r in results]
        assert any("learning rate" in d.lower() for d in descriptions)

    def test_query_by_metric_range(self):
        self.store.add_experiment("c1", "Exp 1", 0.70)
        self.store.add_experiment("c2", "Exp 2", 0.80)
        self.store.add_experiment("c3", "Exp 3", 0.90)
        self.store.add_experiment("c4", "Exp 4", 0.60)

        results = self.store.query_by_metric_range(0.75, 0.85)
        assert len(results) == 1
        assert results[0]["metric_value"] == 0.80

    def test_query_by_metric_range_inclusive(self):
        self.store.add_experiment("c1", "Exp 1", 0.70)
        self.store.add_experiment("c2", "Exp 2", 0.80)

        results = self.store.query_by_metric_range(0.70, 0.80)
        assert len(results) == 2

    def test_empty_store_queries(self):
        assert self.store.get_history() == []
        assert self.store.query_similar("anything") == []
        assert self.store.query_by_metric_range(0, 1) == []

    def test_default_metadata(self):
        self.store.add_experiment("c1", "Test", 0.5)
        history = self.store.get_history()
        assert history[0]["metadata"] == {}

    def test_stats(self):
        self.store.add_experiment("c1", "Exp 1", 0.70)
        self.store.add_experiment("c2", "Exp 2", 0.90)
        stats = self.store.get_stats()
        assert stats["count"] == 2
        assert stats["min_metric"] == 0.70
        assert stats["max_metric"] == 0.90
        assert stats["avg_metric"] == pytest.approx(0.80)
