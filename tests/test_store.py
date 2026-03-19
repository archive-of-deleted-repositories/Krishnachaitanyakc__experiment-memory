"""Tests for experiment store."""

import json
import os

import pytest
from experiment_memory.store import BaseExperimentStore, SimpleMemoryStore


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


class TestTimestamp:
    def test_experiment_has_timestamp(self):
        store = SimpleMemoryStore()
        store.add_experiment("c1", "Test", 0.5)
        history = store.get_history()
        assert "timestamp" in history[0]

    def test_query_by_time_range(self):
        store = SimpleMemoryStore()
        store.add_experiment("c1", "Exp 1", 0.70)
        store.add_experiment("c2", "Exp 2", 0.80)

        history = store.get_history()
        ts = history[0]["timestamp"]
        # Query with a range that includes all
        results = store.query_by_time_range("2000-01-01", "2099-12-31")
        assert len(results) == 2

    def test_query_by_time_range_empty(self):
        store = SimpleMemoryStore()
        store.add_experiment("c1", "Exp 1", 0.70)
        results = store.query_by_time_range("2099-01-01", "2099-12-31")
        assert len(results) == 0


class TestMetadataFiltering:
    def test_query_by_metadata_single_filter(self):
        store = SimpleMemoryStore()
        store.add_experiment("c1", "Exp 1", 0.70, metadata={"status": "success", "gpu": "A100"})
        store.add_experiment("c2", "Exp 2", 0.80, metadata={"status": "failed", "gpu": "A100"})
        store.add_experiment("c3", "Exp 3", 0.90, metadata={"status": "success", "gpu": "V100"})

        results = store.query_by_metadata({"status": "success"})
        assert len(results) == 2

    def test_query_by_metadata_multiple_filters(self):
        store = SimpleMemoryStore()
        store.add_experiment("c1", "Exp 1", 0.70, metadata={"status": "success", "gpu": "A100"})
        store.add_experiment("c2", "Exp 2", 0.80, metadata={"status": "success", "gpu": "V100"})

        results = store.query_by_metadata({"status": "success", "gpu": "A100"})
        assert len(results) == 1
        assert results[0]["commit"] == "c1"

    def test_query_by_metadata_no_match(self):
        store = SimpleMemoryStore()
        store.add_experiment("c1", "Exp 1", 0.70, metadata={"status": "success"})
        results = store.query_by_metadata({"status": "failed"})
        assert len(results) == 0


class TestDeduplication:
    def test_dedup_skips_duplicate_commit(self):
        store = SimpleMemoryStore()
        id1 = store.add_experiment("c1", "Exp 1", 0.70, dedup=True)
        id2 = store.add_experiment("c1", "Exp 1 duplicate", 0.80, dedup=True)
        assert id1 == id2
        assert len(store.get_history()) == 1

    def test_dedup_allows_different_commits(self):
        store = SimpleMemoryStore()
        store.add_experiment("c1", "Exp 1", 0.70, dedup=True)
        store.add_experiment("c2", "Exp 2", 0.80, dedup=True)
        assert len(store.get_history()) == 2

    def test_no_dedup_allows_duplicates(self):
        store = SimpleMemoryStore()
        store.add_experiment("c1", "Exp 1", 0.70, dedup=False)
        store.add_experiment("c1", "Exp 1 dup", 0.80, dedup=False)
        assert len(store.get_history()) == 2


class TestJSONLPersistence:
    def test_persist_and_load(self, tmp_path):
        jsonl_path = str(tmp_path / "experiments.jsonl")
        store1 = SimpleMemoryStore(jsonl_path=jsonl_path)
        store1.add_experiment("c1", "Exp 1", 0.70)
        store1.add_experiment("c2", "Exp 2", 0.80)

        # Load from same file
        store2 = SimpleMemoryStore(jsonl_path=jsonl_path)
        history = store2.get_history()
        assert len(history) == 2
        assert history[0]["commit"] == "c1"
        assert history[1]["commit"] == "c2"

    def test_jsonl_file_format(self, tmp_path):
        jsonl_path = str(tmp_path / "experiments.jsonl")
        store = SimpleMemoryStore(jsonl_path=jsonl_path)
        store.add_experiment("c1", "Exp 1", 0.70)

        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["commit"] == "c1"

    def test_no_file_creates_empty_store(self, tmp_path):
        jsonl_path = str(tmp_path / "nonexistent.jsonl")
        store = SimpleMemoryStore(jsonl_path=jsonl_path)
        assert store.get_history() == []


class TestExportImport:
    def test_export_and_import(self, tmp_path):
        store1 = SimpleMemoryStore()
        store1.add_experiment("c1", "Exp 1", 0.70)
        store1.add_experiment("c2", "Exp 2", 0.80)

        export_path = str(tmp_path / "export.json")
        store1.export_store(export_path)

        store2 = SimpleMemoryStore()
        store2.import_store(export_path)
        history = store2.get_history()
        assert len(history) == 2

    def test_import_deduplicates(self, tmp_path):
        store = SimpleMemoryStore()
        store.add_experiment("c1", "Exp 1", 0.70)

        export_path = str(tmp_path / "export.json")
        store.export_store(export_path)
        store.import_store(export_path)
        assert len(store.get_history()) == 1

    def test_export_format(self, tmp_path):
        store = SimpleMemoryStore()
        store.add_experiment("c1", "Exp 1", 0.70)

        export_path = str(tmp_path / "export.json")
        store.export_store(export_path)

        with open(export_path) as f:
            data = json.load(f)
        assert "experiments" in data
        assert len(data["experiments"]) == 1
