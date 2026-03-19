"""Tests for inspiration tracker."""

import pytest
from autoresearch_memory.inspiration import InspirationTracker


class TestInspirationTracker:
    def setup_method(self):
        self.tracker = InspirationTracker()

    def test_record_inspiration(self):
        self.tracker.record_inspiration("exp3", ["exp1", "exp2"])
        chain = self.tracker.get_inspiration_chain("exp3")
        assert "exp1" in chain
        assert "exp2" in chain

    def test_get_inspiration_chain_single(self):
        self.tracker.record_inspiration("exp2", ["exp1"])
        self.tracker.record_inspiration("exp3", ["exp2"])
        chain = self.tracker.get_inspiration_chain("exp3")
        assert "exp2" in chain
        assert "exp1" in chain

    def test_find_related_experiments(self):
        self.tracker.record_inspiration("exp2", ["exp1"])
        self.tracker.record_inspiration("exp3", ["exp1"])
        self.tracker.record_inspiration("exp4", ["exp2"])

        related = self.tracker.find_related_experiments("exp1")
        assert "exp2" in related
        assert "exp3" in related

    def test_no_inspiration(self):
        chain = self.tracker.get_inspiration_chain("exp1")
        assert chain == []

    def test_find_related_no_relations(self):
        related = self.tracker.find_related_experiments("exp1")
        assert related == []

    def test_circular_chain_handling(self):
        self.tracker.record_inspiration("exp2", ["exp1"])
        self.tracker.record_inspiration("exp1", ["exp2"])
        chain = self.tracker.get_inspiration_chain("exp1")
        assert isinstance(chain, list)
        # Should not infinite loop
        assert len(chain) <= 2

    def test_get_all_inspirations(self):
        self.tracker.record_inspiration("exp2", ["exp1"])
        self.tracker.record_inspiration("exp3", ["exp1", "exp2"])
        all_insp = self.tracker.get_all_inspirations()
        assert len(all_insp) == 2
