"""Tests for journal manager."""

import pytest
from experiment_memory.journal import JournalManager


class TestJournalManager:
    def setup_method(self):
        self.journal = JournalManager()

    def test_append_entry(self):
        self.journal.append_entry("exp1", "Tried higher LR", "Improved by 3%")
        entries = self.journal.get_journal()
        assert len(entries) == 1
        assert entries[0]["experiment_id"] == "exp1"
        assert entries[0]["reasoning"] == "Tried higher LR"
        assert entries[0]["outcome"] == "Improved by 3%"

    def test_multiple_entries(self):
        self.journal.append_entry("exp1", "Reason 1", "Outcome 1")
        self.journal.append_entry("exp2", "Reason 2", "Outcome 2")
        self.journal.append_entry("exp3", "Reason 3", "Outcome 3")
        entries = self.journal.get_journal()
        assert len(entries) == 3

    def test_search_journal(self):
        self.journal.append_entry("exp1", "Tried higher learning rate", "Improved")
        self.journal.append_entry("exp2", "Added dropout", "No change")
        self.journal.append_entry("exp3", "Changed learning rate schedule", "Improved")

        results = self.journal.search_journal("learning rate")
        assert len(results) >= 2
        for r in results:
            assert "learning rate" in r["reasoning"].lower() or "learning rate" in r["outcome"].lower()

    def test_search_journal_no_results(self):
        self.journal.append_entry("exp1", "Tried higher LR", "Improved")
        results = self.journal.search_journal("transformer architecture")
        assert len(results) == 0

    def test_empty_journal(self):
        assert self.journal.get_journal() == []
        assert self.journal.search_journal("anything") == []

    def test_entry_has_timestamp(self):
        self.journal.append_entry("exp1", "Reason", "Outcome")
        entries = self.journal.get_journal()
        assert "timestamp" in entries[0]

    def test_get_entries_for_experiment(self):
        self.journal.append_entry("exp1", "First try", "Failed")
        self.journal.append_entry("exp1", "Second try", "Success")
        self.journal.append_entry("exp2", "Try", "OK")

        entries = self.journal.get_entries_for_experiment("exp1")
        assert len(entries) == 2
