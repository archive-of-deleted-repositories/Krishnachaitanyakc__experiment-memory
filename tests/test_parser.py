"""Tests for results parser."""

import pytest
from autoresearch_memory.parser import ExperimentParser, ExperimentRecord


SAMPLE_TSV = """commit\tmetric_value\tmemory_gb\tstatus\tdescription
abc123\t0.85\t4.2\tsuccess\tIncreased learning rate to 0.01
def456\t0.82\t3.8\tsuccess\tAdded dropout regularization
ghi789\t0.00\t5.1\tfailed\tSwitched to SGD optimizer
jkl012\t0.88\t4.5\tsuccess\tLarger batch size 128
"""


class TestExperimentRecord:
    def test_record_creation(self):
        rec = ExperimentRecord(
            commit="abc123",
            metric_value=0.85,
            memory_gb=4.2,
            status="success",
            description="Test experiment",
        )
        assert rec.commit == "abc123"
        assert rec.metric_value == 0.85
        assert rec.memory_gb == 4.2
        assert rec.status == "success"
        assert rec.description == "Test experiment"


class TestExperimentParser:
    def test_parse_tsv_string(self):
        records = ExperimentParser.parse_tsv(SAMPLE_TSV)
        assert len(records) == 4
        assert records[0].commit == "abc123"
        assert records[0].metric_value == 0.85
        assert records[0].memory_gb == 4.2
        assert records[0].status == "success"
        assert records[0].description == "Increased learning rate to 0.01"

    def test_parse_tsv_file(self, tmp_path):
        tsv_file = tmp_path / "results.tsv"
        tsv_file.write_text(SAMPLE_TSV)
        records = ExperimentParser.parse_tsv_file(str(tsv_file))
        assert len(records) == 4

    def test_parse_failed_experiments(self):
        records = ExperimentParser.parse_tsv(SAMPLE_TSV)
        failed = [r for r in records if r.status == "failed"]
        assert len(failed) == 1
        assert failed[0].commit == "ghi789"

    def test_parse_empty_tsv(self):
        records = ExperimentParser.parse_tsv("commit\tmetric_value\tmemory_gb\tstatus\tdescription\n")
        assert len(records) == 0

    def test_parse_git_diff(self):
        diff = """diff --git a/train.py b/train.py
--- a/train.py
+++ b/train.py
@@ -10,7 +10,7 @@
-    lr = 0.001
+    lr = 0.01
"""
        context = ExperimentParser.parse_git_diff(diff)
        assert "train.py" in context["files_changed"]
        assert len(context["additions"]) > 0
        assert len(context["deletions"]) > 0

    def test_parse_git_diff_empty(self):
        context = ExperimentParser.parse_git_diff("")
        assert context["files_changed"] == []
        assert context["additions"] == []
        assert context["deletions"] == []
