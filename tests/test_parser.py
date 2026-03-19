"""Tests for results parser."""

import json

import pytest
from autoresearch_memory.parser import ExperimentParser, ExperimentRecord


SAMPLE_TSV = """commit\tmetric_value\tmemory_gb\tstatus\tdescription
abc123\t0.85\t4.2\tsuccess\tIncreased learning rate to 0.01
def456\t0.82\t3.8\tsuccess\tAdded dropout regularization
ghi789\t0.00\t5.1\tfailed\tSwitched to SGD optimizer
jkl012\t0.88\t4.5\tsuccess\tLarger batch size 128
"""

SAMPLE_JSON = json.dumps([
    {"commit": "abc123", "metric_value": 0.85, "memory_gb": 4.2, "status": "success", "description": "Increased learning rate to 0.01"},
    {"commit": "def456", "metric_value": 0.82, "memory_gb": 3.8, "status": "success", "description": "Added dropout regularization"},
])

SAMPLE_CSV = """commit,metric_value,memory_gb,status,description
abc123,0.85,4.2,success,Increased learning rate to 0.01
def456,0.82,3.8,success,Added dropout regularization
ghi789,0.00,5.1,failed,Switched to SGD optimizer
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

    def test_parse_json_string(self):
        records = ExperimentParser.parse_json(SAMPLE_JSON)
        assert len(records) == 2
        assert records[0].commit == "abc123"
        assert records[0].metric_value == 0.85
        assert records[1].description == "Added dropout regularization"

    def test_parse_json_file(self, tmp_path):
        json_file = tmp_path / "results.json"
        json_file.write_text(SAMPLE_JSON)
        records = ExperimentParser.parse_json_file(str(json_file))
        assert len(records) == 2

    def test_parse_json_single_object(self):
        single = json.dumps({"commit": "abc", "metric_value": 0.9, "description": "test"})
        records = ExperimentParser.parse_json(single)
        assert len(records) == 1

    def test_parse_csv_string(self):
        records = ExperimentParser.parse_csv(SAMPLE_CSV)
        assert len(records) == 3
        assert records[0].commit == "abc123"
        assert records[0].metric_value == 0.85
        assert records[2].status == "failed"

    def test_parse_csv_file(self, tmp_path):
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(SAMPLE_CSV)
        records = ExperimentParser.parse_csv_file(str(csv_file))
        assert len(records) == 3

    def test_parse_file_auto_detect_tsv(self, tmp_path):
        tsv_file = tmp_path / "results.tsv"
        tsv_file.write_text(SAMPLE_TSV)
        records = ExperimentParser.parse_file(str(tsv_file))
        assert len(records) == 4

    def test_parse_file_auto_detect_json(self, tmp_path):
        json_file = tmp_path / "results.json"
        json_file.write_text(SAMPLE_JSON)
        records = ExperimentParser.parse_file(str(json_file))
        assert len(records) == 2

    def test_parse_file_auto_detect_csv(self, tmp_path):
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(SAMPLE_CSV)
        records = ExperimentParser.parse_file(str(csv_file))
        assert len(records) == 3

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
