"""Tests for CLI."""

import json

import pytest
from click.testing import CliRunner
from experiment_memory.cli import cli


SAMPLE_TSV = """commit\tmetric_value\tmemory_gb\tstatus\tdescription
abc123\t0.85\t4.2\tsuccess\tIncreased learning rate to 0.01
def456\t0.82\t3.8\tsuccess\tAdded dropout regularization
"""

SAMPLE_JSON = json.dumps([
    {"commit": "abc123", "metric_value": 0.85, "memory_gb": 4.2, "status": "success", "description": "Increased learning rate to 0.01"},
    {"commit": "def456", "metric_value": 0.82, "memory_gb": 3.8, "status": "success", "description": "Added dropout regularization"},
])

SAMPLE_CSV = """commit,metric_value,memory_gb,status,description
abc123,0.85,4.2,success,Increased learning rate to 0.01
def456,0.82,3.8,success,Added dropout regularization
"""


class TestCLI:
    def setup_method(self):
        self.runner = CliRunner()

    def test_cli_help(self):
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "autoresearch-memory" in result.output.lower() or "experiment" in result.output.lower()

    def test_ingest_command(self, tmp_path):
        tsv_file = tmp_path / "results.tsv"
        tsv_file.write_text(SAMPLE_TSV)
        result = self.runner.invoke(cli, ["ingest", str(tsv_file)])
        assert result.exit_code == 0
        assert "2" in result.output or "ingest" in result.output.lower()

    def test_ingest_json(self, tmp_path):
        json_file = tmp_path / "results.json"
        json_file.write_text(SAMPLE_JSON)
        result = self.runner.invoke(cli, ["ingest", str(json_file)])
        assert result.exit_code == 0
        assert "2" in result.output

    def test_ingest_csv(self, tmp_path):
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(SAMPLE_CSV)
        result = self.runner.invoke(cli, ["ingest", str(csv_file)])
        assert result.exit_code == 0
        assert "2" in result.output

    def test_ingest_dedup(self, tmp_path):
        tsv_file = tmp_path / "results.tsv"
        tsv_file.write_text(SAMPLE_TSV)
        result1 = self.runner.invoke(cli, ["ingest", str(tsv_file)])
        assert result1.exit_code == 0
        result2 = self.runner.invoke(cli, ["ingest", "--dedup", str(tsv_file)])
        assert result2.exit_code == 0

    def test_query_command(self, tmp_path):
        tsv_file = tmp_path / "results.tsv"
        tsv_file.write_text(SAMPLE_TSV)
        self.runner.invoke(cli, ["ingest", str(tsv_file)])
        result = self.runner.invoke(cli, ["query", "learning rate"])
        assert result.exit_code == 0

    def test_stats_command(self, tmp_path):
        tsv_file = tmp_path / "results.tsv"
        tsv_file.write_text(SAMPLE_TSV)
        self.runner.invoke(cli, ["ingest", str(tsv_file)])
        result = self.runner.invoke(cli, ["stats"])
        assert result.exit_code == 0

    def test_journal_command(self):
        result = self.runner.invoke(cli, ["journal"])
        assert result.exit_code == 0

    def test_inspire_command(self):
        result = self.runner.invoke(cli, ["inspire", "--experiment-id", "exp1"])
        assert result.exit_code == 0

    def test_ingest_missing_file(self):
        result = self.runner.invoke(cli, ["ingest", "/nonexistent/file.tsv"])
        assert result.exit_code != 0 or "error" in result.output.lower() or "not found" in result.output.lower()

    def test_export_command(self, tmp_path):
        tsv_file = tmp_path / "results.tsv"
        tsv_file.write_text(SAMPLE_TSV)
        self.runner.invoke(cli, ["ingest", str(tsv_file)])
        export_file = str(tmp_path / "export.json")
        result = self.runner.invoke(cli, ["export", export_file])
        assert result.exit_code == 0
        assert "exported" in result.output.lower()

    def test_import_command(self, tmp_path):
        tsv_file = tmp_path / "results.tsv"
        tsv_file.write_text(SAMPLE_TSV)
        self.runner.invoke(cli, ["ingest", str(tsv_file)])
        export_file = str(tmp_path / "export.json")
        self.runner.invoke(cli, ["export", export_file])
        result = self.runner.invoke(cli, ["import", export_file])
        assert result.exit_code == 0
        assert "imported" in result.output.lower()

    def test_suggest_command(self, tmp_path):
        tsv_file = tmp_path / "results.tsv"
        tsv_file.write_text(SAMPLE_TSV)
        self.runner.invoke(cli, ["ingest", str(tsv_file)])
        result = self.runner.invoke(cli, ["suggest"])
        assert result.exit_code == 0

    def test_suggest_empty(self):
        result = self.runner.invoke(cli, ["suggest"])
        assert result.exit_code == 0

    def test_project_flag(self, tmp_path):
        tsv_file = tmp_path / "results.tsv"
        tsv_file.write_text(SAMPLE_TSV)
        result = self.runner.invoke(cli, ["--project", "testproj", "ingest", str(tsv_file)])
        assert result.exit_code == 0
        assert "2" in result.output
