"""Tests for CLI."""

import pytest
from click.testing import CliRunner
from autoresearch_memory.cli import cli


SAMPLE_TSV = """commit\tmetric_value\tmemory_gb\tstatus\tdescription
abc123\t0.85\t4.2\tsuccess\tIncreased learning rate to 0.01
def456\t0.82\t3.8\tsuccess\tAdded dropout regularization
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
