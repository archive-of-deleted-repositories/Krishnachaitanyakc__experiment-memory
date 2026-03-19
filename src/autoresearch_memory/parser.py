"""Parse results files (TSV, JSON, CSV) and git diffs."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass


@dataclass
class ExperimentRecord:
    """A single experiment record from results files."""

    commit: str
    metric_value: float
    memory_gb: float
    status: str
    description: str


class ExperimentParser:
    """Parse experiment results from TSV, JSON, CSV files and git diffs."""

    @staticmethod
    def parse_tsv(tsv_content: str) -> list[ExperimentRecord]:
        """Parse a TSV string into experiment records."""
        lines = tsv_content.strip().split("\n")
        if len(lines) <= 1:
            return []

        records = []
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            records.append(
                ExperimentRecord(
                    commit=parts[0].strip(),
                    metric_value=float(parts[1].strip()),
                    memory_gb=float(parts[2].strip()),
                    status=parts[3].strip(),
                    description=parts[4].strip(),
                )
            )
        return records

    @staticmethod
    def parse_json(json_content: str) -> list[ExperimentRecord]:
        """Parse a JSON string into experiment records.

        Expects a JSON array of objects with keys:
        commit, metric_value, memory_gb, status, description.
        """
        data = json.loads(json_content)
        if not isinstance(data, list):
            data = [data]
        records = []
        for item in data:
            records.append(
                ExperimentRecord(
                    commit=str(item["commit"]),
                    metric_value=float(item["metric_value"]),
                    memory_gb=float(item.get("memory_gb", 0.0)),
                    status=str(item.get("status", "unknown")),
                    description=str(item["description"]),
                )
            )
        return records

    @staticmethod
    def parse_csv(csv_content: str) -> list[ExperimentRecord]:
        """Parse a CSV string into experiment records.

        Expects headers: commit, metric_value, memory_gb, status, description.
        """
        reader = csv.DictReader(io.StringIO(csv_content))
        records = []
        for row in reader:
            records.append(
                ExperimentRecord(
                    commit=row["commit"].strip(),
                    metric_value=float(row["metric_value"].strip()),
                    memory_gb=float(row.get("memory_gb", "0").strip()),
                    status=row.get("status", "unknown").strip(),
                    description=row["description"].strip(),
                )
            )
        return records

    @staticmethod
    def parse_tsv_file(filepath: str) -> list[ExperimentRecord]:
        """Parse a TSV file into experiment records."""
        with open(filepath, "r") as f:
            return ExperimentParser.parse_tsv(f.read())

    @staticmethod
    def parse_json_file(filepath: str) -> list[ExperimentRecord]:
        """Parse a JSON file into experiment records."""
        with open(filepath, "r") as f:
            return ExperimentParser.parse_json(f.read())

    @staticmethod
    def parse_csv_file(filepath: str) -> list[ExperimentRecord]:
        """Parse a CSV file into experiment records."""
        with open(filepath, "r") as f:
            return ExperimentParser.parse_csv(f.read())

    @staticmethod
    def parse_file(filepath: str) -> list[ExperimentRecord]:
        """Auto-detect format from file extension and parse."""
        if filepath.endswith(".json"):
            return ExperimentParser.parse_json_file(filepath)
        elif filepath.endswith(".csv"):
            return ExperimentParser.parse_csv_file(filepath)
        else:
            return ExperimentParser.parse_tsv_file(filepath)

    @staticmethod
    def parse_git_diff(diff_text: str) -> dict:
        """Parse a git diff and extract context."""
        if not diff_text.strip():
            return {"files_changed": [], "additions": [], "deletions": []}

        files_changed = re.findall(r"diff --git a/(.+?) b/", diff_text)
        additions = [
            line[1:].strip()
            for line in diff_text.split("\n")
            if line.startswith("+") and not line.startswith("+++")
        ]
        deletions = [
            line[1:].strip()
            for line in diff_text.split("\n")
            if line.startswith("-") and not line.startswith("---")
        ]

        return {
            "files_changed": files_changed,
            "additions": additions,
            "deletions": deletions,
        }
