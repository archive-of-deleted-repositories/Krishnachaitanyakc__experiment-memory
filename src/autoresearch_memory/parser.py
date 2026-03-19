"""Parse results.tsv files and git diffs."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ExperimentRecord:
    """A single experiment record from results.tsv."""

    commit: str
    metric_value: float
    memory_gb: float
    status: str
    description: str


class ExperimentParser:
    """Parse experiment results from TSV files and git diffs."""

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
    def parse_tsv_file(filepath: str) -> list[ExperimentRecord]:
        """Parse a TSV file into experiment records."""
        with open(filepath, "r") as f:
            return ExperimentParser.parse_tsv(f.read())

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
