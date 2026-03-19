"""CLI for autoresearch-memory."""

from __future__ import annotations

import os

import click

from autoresearch_memory.inspiration import InspirationTracker
from autoresearch_memory.journal import JournalManager
from autoresearch_memory.parser import ExperimentParser
from autoresearch_memory.store import SimpleMemoryStore

# Global state for CLI session
_store = SimpleMemoryStore()
_journal = JournalManager()
_inspiration = InspirationTracker()


@click.group()
def cli() -> None:
    """autoresearch-memory: Query and explore experiment history."""


@cli.command()
@click.argument("filepath")
def ingest(filepath: str) -> None:
    """Ingest a results.tsv file into the experiment store."""
    if not os.path.exists(filepath):
        click.echo(f"Error: File not found: {filepath}")
        raise SystemExit(1)

    records = ExperimentParser.parse_tsv_file(filepath)
    for rec in records:
        _store.add_experiment(
            commit=rec.commit,
            description=rec.description,
            metric_value=rec.metric_value,
            metadata={"memory_gb": rec.memory_gb, "status": rec.status},
        )
    click.echo(f"Ingested {len(records)} experiments from {filepath}")


@cli.command()
@click.argument("query_text")
@click.option("-k", "--top-k", default=5, help="Number of results to return")
def query(query_text: str, top_k: int) -> None:
    """Search for similar experiments."""
    results = _store.query_similar(query_text, k=top_k)
    if not results:
        click.echo("No experiments found.")
        return
    for r in results:
        click.echo(
            f"[{r['commit']}] metric={r['metric_value']:.4f} | {r['description']}"
        )


@cli.command()
@click.option("--search", default=None, help="Search term for journal")
def journal(search: str | None) -> None:
    """View or search the reasoning journal."""
    if search:
        entries = _journal.search_journal(search)
    else:
        entries = _journal.get_journal()

    if not entries:
        click.echo("No journal entries found.")
        return
    for e in entries:
        click.echo(
            f"[{e['experiment_id']}] {e['reasoning']} -> {e['outcome']}"
        )


@cli.command()
@click.option("--experiment-id", required=True, help="Experiment ID")
def inspire(experiment_id: str) -> None:
    """Show inspiration chain for an experiment."""
    chain = _inspiration.get_inspiration_chain(experiment_id)
    if not chain:
        click.echo(f"No inspiration chain found for {experiment_id}")
        return
    click.echo(f"Inspiration chain for {experiment_id}:")
    for exp_id in chain:
        click.echo(f"  <- {exp_id}")


@cli.command()
def stats() -> None:
    """Show experiment store statistics."""
    s = _store.get_stats()
    click.echo(f"Total experiments: {s['count']}")
    if s["count"] > 0:
        click.echo(f"Metric range: [{s['min_metric']:.4f}, {s['max_metric']:.4f}]")
        click.echo(f"Average metric: {s['avg_metric']:.4f}")


if __name__ == "__main__":
    cli()
