"""CLI for autoresearch-memory."""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

import click

from autoresearch_memory.inspiration import InspirationTracker
from autoresearch_memory.journal import JournalManager
from autoresearch_memory.parser import ExperimentParser
from autoresearch_memory.store import SimpleMemoryStore


def _get_project_dir(project: str) -> str:
    """Get the project directory path."""
    base = os.path.join(os.path.expanduser("~"), ".autoresearch", "projects", project)
    os.makedirs(base, exist_ok=True)
    return base


def _get_store(project: str) -> SimpleMemoryStore:
    """Get a store for the given project."""
    project_dir = _get_project_dir(project)
    jsonl_path = os.path.join(project_dir, "experiments.jsonl")
    return SimpleMemoryStore(jsonl_path=jsonl_path)


# Global state for CLI session (used when --project is not specified for backward compat)
_store = SimpleMemoryStore()
_journal = JournalManager()
_inspiration = InspirationTracker()


@click.group()
@click.option("--project", default=None, help="Project name for multi-project stores")
@click.pass_context
def cli(ctx: click.Context, project: str | None) -> None:
    """autoresearch-memory: Query and explore experiment history."""
    ctx.ensure_object(dict)
    if project:
        ctx.obj["store"] = _get_store(project)
        ctx.obj["project"] = project
    else:
        ctx.obj["store"] = _store
        ctx.obj["project"] = None


@cli.command()
@click.argument("filepath")
@click.option("--dedup", is_flag=True, default=False, help="Skip duplicate commits")
@click.pass_context
def ingest(ctx: click.Context, filepath: str, dedup: bool) -> None:
    """Ingest a results file into the experiment store. Auto-detects format from extension."""
    if not os.path.exists(filepath):
        click.echo(f"Error: File not found: {filepath}")
        raise SystemExit(1)

    records = ExperimentParser.parse_file(filepath)
    store = ctx.obj["store"]
    added = 0
    for rec in records:
        old_count = len(store.get_history(n=999999))
        store.add_experiment(
            commit=rec.commit,
            description=rec.description,
            metric_value=rec.metric_value,
            metadata={"memory_gb": rec.memory_gb, "status": rec.status},
            dedup=dedup,
        )
        new_count = len(store.get_history(n=999999))
        if new_count > old_count:
            added += 1
    click.echo(f"Ingested {added} experiments from {filepath}")


@cli.command()
@click.argument("query_text")
@click.option("-k", "--top-k", default=5, help="Number of results to return")
@click.pass_context
def query(ctx: click.Context, query_text: str, top_k: int) -> None:
    """Search for similar experiments."""
    store = ctx.obj["store"]
    results = store.query_similar(query_text, k=top_k)
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
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show experiment store statistics."""
    store = ctx.obj["store"]
    s = store.get_stats()
    click.echo(f"Total experiments: {s['count']}")
    if s["count"] > 0:
        click.echo(f"Metric range: [{s['min_metric']:.4f}, {s['max_metric']:.4f}]")
        click.echo(f"Average metric: {s['avg_metric']:.4f}")


@cli.command(name="export")
@click.argument("path")
@click.pass_context
def export_cmd(ctx: click.Context, path: str) -> None:
    """Export experiment store to a JSON file."""
    store = ctx.obj["store"]
    store.export_store(path)
    click.echo(f"Exported store to {path}")


@cli.command(name="import")
@click.argument("path")
@click.pass_context
def import_cmd(ctx: click.Context, path: str) -> None:
    """Import experiment store from a JSON file."""
    if not os.path.exists(path):
        click.echo(f"Error: File not found: {path}")
        raise SystemExit(1)
    store = ctx.obj["store"]
    store.import_store(path)
    click.echo(f"Imported store from {path}")


@cli.command()
@click.pass_context
def summarize(ctx: click.Context) -> None:
    """Summarize experiment results using Claude LLM."""
    store = ctx.obj["store"]
    experiments = store.get_history(n=999999)
    if not experiments:
        click.echo("No experiments to summarize.")
        return

    try:
        import anthropic
    except ImportError:
        click.echo("Error: anthropic package not installed. Run: pip install anthropic")
        raise SystemExit(1)

    exp_text = "\n".join(
        f"- [{e['commit']}] metric={e['metric_value']:.4f} | {e['description']}"
        for e in experiments
    )

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize these experiment results focusing on key trends "
                    "and findings. Output markdown.\n\n" + exp_text
                ),
            }
        ],
    )
    click.echo(message.content[0].text)


@cli.command()
@click.pass_context
def suggest(ctx: click.Context) -> None:
    """Suggest next experiment based on historical success patterns."""
    store = ctx.obj["store"]
    experiments = store.get_history(n=999999)
    if not experiments:
        click.echo("No experiments to analyze.")
        return

    _print_suggestions(experiments)


def _analyze_change_types(experiments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Analyze which change types led to improvements most often."""
    # Simple keyword-based change type classification
    change_keywords = {
        "learning_rate": ["learning rate", "lr"],
        "regularization": ["dropout", "regularization", "weight decay", "l2", "l1"],
        "optimizer": ["optimizer", "adam", "sgd", "adamw"],
        "architecture": ["layer", "hidden", "attention", "transformer", "conv"],
        "batch_size": ["batch size", "batch"],
        "data": ["data", "augment", "preprocess", "dataset"],
        "training": ["epoch", "schedule", "warmup", "training"],
    }

    type_metrics: dict[str, list[float]] = {}
    for exp in experiments:
        desc = exp["description"].lower()
        for change_type, keywords in change_keywords.items():
            if any(kw in desc for kw in keywords):
                type_metrics.setdefault(change_type, []).append(exp["metric_value"])

    results: dict[str, dict[str, Any]] = {}
    for change_type, metrics in type_metrics.items():
        results[change_type] = {
            "count": len(metrics),
            "avg_metric": sum(metrics) / len(metrics),
            "max_metric": max(metrics),
        }
    return results


def _print_suggestions(experiments: list[dict[str, Any]]) -> None:
    """Print suggestions based on analysis."""
    analysis = _analyze_change_types(experiments)
    if not analysis:
        click.echo("Not enough data to make suggestions.")
        return

    # Sort by average metric descending
    sorted_types = sorted(
        analysis.items(), key=lambda x: x[1]["avg_metric"], reverse=True
    )

    click.echo("Suggested experiment types (by average metric):")
    for change_type, stats in sorted_types:
        click.echo(
            f"  {change_type}: avg={stats['avg_metric']:.4f} "
            f"max={stats['max_metric']:.4f} (n={stats['count']})"
        )

    best = sorted_types[0]
    click.echo(f"\nRecommendation: Try more '{best[0]}' experiments "
               f"(avg metric: {best[1]['avg_metric']:.4f})")


if __name__ == "__main__":
    cli()
