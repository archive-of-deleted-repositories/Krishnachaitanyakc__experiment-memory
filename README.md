# autoresearch-memory

Vector DB storing experiment history so agents can query past results before proposing changes.

## Features

- Store and query experiment records with vector similarity search
- Parse `results.tsv` files from autoresearch runs
- Journal manager for capturing agent reasoning chains
- Inspiration tracker for cross-pollination between experiments
- CLI for ingesting, querying, and exploring experiment history

## Installation

```bash
pip install -e ".[dev]"
```

For production use with ChromaDB:

```bash
pip install -e ".[chromadb]"
```

## Usage

### CLI

```bash
# Ingest experiment results
autoresearch-memory ingest results.tsv

# Query similar experiments
autoresearch-memory query "increased learning rate to 0.01"

# View reasoning journal
autoresearch-memory journal

# Show inspiration chains
autoresearch-memory inspire --experiment-id abc123

# Show store statistics
autoresearch-memory stats
```

### Python API

```python
from autoresearch_memory.store import SimpleMemoryStore

store = SimpleMemoryStore()
store.add_experiment(
    commit="abc123",
    description="Increased learning rate from 0.001 to 0.01",
    metric_value=0.85,
    metadata={"epoch": 10}
)

results = store.query_similar("learning rate change", k=5)
```

## Architecture

- `BaseExperimentStore` - Abstract base class defining the store interface
- `SimpleMemoryStore` - In-memory implementation using TF-IDF cosine similarity
- `ChromaDBStore` - Production implementation using ChromaDB + sentence-transformers
- `ExperimentParser` - Parses results.tsv and git diffs
- `JournalManager` - Tracks reasoning chains
- `InspirationTracker` - Tracks experiment cross-pollination
