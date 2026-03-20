# experiment-memory

**Memory for intelligent experimentation.**

experiment-memory gives your research agents long-term memory. It stores experiment records in a vector database so agents can query past results by semantic similarity before proposing new changes -- avoiding repeated mistakes and building on what worked.

## Quick Start

```bash
pip install -e .

# Ingest experiment results
experiment-memory ingest results.tsv

# Find similar past experiments
experiment-memory query "increased learning rate to 0.01"
```

## Features

- **Vector similarity search** -- find semantically similar past experiments using TF-IDF or sentence-transformers
- **Dual backends** -- lightweight in-memory store (SimpleMemoryStore) or production-grade ChromaDB
- **Results parser** -- automatically ingest `results.tsv` files and git diffs
- **Reasoning journal** -- capture and replay agent reasoning chains
- **Inspiration tracker** -- trace cross-pollination between experiments
- **Store statistics** -- monitor memory size, query patterns, and coverage

## Usage

### CLI

```bash
# Ingest experiment results
experiment-memory ingest results.tsv

# Query similar experiments
experiment-memory query "increased learning rate to 0.01"

# View the reasoning journal
experiment-memory journal

# Show inspiration chains for an experiment
experiment-memory inspire --experiment-id abc123

# Show store statistics
experiment-memory stats
```

### Python API

```python
from experiment_memory.store import SimpleMemoryStore

store = SimpleMemoryStore()
store.add_experiment(
    commit="abc123",
    description="Increased learning rate from 0.001 to 0.01",
    metric_value=0.85,
    metadata={"epoch": 10}
)

# Find the 5 most similar past experiments
results = store.query_similar("learning rate change", k=5)
for r in results:
    print(f"{r.description} -> {r.metric_value}")
```

### Production Setup with ChromaDB

```bash
pip install -e ".[chromadb]"
```

```python
from experiment_memory.store import ChromaDBStore

store = ChromaDBStore(persist_directory="./memory_db")
store.add_experiment(commit="abc123", description="...", metric_value=0.85)
```

## Architecture

| Component | Role |
|---|---|
| `SimpleMemoryStore` | In-memory store with TF-IDF cosine similarity |
| `ChromaDBStore` | Persistent store using ChromaDB + sentence-transformers |
| `ExperimentParser` | Parses results.tsv and git diffs into records |
| `JournalManager` | Tracks agent reasoning chains |
| `InspirationTracker` | Maps cross-pollination between experiments |

## License

MIT
