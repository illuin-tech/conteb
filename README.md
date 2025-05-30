# Contextualized Embeddings Benchmark

This repository contains evaluation code for the "Contextualized Embeddings" project.

## Installation
```bash
pip install -e .
```

## Usage

Refer to `scripts/evaluation/eval_conteb.py` for an example of how to use the code.

```python
from cde_benchmark.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from cde_benchmark.embedders.contextual_embedder import ContextualEmbedder
from cde_benchmark.formatters.data_formatter import DataFormatter

# Datasets should be correctly formatted
formatter = DataFormatter("illuin-cde/chunked-mldr", split="test")

# Non-nested example
embedder = SentenceTransformerEmbedder("nomic-ai/modernbert-embed-base")
metrics = embedder.compute_metrics_e2e(formatter)
print(metrics)

# Nested example (for conxtualized embeddings models)
embedder = NaiveContextualEmbedder("nomic-ai/modernbert-embed-base")
metrics = embedder.compute_metrics_e2e(formatter)
print(metrics)
```