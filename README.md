# ConTEB

This repository contains evaluation code for the "Contextual Embeddings" project.

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
formatter = DataFormatter("illuin-cde/cities", split="test")

base_model = SentenceTransformer(model_path)

# Non-nested example
embedder = SentenceTransformerEmbedder(base_model)
metrics = embedder.compute_metrics_e2e(formatter)
print(metrics)

# Nested example (for contextualized embeddings models)
embedder = ContextualEmbedder(base_model)
metrics = embedder.compute_metrics_e2e(formatter)
print(metrics)
```
