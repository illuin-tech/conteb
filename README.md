# ConTEB: Context is Gold to find the Gold Passage: Evaluating and Training Contextual Document Embeddings

This repository contains evaluation code released with our preprint [*Context is Gold to find the Gold Passage: Evaluating and Training Contextual Document Embeddings*](https://arxiv.org/abs/2505.24782).


[![arXiv](https://img.shields.io/badge/arXiv-2505.24782-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2505.24782)

<img src="https://cdn-uploads.huggingface.co/production/uploads/60f2e021adf471cbdf8bb660/jq_zYRy23bOZ9qey3VY4v.png" width="800">

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

### Abstract

A limitation of modern document retrieval embedding methods is that they typically encode passages (chunks) from the same documents independently, often overlooking crucial contextual information from the rest of the document that could greatly improve individual chunk representations.

In this work, we introduce *ConTEB* (Context-aware Text Embedding Benchmark), a benchmark designed to evaluate retrieval models on their ability to leverage document-wide context. Our results show that state-of-the-art embedding models struggle in retrieval scenarios where context is required. To address this limitation, we propose *InSeNT* (In-sequence Negative Training), a novel contrastive post-training approach which combined with \textit{late chunking} pooling enhances contextual representation learning while preserving computational efficiency. Our method significantly improves retrieval quality on *ConTEB* without sacrificing base model performance. 
We further find chunks embedded with our method are more robust to suboptimal chunking strategies and larger retrieval corpus sizes.
We open-source all artifacts here and at https://github.com/illuin-tech/contextual-embeddings.

## Ressources

- [*HuggingFace Project Page*](https://huggingface.co/illuin-conteb): The HF page centralizing everything!
- [*(Model) ModernBERT*](https://huggingface.co/illuin-conteb/modernbert-large-insent): The Contextualized ModernBERT bi-encoder trained with InSENT loss and Late Chunking
- [*(Model) ModernColBERT*](https://huggingface.co/illuin-conteb/modern-colbert-insent): The Contextualized ModernColBERT trained with InSENT loss and Late Chunking
- [*Leaderboard*](TODO): Coming Soon
- [*(Data) ConTEB Benchmark Datasets*]([TODO](https://huggingface.co/collections/illuin-conteb/conteb-evaluation-datasets-6839fffd25f1d3685f3ad604)): Datasets included in ConTEB.
- [*(Code) Contextual Document Engine*](https://github.com/illuin-tech/contextual-embeddings): The code used to train and run inference with our architecture.
- [*(Code) ConTEB Benchmarkk*](https://github.com/illuin-tech/conteb): A Python package/CLI tool to evaluate document retrieval systems on the ConTEB benchmark.
- [*Preprint*](https://arxiv.org/abs/2505.24782): The paper with all details!
- [*Blog*](https://huggingface.co/blog/manu/conteb): A blogpost that covers the paper in a 5 minute read.

## Contact of the first authors

- Manuel Faysse: manuel.faysse@illuin.tech
- Max Conti: max.conti@illuin.tech

## Citation

If you use any datasets or models from this organization in your research, please cite the original dataset as follows:

```latex
@misc{conti2025contextgoldgoldpassage,
      title={Context is Gold to find the Gold Passage: Evaluating and Training Contextual Document Embeddings}, 
      author={Max Conti and Manuel Faysse and Gautier Viaud and Antoine Bosselut and Céline Hudelot and Pierre Colombo},
      year={2025},
      eprint={2505.24782},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2505.24782}, 
}
```

## Acknowledgments

This work is partially supported by [ILLUIN Technology](https://www.illuin.tech/), and by a grant from ANRT France.
This work was performed using HPC resources from the GENCI Jeanzay supercomputer with grant AD011016393.

