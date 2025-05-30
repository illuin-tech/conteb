import json
import sys
from datetime import datetime
from typing import List

import torch
import typer
from pylate.models import ColBERT
from sentence_transformers import SentenceTransformer

from cde_benchmark.embedders.base_embedder import Embedder
from cde_benchmark.embedders.colbert_embedder import ColBERTEmbedder
from cde_benchmark.embedders.contextual_embedder import (
    ContextualEmbedder,
    LateInteractionContextualEmbedder,
    MergedContextualEmbedder,
)
from cde_benchmark.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from cde_benchmark.evaluators.longembed import LongEmbedEvaluator
from cde_benchmark.evaluators.nanobeir import NanoBEIR
from cde_benchmark.formatters.data_formatter import DataFormatter

DATASETS = {
    "football-new": {
        "path": "football_new",
    },
    "cities": {},
    "chunked-mldr": {
        "split": "test",
    },
    "covid-qa": {},
    "eiopa_europa": {},
    "narrative_qa": {
        "split": "test",
    },
    "squad-chunked-par-100": {"path": "squad-chunked-par-100", "split": "validation"},
    "restaurants": {"path": "vidore-restaurants-esg", "split": "test"},
}

SQUAD_CHUNK_SIZES = [8000, 1000, 500, 300, 250, 200, 175, 150, 125, 100]
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def get_model(
    model_path: str, colbert_model: bool, model_type: str, use_prefix: bool, extend_context: bool
) -> Embedder:
    """
    Get the embedding model based on the type passed by the user.
    Args:
        base_model (SentenceTransformer): The base model to use.
        model_type (str): The type of model to use. Can be one of
          "contextual", "merged_contextual", "late_interaction", or "sentence_transformer".
        use_prefix (bool): Whether to add prefixes to the chunks and queries.
    """
    if colbert_model:
        # Load the ColBERT model
        if extend_context:
            base_model = ColBERT(model_path, device=DEVICE, document_length=sys.maxsize)
        else:
            base_model = ColBERT(model_path, device=DEVICE)
    else:
        base_model = SentenceTransformer(model_path, trust_remote_code=True, device=DEVICE)

    if extend_context:
        base_model.max_seq_length = sys.maxsize
        base_model.tokenizer.model_max_length = sys.maxsize

    match model_type:
        case "contextual":
            return ContextualEmbedder(base_model, add_prefix=use_prefix, device=DEVICE)
        case "merged_contextual":
            return MergedContextualEmbedder(base_model, add_prefix=use_prefix, device=DEVICE)
        case "late_interaction" | "li":
            return LateInteractionContextualEmbedder(base_model, add_prefix=use_prefix, device=DEVICE)
        case "sentence_transformer":
            return SentenceTransformerEmbedder(base_model, add_prefix=use_prefix, batch_size=128, device=DEVICE)
        case "colbert":
            if extend_context:
                batch_size = 1
            else:
                batch_size = 16
            return ColBERTEmbedder(base_model, batch_size=batch_size, device=DEVICE)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")


def eval_model(
    model_path: str,
    model_type: str,
    model_name: str,
    colbert_model: bool = False,
    datasets: List[str] = DATASETS.keys(),
    run_squad_chunked: bool = False,
    run_nanobeir: bool = False,
    run_longembed: bool = False,
    use_prefix: bool = False,
    data_base_path: str = ".",
    save_dir: str = "results",
    extend_context: bool = False,
):
    embedder = get_model(model_path, colbert_model, model_type, use_prefix, extend_context)

    output_dic = {
        "model_path": model_path,
        "model_name": model_name,
        "model_type": model_type,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "extended_context": extend_context,
        "use_prefix": use_prefix,
        "colbert_model": colbert_model,
    }

    metrics = {}

    for dataset in datasets:
        if dataset not in DATASETS:
            print(f"Dataset {dataset} not found in DATASETS. Skipping.")
            continue

        path = DATASETS[dataset].get("path", dataset)
        ds_path = f"{data_base_path}/{path}"
        split = DATASETS[dataset].get("split", "train")
        query_key = DATASETS[dataset].get("query_key", "queries")
        formatter = DataFormatter(ds_path, split, query_key=query_key)

        ds_metrics = embedder.compute_metrics_e2e(formatter)
        metrics[dataset] = ds_metrics

        # pretty print all results in terminal
        print("=" * 20)
        print(f"Dataset: {dataset}")
        print("=" * 20)

        for k, v in ds_metrics.items():
            print(f"\t{k}: {v}")

    # intermediary saving
    output_dic["metrics"] = metrics

    # Save the metrics to a JSON file
    output_name = f"metrics_{model_name}_{model_type}.json"
    output_path = f"{save_dir}/{output_name}"
    with open(output_path, "w") as f:
        json.dump(output_dic, f, indent=4)

    if run_squad_chunked:
        for chunk_size in SQUAD_CHUNK_SIZES:
            ds_path = f"{data_base_path}/squad-chunked-par-{chunk_size}"
            formatter = DataFormatter(ds_path, "validation", query_key="queries")
            ds_metrics = embedder.compute_metrics_e2e(formatter)
            metrics[f"squad-chunked-par{chunk_size}"] = ds_metrics

            # pretty print all results in terminal
            print("=" * 20)
            print(f"Dataset: squad-chunked-par-{chunk_size}")
            print("=" * 20)

            for k, v in ds_metrics.items():
                print(f"\t{k}: {v}")

    if run_nanobeir:
        nanobeir = NanoBEIR(f"{data_base_path}/nanobeir", embedder)
        nanobeir_metrics = nanobeir.run_all_tasks()
        metrics.update(nanobeir_metrics)

    if run_longembed:
        print("Running LongEmbed tasks...")
        longembed = LongEmbedEvaluator(
            f"{data_base_path}/LongEmbed",
            embedder,
        )
        longembed_metrics = longembed.run_all_tasks()
        metrics.update(longembed_metrics)

    output_dic["metrics"] = metrics

    # Save the metrics to a JSON file
    output_name = f"metrics_{model_name}_{model_type}.json"
    output_path = f"{save_dir}/{output_name}"
    with open(output_path, "w") as f:
        json.dump(output_dic, f, indent=4)
    print(f"Metrics saved to {output_path}!")


if __name__ == "__main__":
    typer.run(eval_model)
