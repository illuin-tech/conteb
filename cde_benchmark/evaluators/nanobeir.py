from cde_benchmark.embedders.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from cde_benchmark.formatters.data_formatter import BEIRDataFormatter


class NanoBEIR:
    def __init__(self, path, embedder, query_prompt="", doc_prompt=""):
        self.datasets = [
            "NanoClimateFEVER",
            "NanoDBPedia",
            "NanoFEVER",
            "NanoFiQA2018",
            "NanoHotpotQA",
            "NanoMSMARCO",
            "NanoNFCorpus",
            "NanoNQ",
            "NanoQuoraRetrieval",
            "NanoSCIDOCS",
            "NanoArguAna",
            "NanoSciFact",
            "NanoTouche2020",
        ]
        self.path = path
        self.embedder = embedder
        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt

    def run_task(self, task):
        formatter = BEIRDataFormatter(
            self.path + "/" + task, "train", query_prompt=self.query_prompt, doc_prompt=self.doc_prompt
        )
        metrics = self.embedder.compute_metrics_e2e(formatter)
        return metrics

    def run_all_tasks(self):
        results = {}
        for task in self.datasets:
            print(f"Running task: {task}")
            res = self.run_task(task)
            print(res)
            results[task] = res
        return results


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("nomic-ai/modernbert-embed-base")
    model._modules["0"].auto_model.config.reference_compile = False

    nanobeir = NanoBEIR("zeta-alpha-ai", SentenceTransformerEmbedder(model))
    results = nanobeir.run_all_tasks()
    print(results)
