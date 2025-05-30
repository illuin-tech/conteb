import time

import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

from cde_benchmark.embedders.base_embedder import Embedder


class BM25Embedder(Embedder):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(is_contextual_model=False)
        self.tokenizer = tokenizer

    def compute_metrics_e2e(self, data_formatter, **kwargs):
        queries, query_ids = data_formatter.get_queries()
        documents, doc_ids = data_formatter.get_flattened()
        scores = []
        start_time = time.time()
        tokenized_corpus = [self.tokenizer(doc, return_tensors="pt")["input_ids"][0].numpy() for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        # bm25 = BM25Okapi(documents, tokenizer=self.tokenizer.encode)
        end_time = time.time()
        for query in queries:
            tokenized_query = self.tokenizer(query, return_tensors="pt")["input_ids"][0].numpy()
            query_scores = bm25.get_scores(tokenized_query)
            scores.append(query_scores)

        scores = np.array(scores)
        metrics = self.get_metrics(scores, doc_ids, query_ids)
        metrics["runtime"] = end_time - start_time
        return metrics
