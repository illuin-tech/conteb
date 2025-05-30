import numpy as np
from sentence_transformers import SentenceTransformer

from cde_benchmark.embedders.base_embedder import Embedder


class StatsEmbedder(Embedder):
    def __init__(
        self,
        model: SentenceTransformer = None,
        batch_size: int = 16,
        show_progress_bar: bool = True,
    ):
        super().__init__(is_contextual_model=True)
        self.model: SentenceTransformer = model
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.sep_token = self.model.tokenizer.sep_token

    def embed_queries(self, queries):
        raise NotImplementedError

    def embed_documents(self, documents):
        raise NotImplementedError

    def stats_documents(self, documents):
        # using the tokenizer to get the number of tokens in each document
        num_tokens = []
        for doc in documents:
            # if doc is a list of documents, tokenize and count all of them
            if isinstance(doc, list):
                num_tokens_doc = [len(self.model.tokenizer.tokenize(d)) for d in doc]
                # sum the number of tokens in each document
                num_tokens.append(
                    {
                        "num_tokens": sum(num_tokens_doc),
                        "num_tokens_per_chunk": np.mean(num_tokens_doc),
                        "num_chunks": len(num_tokens_doc),
                    }
                )
            else:
                raise NotImplementedError
        # compute averaged stats
        stats = {}
        stats["num_docs"] = len(documents)
        stats["average_num_tokens_per_doc"] = np.mean([d["num_tokens"] for d in num_tokens])
        stats["average_num_tokens_per_chunk"] = np.mean([d["num_tokens_per_chunk"] for d in num_tokens])
        stats["average_num_chunks_per_doc"] = np.mean([d["num_chunks"] for d in num_tokens])
        return stats

    def stats_queries(self, queries):
        # using the tokenizer to get the number of tokens in each document
        num_tokens = []
        for doc in queries:
            # if doc is a list of documents, tokenize and count all of them
            num_tokens_doc = len(self.model.tokenizer.tokenize(doc))
            # sum the number of tokens in each document
            num_tokens.append({"num_tokens": num_tokens_doc})
        # compute averaged stats
        stats = {}
        stats["num_queries"] = len(queries)
        stats["average_num_tokens_per_query"] = np.mean([d["num_tokens"] for d in num_tokens])
        return stats

    def process_queries(self, data_formatter):
        if self.use_nested_queries:
            queries, document_ids = data_formatter.get_nested_queries()
            document_ids = [id_ for nested_ids in document_ids for id_ in nested_ids]
        else:
            queries, document_ids = data_formatter.get_queries()

        # compute stats
        stats = self.stats_queries(queries)
        # make into a contiguous tensor, and map position to document_ids
        return stats

    def process_documents(self, data_formatter):
        if self.merge_embeddings:
            raise NotImplementedError
        else:
            if self.is_contextual_model:
                documents, document_ids = data_formatter.get_nested()

                # compute stats
                stats = self.stats_documents(documents)

            else:
                documents, document_ids = data_formatter.get_flattened()
                # compute stats
                raise NotImplementedError

        # make into a contiguous tensor, and map position to document_ids
        return stats

    def get_similarities(self, query_embeddings, doc_embeddings):
        raise NotImplementedError

    def get_metrics(self, scores, all_document_ids, label_documents_id, **kwargs):
        raise NotImplementedError

    def compute_metrics_e2e(self, data_formatter, **kwargs):
        stats_queries = self.process_queries(data_formatter)
        stats_docs = self.process_documents(data_formatter)

        # concat both dicts
        stats = {**stats_queries, **stats_docs}
        return stats
