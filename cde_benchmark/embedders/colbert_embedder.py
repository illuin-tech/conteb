import torch
from pylate.models import ColBERT
from tqdm import tqdm

from cde_benchmark.embedders.base_embedder import Embedder


class ColBERTEmbedder(Embedder):
    def __init__(
        self,
        model: ColBERT,
        batch_size: int = 128,
        device: str = "cpu",
    ):
        super().__init__(is_contextual_model=False, device=device)
        self.model = model
        self.batch_size = batch_size

    def embed_queries(self, queries):
        outputs = self.model.encode(
            queries,
            show_progress_bar=True,
            convert_to_tensor=True,
            batch_size=self.batch_size,
        )
        # outputs = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True)
        return outputs

    def embed_documents(self, documents):
        outputs = self.model.encode(
            documents,
            show_progress_bar=True,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            is_query=False,
        )
        # outputs = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True)
        return outputs

    def compute_max_sim_scores(
        self,
        q_token_embeddings: torch.Tensor,
        d_token_embeddings: torch.Tensor,
    ):
        # normalize the embeddings
        q_token_embeddings = torch.nn.functional.normalize(q_token_embeddings, p=2, dim=-1)
        d_token_embeddings = torch.nn.functional.normalize(d_token_embeddings, p=2, dim=-1)

        # perform dot product between query and document embeddings
        sim_scores = torch.einsum(
            "qnd,bmd->qbnm", q_token_embeddings, d_token_embeddings
        )  # (n_queries, n_docs, max_q_len, max_doc_len)

        # max_sim: take the max over doc embeddings, then sum over query embeddings
        max_sim_scores = sim_scores.max(dim=3)[0].sum(dim=2)
        return max_sim_scores  # (n_queries, n_docs)

    def compute_single_scores(self, query_embeddings, doc_embeddings):
        all_scores = []
        q_scores = []
        # breakpoint()
        for q_embedding in tqdm(query_embeddings, desc="Computing scores for each query", leave=False):
            for d_embedding in doc_embeddings:
                # compute similarity
                sim_scores = self.compute_max_sim_scores(
                    q_token_embeddings=q_embedding.unsqueeze(0), d_token_embeddings=d_embedding.unsqueeze(0)
                )
                q_scores.append(sim_scores.item())

            q_tensor = torch.tensor(q_scores).unsqueeze(0)
            all_scores.append(q_tensor)
            q_scores = []

        all_scores = torch.cat(all_scores, dim=0)
        all_scores = all_scores.cpu()
        return all_scores

    def get_similarities(self, query_embeddings, doc_embeddings):
        # compute late interaction in batch to avoid OOM

        # if batch_size is 1, compute scores in pairwise fashion
        if self.batch_size == 1:
            return self.compute_single_scores(query_embeddings, doc_embeddings)

        all_scores = []
        for q_embedding in query_embeddings:
            q_embedding = q_embedding.unsqueeze(0)
            # compute similarity
            sim_scores = self.compute_max_sim_scores(q_token_embeddings=q_embedding, d_token_embeddings=doc_embeddings)
            all_scores.append(sim_scores)

        # stack scores
        all_scores = torch.cat(all_scores, dim=0)
        # move to CPU
        all_scores = all_scores.cpu()
        return all_scores

    def compute_metrics_e2e(self, data_formatter, **kwargs):
        queries_embeddings, label_ids = self.process_queries(data_formatter)

        # time the embedding of documents
        import time

        start = time.time()
        documents_embeddings, all_doc_ids = self.process_documents(data_formatter)
        runtime = time.time() - start
        scores = self.get_similarities(queries_embeddings, documents_embeddings)
        # cast scores to numpy
        scores = scores.numpy()
        metrics = self.get_metrics(scores, all_doc_ids, label_ids, **kwargs)
        metrics["runtime"] = runtime

        return metrics
