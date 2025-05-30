import torch
from contextual_embeddings import LongContextEmbeddingModel
from sentence_transformers import SentenceTransformer

from cde_benchmark.embedders.base_embedder import Embedder


class ContextualEmbedder(LongContextEmbeddingModel, Embedder):
    def __init__(
        self,
        model: SentenceTransformer = None,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        device: str = "cpu",
        add_prefix: bool = False,
    ):
        LongContextEmbeddingModel.__init__(self, base_model=model, add_prefix=add_prefix)
        Embedder.__init__(self, is_contextual_model=True, device=device)
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size


class MergedContextualEmbedder(LongContextEmbeddingModel, Embedder):
    def __init__(
        self,
        model: SentenceTransformer = None,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        device: str = "cpu",
        add_prefix: bool = False,
    ):
        LongContextEmbeddingModel.__init__(self, base_model=model, add_prefix=add_prefix)
        Embedder.__init__(self, is_contextual_model=True, device=device)
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.merge_embeddings = True


class LateInteractionContextualEmbedder(LongContextEmbeddingModel, Embedder):
    def __init__(
        self,
        model: SentenceTransformer = None,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        device: str = "cpu",
        add_prefix: bool = False,
    ):
        LongContextEmbeddingModel.__init__(self, base_model=model, add_prefix=add_prefix, pooling_mode="tokens")
        Embedder.__init__(self, is_contextual_model=True, device=device)
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size

    def process_documents(self, data_formatter, col="chunk"):
        doc_embeddings, document_ids = super().process_documents(data_formatter, col=col)
        doc_embeddings = torch.nn.utils.rnn.pad_sequence(doc_embeddings, batch_first=True)
        return doc_embeddings, document_ids

    def get_similarities(self, query_embeddings, doc_embeddings):
        # compute late interaction in batch to avoid OOM
        all_scores = []
        for q_embedding in query_embeddings:
            q_embedding = q_embedding.unsqueeze(0)
            # compute similarity
            sim_scores = self._compute_max_sim_scores(q_token_embeddings=q_embedding, d_token_embeddings=doc_embeddings)
            all_scores.append(sim_scores)

        # stack scores
        all_scores = torch.cat(all_scores, dim=0)
        # convert to numpy
        all_scores = all_scores.cpu()
        return all_scores
