from sentence_transformers import SentenceTransformer

from cde_benchmark.embedders.base_embedder import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model: SentenceTransformer = None,
        batch_size: int = 128,
        show_progress_bar: bool = True,
        add_prefix: bool = False,
        device: str = "cpu",
    ):
        super().__init__(is_contextual_model=False, device=device)
        self.model = model
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.query_prompt = "search_query: " if add_prefix else ""
        self.doc_prompt = "search_document: " if add_prefix else ""

    def embed_queries(self, queries):
        return self.model.encode(
            queries,
            show_progress_bar=self.show_progress_bar,
            batch_size=self.batch_size,
            prompt=self.query_prompt,
        )

    def embed_documents(self, documents):
        return self.model.encode(
            documents,
            show_progress_bar=self.show_progress_bar,
            batch_size=self.batch_size,
            prompt=self.doc_prompt,
        )
