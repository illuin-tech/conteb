from random import random
from typing import List, Tuple

from datasets import load_dataset


class BaseDataFormatter:
    def get_nested(self) -> Tuple[List[List[str]], List[List[str]]]:
        raise NotImplementedError

    def get_flattened(self) -> Tuple[List[str], List[str]]:
        raise NotImplementedError

    def get_queries(self) -> Tuple[List[str], List[str]]:
        raise NotImplementedError


class DataFormatter(BaseDataFormatter):
    def __init__(
        self,
        dataset_path,
        split,
        query_key="queries",
        doc_key="documents",
        query_prompt="",
        doc_prompt="",
        og_ratio=0.0,
    ):
        self.doc_dataset = None
        self.queries_dataset = None
        self._load_from_path(dataset_path, split, query_key, doc_key)

        def replace_chunk(sample):
            if random() < og_ratio:
                sample["chunk"] = sample["og_chunk"]
            return sample

        if og_ratio > 0.0:
            print(f"Using original chunks with ratio {og_ratio}")
            self.doc_dataset = self.doc_dataset.map(replace_chunk)

        self.doc_dataset = self.doc_dataset.map(self.parse_id)
        self.queries_dataset = self.queries_dataset.map(self.parse_id)
        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt

    def _load_from_path(self, path, split, query_key, doc_key):
        self.doc_dataset = load_dataset(path, doc_key, split=split)
        self.queries_dataset = load_dataset(path, query_key, split=split)
        # mapping dataset is used to map queries to relevant documents

    @staticmethod
    def parse_id(sample):
        if "chunk_id" in sample:
            doc_id, internal_id = sample["chunk_id"].split("_")
            return {"doc_id": doc_id, "internal_id": int(internal_id)}
        elif "chunk_ids" in sample:
            doc_id, _ = sample["chunk_ids"][0].split("_")
            internal_ids = [int(id_.split("_")[1]) for id_ in sample["chunk_ids"]]
            return {"doc_id": doc_id, "internal_ids": internal_ids, "chunk_id": sample["chunk_ids"]}
        else:
            raise ValueError("chunk_id or chunk_ids not found in sample")

    def get_nested(self, col="chunk") -> Tuple[List[List[str]], List[List[str]]]:
        # TODO: verify it's sorted
        return list(self.doc_dataset.to_pandas().groupby("doc_id", sort=True)[col].apply(list)), list(
            self.doc_dataset.to_pandas().groupby("doc_id", sort=True)["chunk_id"].apply(list)
        )

    def get_flattened(self, col="chunk") -> Tuple[List[str], List[str]]:
        # flatten data
        docs = [f"{self.doc_prompt}{doc}" for doc in self.doc_dataset[col]]
        return docs, self.doc_dataset["chunk_id"]

    def get_queries(self) -> Tuple[List[str], List[str]]:
        queries = [f"{self.query_prompt}{q}" for q in self.queries_dataset["query"]]
        return queries, self.queries_dataset["chunk_id"]

    def get_nested_queries(self) -> Tuple[List[List[str]], List[List[str]]]:
        return list(self.queries_dataset.to_pandas().groupby("doc_id", sort=True)["query"].apply(list)), list(
            self.queries_dataset.to_pandas().groupby("doc_id", sort=True)["chunk_id"].apply(list)
        )


class BEIRDataFormatter(BaseDataFormatter):
    def __init__(
        self,
        dataset_path,
        split,
        query_key="queries",
        doc_key="corpus",
        query_prompt="",
        doc_prompt="",
    ):
        self.doc_dataset = None
        self.queries_dataset = None
        self.mapping = {}
        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt
        self._load_from_path(dataset_path, split, query_key, doc_key)

    def _load_from_path(self, path, split, query_key, doc_key):
        self.doc_dataset = load_dataset(path, doc_key, split=split)
        self.queries_dataset = load_dataset(path, query_key, split=split)
        mapping_dataset = load_dataset(path, "qrels", split=split)
        # self.mapping = {
        #     query["query-id"]: query["corpus-id"] for query in mapping_dataset
        # }
        for query in mapping_dataset:
            if query["query-id"] not in self.mapping:
                self.mapping[query["query-id"]] = [query["corpus-id"]]
            else:
                # construct list
                self.mapping[query["query-id"]].append(query["corpus-id"])

    def get_nested(self) -> Tuple[List[List[str]], List[List[str]]]:
        raise NotImplementedError

    def get_flattened(self) -> Tuple[List[str], List[str]]:
        # flatten data
        docs = [f"{self.doc_prompt}{doc}" for doc in self.doc_dataset["text"]]
        return docs, self.doc_dataset["_id"]

    def get_queries(self) -> Tuple[List[str], List[str]]:
        queries = [f"{self.query_prompt}{query}" for query in self.queries_dataset["text"]]
        gold_docs = []
        for query in self.queries_dataset:
            gold_docs.append(self.mapping[query["_id"]])
        return queries, gold_docs


class LongEmbedDataFormatter(BaseDataFormatter):
    def __init__(
        self,
        dataset_path,
        dataset_name,
        query_key="queries",
        doc_key="corpus",
        query_prompt="",
        doc_prompt="",
        context_length=None,
    ):
        self.doc_dataset = None
        self.queries_dataset = None
        self.mapping = {}
        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt
        self.context_length = context_length
        self._load_from_path(dataset_path, dataset_name, query_key, doc_key)

    def _load_from_path(self, path, dataset_name, query_key, doc_key):
        self.doc_dataset = load_dataset(path, dataset_name, split=doc_key)
        self.queries_dataset = load_dataset(path, dataset_name, split=query_key)
        mapping_dataset = load_dataset(path, dataset_name, split="qrels")
        # self.mapping = {
        #     query["query-id"]: query["corpus-id"] for query in mapping_dataset
        # }
        for query in mapping_dataset:
            if query["qid"] not in self.mapping:
                self.mapping[query["qid"]] = [query["doc_id"]]
            else:
                # construct list
                self.mapping[query["qid"]].append(query["doc_id"])

    def get_nested(self) -> Tuple[List[List[str]], List[List[str]]]:
        raise NotImplementedError

    def get_flattened(self) -> Tuple[List[str], List[str]]:
        # flatten data
        docs = self.doc_dataset.filter(
            lambda x: not self.context_length or int(self.context_length) == int(x["context_length"])
        )
        return docs["text"], docs["doc_id"]

    def get_queries(self) -> Tuple[List[str], List[str]]:
        queries = [
            f"{self.query_prompt}{query['text']}"
            for query in self.queries_dataset
            if not self.context_length or self.context_length in query["qid"]
        ]
        gold_docs = []
        for query in self.queries_dataset:
            if not self.context_length or self.context_length in query["qid"]:
                gold_docs.append(self.mapping[query["qid"]])

        assert len(queries) == len(gold_docs), (
            f"Queries and gold_docs length mismatch: {len(queries)} != {len(gold_docs)}"
        )
        return queries, gold_docs
