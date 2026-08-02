import pytest

from src.embeddings.base_embedder import BaseEmbedder
from src.retrieval.base_vector_store import BaseVectorStore
from src.retrieval.dense_retriever import DenseRetriever


class DummyEmbedder(BaseEmbedder):

    @property
    def dimension(self):
        return 384

    def encode_chunks(self, chunks):
        raise NotImplementedError

    def encode_query(self, query):
        raise NotImplementedError


class DummyVectorStore(BaseVectorStore):

    def __len__(self):
        return 0

    def clear(self):
        pass

    def add(self, chunks, embeddings):
        raise NotImplementedError

    def search(self, query_embedding, top_k=5):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


def test_constructor():

    embedder = DummyEmbedder()

    store = DummyVectorStore()

    retriever = DenseRetriever(
        embedder,
        store,
    )

    assert retriever.embedder is embedder
    assert retriever.vector_store is store
