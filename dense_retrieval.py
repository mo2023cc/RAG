from dataclasses import dataclass

from chromadb.api.models.Collection import Collection
from numpy.typing import NDArray

from config import RETRIEVAL

__all__ = ["SearchResult", "search"]


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    text: str
    source: str
    heading: str
    pages: str
    distance: float

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


def search(
    collection: Collection,
    query_embedding: NDArray,
    top_k: int = RETRIEVAL.context_top_k,
) -> list[SearchResult]:
    if top_k < 1:
        raise ValueError("top_k muss mindestens 1 sein.")

    count = collection.count()
    if count == 0:
        return []

    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    ids = result["ids"][0]
    documents = (result["documents"] or [[]])[0]
    metadatas = (result["metadatas"] or [[]])[0]
    distances = (result["distances"] or [[]])[0]

    return [
        SearchResult(
            chunk_id=chunk_id,
            text=document,
            source=str(metadata.get("source", "")),
            heading=str(metadata.get("heading", "")),
            pages=str(metadata.get("pages", "")),
            distance=float(distance),
        )
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        )
    ]
