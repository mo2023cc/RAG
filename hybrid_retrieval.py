from dataclasses import dataclass

from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from bm25_retrieval import BM25Result, BM25Retriever
from config import RETRIEVAL
from dense_retrieval import SearchResult, search as dense_search
from embedding import embed_query


__all__ = ["HybridResult", "HybridRetriever", "reciprocal_rank_fusion"]


@dataclass(frozen=True)
class HybridResult:
    chunk_id: str
    text: str
    source: str
    heading: str
    pages: str
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None


def reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    bm25_results: list[BM25Result],
    *,
    top_k: int = RETRIEVAL.context_top_k,
    rrf_k: int = RETRIEVAL.rrf_k,
    dense_weight: float = RETRIEVAL.dense_weight,
    bm25_weight: float = RETRIEVAL.bm25_weight,
) -> list[HybridResult]:
    if top_k < 1:
        raise ValueError("top_k muss mindestens 1 sein.")
    if rrf_k < 1:
        raise ValueError("rrf_k muss mindestens 1 sein.")
    if dense_weight <= 0:
        raise ValueError("dense_weight muss größer als 0 sein.")
    if bm25_weight <= 0:
        raise ValueError("bm25_weight muss größer als 0 sein.")

    scores: dict[str, float] = {}
    records: dict[str, SearchResult | BM25Result] = {}
    dense_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result.chunk_id

        if chunk_id in dense_ranks:
            continue

        dense_ranks[chunk_id] = rank
        records.setdefault(chunk_id, result)
        scores[chunk_id] = (
            scores.get(chunk_id, 0.0)
            + dense_weight / (rrf_k + rank)
        )

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result.chunk_id

        if chunk_id in bm25_ranks:
            continue

        bm25_ranks[chunk_id] = rank
        records.setdefault(chunk_id, result)
        scores[chunk_id] = (
            scores.get(chunk_id, 0.0)
            + bm25_weight / (rrf_k + rank)
        )

    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            dense_ranks.get(chunk_id, float("inf")),
            bm25_ranks.get(chunk_id, float("inf")),
        ),
    )[:top_k]

    return [
        HybridResult(
            chunk_id=chunk_id,
            text=records[chunk_id].text,
            source=records[chunk_id].source,
            heading=records[chunk_id].heading,
            pages=records[chunk_id].pages,
            rrf_score=scores[chunk_id],
            dense_rank=dense_ranks.get(chunk_id),
            bm25_rank=bm25_ranks.get(chunk_id),
        )
        for chunk_id in ranked_ids
    ]


class HybridRetriever:
    def __init__(
        self,
        collection: Collection,
        model: SentenceTransformer,
        *,
        use_stemming: bool = RETRIEVAL.use_stemming,
        candidate_k: int = RETRIEVAL.candidate_k,
        rrf_k: int = RETRIEVAL.rrf_k,
        dense_weight: float = RETRIEVAL.dense_weight,
        bm25_weight: float = RETRIEVAL.bm25_weight,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k muss mindestens 1 sein.")
        if rrf_k < 1:
            raise ValueError("rrf_k muss mindestens 1 sein.")
        if dense_weight <= 0:
            raise ValueError("dense_weight muss größer als 0 sein.")
        if bm25_weight <= 0:
            raise ValueError("bm25_weight muss größer als 0 sein.")

        self.collection = collection
        self.model = model
        self.bm25 = BM25Retriever(
            collection,
            use_stemming=use_stemming,
        )
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def search(
        self,
        query: str,
        top_k: int = RETRIEVAL.context_top_k,
    ) -> list[HybridResult]:
        if top_k < 1:
            raise ValueError("top_k muss mindestens 1 sein.")
        if top_k > self.candidate_k:
            raise ValueError(
                "top_k darf nicht größer als candidate_k sein."
            )

        query_embedding = embed_query(self.model, query)

        dense_results = dense_search(
            self.collection,
            query_embedding,
            top_k=self.candidate_k,
        )

        bm25_results = self.bm25.search(
            query,
            top_k=self.candidate_k,
        )

        return reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            top_k=top_k,
            rrf_k=self.rrf_k,
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
        )
