import argparse
from collections.abc import Callable
from dataclasses import dataclass

from bm25_retrieval import BM25Retriever
from config import CHUNKING, RETRIEVAL
from dense_retrieval import search as dense_search
from embedding import embed_query, load_embedding_model, process_document
from evaluation import (
    DEFAULT_QUESTIONS_PATH,
    EvalCase,
    RetrievalResult,
    is_relevant,
    load_cases,
)
from hybrid_retrieval import HybridRetriever
from ingestion import load_documents
from vector_store import get_collection, index_chunks


Retrieve = Callable[[str, int], list[RetrievalResult]]


@dataclass(frozen=True)
class Metrics:
    hit_at_1: float
    hit_at_k: float
    mrr: float


def evaluate_retriever(
    cases: list[EvalCase],
    retrieve: Retrieve,
    top_k: int,
) -> Metrics:
    answerable_cases = [case for case in cases if case["answerable"]]
    if not answerable_cases:
        raise ValueError("Keine beantwortbaren Evaluationsfragen vorhanden.")

    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_rank_sum = 0.0

    for case in answerable_cases:
        results = retrieve(case["query"], top_k)
        expected = case.get("expected_headings", [])
        relevant_rank = next(
            (
                rank
                for rank, result in enumerate(results, start=1)
                if is_relevant(result, expected)
            ),
            None,
        )
        if relevant_rank == 1:
            hits_at_1 += 1
        if relevant_rank is not None:
            hits_at_k += 1
            reciprocal_rank_sum += 1 / relevant_rank

    count = len(answerable_cases)
    return Metrics(
        hit_at_1=hits_at_1 / count,
        hit_at_k=hits_at_k / count,
        mrr=reciprocal_rank_sum / count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunkgrößen in getrennten Chroma-Collections vergleichen"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(CHUNKING.size_candidates),
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=CHUNKING.overlap_tokens,
    )
    parser.add_argument(
        "--retriever",
        choices=("dense", "bm25", "hybrid"),
        default="dense",
    )
    parser.add_argument("--top-k", type=int, default=RETRIEVAL.context_top_k)
    parser.add_argument("--candidate-k", type=int, default=RETRIEVAL.candidate_k)
    parser.add_argument("--rrf-k", type=int, default=RETRIEVAL.rrf_k)
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=RETRIEVAL.dense_weight,
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=RETRIEVAL.bm25_weight,
    )
    parser.add_argument("--data-folder", default="data")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--base-collection", default="localrag-chunks")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-stemming", action="store_true")
    args = parser.parse_args()

    if any(size < 1 for size in args.sizes):
        parser.error("Alle Chunkgrößen müssen mindestens 1 sein")
    if args.overlap_tokens < 0:
        parser.error("--overlap-tokens darf nicht negativ sein")
    if any(args.overlap_tokens >= size for size in args.sizes):
        parser.error("--overlap-tokens muss kleiner als jede Chunkgröße sein")
    if args.top_k < 1:
        parser.error("--top-k muss mindestens 1 sein")
    if args.candidate_k < args.top_k:
        parser.error("--candidate-k darf nicht kleiner als --top-k sein")
    if args.rrf_k < 1:
        parser.error("--rrf-k muss mindestens 1 sein")
    if args.dense_weight <= 0:
        parser.error("--dense-weight muss größer als 0 sein")
    if args.bm25_weight <= 0:
        parser.error("--bm25-weight muss größer als 0 sein")
    return args


def main() -> None:
    args = parse_args()
    cases = load_cases(args.questions)
    documents = load_documents(
        args.data_folder,
        use_cache=not args.no_cache,
    )
    model = load_embedding_model()
    rows: list[tuple[int, int, str, Metrics]] = []

    for max_tokens in args.sizes:
        collection_name = (
            f"{args.base_collection}-t{max_tokens}-o{args.overlap_tokens}"
        )
        collection = get_collection(collection_name=collection_name)

        print(
            f"\nIndexiere {collection_name}: "
            f"max_tokens={max_tokens}, overlap={args.overlap_tokens}"
        )
        for item in documents:
            chunks, embeddings = process_document(
                model,
                item["document"],
                max_tokens=max_tokens,
                overlap_tokens=args.overlap_tokens,
            )
            index_chunks(
                collection,
                chunks,
                embeddings,
                source=item["source"],
            )

        if args.retriever == "dense":
            def retrieve(query: str, top_k: int) -> list[RetrievalResult]:
                query_embedding = embed_query(model, query)
                return list(
                    dense_search(collection, query_embedding, top_k=top_k)
                )
        elif args.retriever == "bm25":
            bm25 = BM25Retriever(
                collection,
                use_stemming=not args.no_stemming,
            )

            def retrieve(query: str, top_k: int) -> list[RetrievalResult]:
                return list(bm25.search(query, top_k=top_k))
        else:
            hybrid = HybridRetriever(
                collection,
                model,
                use_stemming=not args.no_stemming,
                candidate_k=args.candidate_k,
                rrf_k=args.rrf_k,
                dense_weight=args.dense_weight,
                bm25_weight=args.bm25_weight,
            )

            def retrieve(query: str, top_k: int) -> list[RetrievalResult]:
                return list(hybrid.search(query, top_k=top_k))

        metrics = evaluate_retriever(cases, retrieve, args.top_k)
        rows.append((max_tokens, collection.count(), collection_name, metrics))
        print(
            f"Ergebnis: {collection.count()} Chunks | "
            f"Hit@1 {metrics.hit_at_1:.1%} | "
            f"Hit@{args.top_k} {metrics.hit_at_k:.1%} | "
            f"MRR {metrics.mrr:.3f}"
        )

    print("\n--- Vergleich ---")
    print("Tokens | Chunks | Hit@1 | Hit@k | MRR   | Collection")
    for max_tokens, chunk_count, collection_name, metrics in rows:
        print(
            f"{max_tokens:>6} | {chunk_count:>6} | "
            f"{metrics.hit_at_1:>6.1%} | {metrics.hit_at_k:>5.1%} | "
            f"{metrics.mrr:.3f} | {collection_name}"
        )


if __name__ == "__main__":
    main()
