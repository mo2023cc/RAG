import argparse

from bm25_retrieval import BM25Result, BM25Retriever
from config import CHUNKING, RETRIEVAL
from dense_retrieval import SearchResult, search as dense_search
from embedding import embed_query, load_embedding_model, process_document
from hybrid_retrieval import HybridResult, HybridRetriever
from ingestion import load_documents
from vector_store import DEFAULT_COLLECTION, get_collection, index_chunks


def run_index(
    data_folder: str = "data",
    use_cache: bool = True,
    *,
    collection_name: str = DEFAULT_COLLECTION,
    max_tokens: int = CHUNKING.max_tokens,
    overlap_tokens: int = CHUNKING.overlap_tokens,
) -> None:
    model = load_embedding_model()
    collection = get_collection(collection_name=collection_name)

    for item in load_documents(data_folder, use_cache=use_cache):
        chunks, embeddings = process_document(
            model,
            item["document"],
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        n_indexed = index_chunks(collection, chunks, embeddings, source=item["source"])
        print(f"{item['source']}: {len(chunks)} Chunks, {n_indexed} indexiert")


def run_query(
    query_text: str,
    top_k: int = RETRIEVAL.context_top_k,
    retriever: str = "dense",
    collection_name: str = DEFAULT_COLLECTION,
    candidate_k: int = RETRIEVAL.candidate_k,
    rrf_k: int = RETRIEVAL.rrf_k,
    dense_weight: float = RETRIEVAL.dense_weight,
    bm25_weight: float = RETRIEVAL.bm25_weight,
) -> None:
    collection = get_collection(collection_name=collection_name)

    if retriever == "bm25":
        results: list[SearchResult | BM25Result | HybridResult] = list(
            BM25Retriever(collection).search(query_text, top_k=top_k)
        )
    else:
        model = load_embedding_model()
        if retriever == "hybrid":
            results = list(
                HybridRetriever(
                    collection,
                    model,
                    candidate_k=max(candidate_k, top_k),
                    rrf_k=rrf_k,
                    dense_weight=dense_weight,
                    bm25_weight=bm25_weight,
                ).search(query_text, top_k=top_k)
            )
        else:
            query_embedding = embed_query(model, query_text)
            results = list(
                dense_search(collection, query_embedding, top_k=top_k)
            )

    if not results:
        print("Keine Treffer. Wurde der Index bereits aufgebaut? (python main.py index)")
        return

    for i, r in enumerate(results, 1):
        if isinstance(r, SearchResult):
            score = f"Ähnlichkeit: {r.similarity:.3f}"
        elif isinstance(r, BM25Result):
            score = f"BM25-Score: {r.score:.3f}"
        else:
            score = (
                f"RRF-Score: {r.rrf_score:.4f}, "
                f"Dense-Rang: {r.dense_rank or '-'}, "
                f"BM25-Rang: {r.bm25_rank or '-'}"
            )

        print(f"\n--- Treffer {i} ({score}) ---")
        print(f"Quelle: {r.source} | {r.heading} | Seiten: {r.pages}")
        print(r.text)


def main() -> None:
    parser = argparse.ArgumentParser(description="localRAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Dokumente chunken, embedden und indexieren")
    index_parser.add_argument("--data-folder", default="data")
    index_parser.add_argument("--no-cache", action="store_true")
    index_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    index_parser.add_argument(
        "--max-tokens",
        type=int,
        default=CHUNKING.max_tokens,
    )
    index_parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=CHUNKING.overlap_tokens,
    )

    query_parser = subparsers.add_parser("query", help="Suchanfrage gegen den Index stellen")
    query_parser.add_argument("text")
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=RETRIEVAL.context_top_k,
    )
    query_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    query_parser.add_argument(
        "--candidate-k",
        type=int,
        default=RETRIEVAL.candidate_k,
    )
    query_parser.add_argument("--rrf-k", type=int, default=RETRIEVAL.rrf_k)
    query_parser.add_argument(
        "--dense-weight",
        type=float,
        default=RETRIEVAL.dense_weight,
    )
    query_parser.add_argument(
        "--bm25-weight",
        type=float,
        default=RETRIEVAL.bm25_weight,
    )
    query_parser.add_argument(
        "--retriever",
        choices=("dense", "bm25", "hybrid"),
        default="dense",
    )

    args = parser.parse_args()

    if args.command == "index":
        if args.max_tokens < 1:
            index_parser.error("--max-tokens muss mindestens 1 sein")
        if args.overlap_tokens < 0:
            index_parser.error("--overlap-tokens darf nicht negativ sein")
        if args.overlap_tokens >= args.max_tokens:
            index_parser.error(
                "--overlap-tokens muss kleiner als --max-tokens sein"
            )
        run_index(
            data_folder=args.data_folder,
            use_cache=not args.no_cache,
            collection_name=args.collection,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
        )
    elif args.command == "query":
        if args.top_k < 1:
            query_parser.error("--top-k muss mindestens 1 sein")
        if args.candidate_k < 1:
            query_parser.error("--candidate-k muss mindestens 1 sein")
        if args.rrf_k < 1:
            query_parser.error("--rrf-k muss mindestens 1 sein")
        if args.dense_weight <= 0 or args.bm25_weight <= 0:
            query_parser.error("RRF-Gewichte müssen größer als 0 sein")
        run_query(
            args.text,
            top_k=args.top_k,
            retriever=args.retriever,
            collection_name=args.collection,
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
        )


if __name__ == "__main__":
    main()
