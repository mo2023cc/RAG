import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

from bm25_retrieval import BM25Result, BM25Retriever
from config import RETRIEVAL
from dense_retrieval import SearchResult, search as dense_search
from embedding import embed_query, load_embedding_model
from hybrid_retrieval import HybridResult, HybridRetriever
from vector_store import DEFAULT_COLLECTION, get_collection


DEFAULT_QUESTIONS_PATH = "eval/questions.json"

RetrieverName = Literal["dense", "bm25", "hybrid"]
RetrievalResult = SearchResult | BM25Result | HybridResult
Retrieve = Callable[[str, int], list[RetrievalResult]]


class EvalCase(TypedDict):
    id: str
    query: str
    answerable: bool
    expected_headings: NotRequired[list[str]]
    category: NotRequired[str]


@dataclass(frozen=True)
class EvaluationConfig:
    retriever: RetrieverName
    questions_path: str
    top_k: int
    use_stemming: bool
    candidate_k: int
    rrf_k: int
    dense_weight: float
    bm25_weight: float
    collection_name: str


@dataclass(frozen=True)
class RetrieverRuntime:
    name: str
    score_label: str
    retrieve: Retrieve


def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Dense-, BM25- und hybrides RRF-Retrieval mit derselben "
            "Ground Truth evaluieren"
        )
    )
    parser.add_argument(
        "--retriever",
        choices=("dense", "bm25", "hybrid"),
        default="dense",
    )
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument(
        "--top-k",
        type=int,
        default=RETRIEVAL.context_top_k,
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=RETRIEVAL.candidate_k,
        help=(
            "Kandidaten pro Einzel-Retriever für RRF "
            f"(Standard: {RETRIEVAL.candidate_k})"
        ),
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=RETRIEVAL.rrf_k,
        help=f"RRF-Rankkonstante (Standard: {RETRIEVAL.rrf_k})",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=RETRIEVAL.dense_weight,
        help=f"Dense-Gewicht für RRF (Standard: {RETRIEVAL.dense_weight})",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=RETRIEVAL.bm25_weight,
        help=f"BM25-Gewicht für RRF (Standard: {RETRIEVAL.bm25_weight})",
    )
    parser.add_argument(
        "--no-stemming",
        action="store_true",
        help="Deutsches Stemming für BM25 deaktivieren",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k muss mindestens 1 sein")
    if args.candidate_k < 1:
        parser.error("--candidate-k muss mindestens 1 sein")
    if args.rrf_k < 1:
        parser.error("--rrf-k muss mindestens 1 sein")
    if args.dense_weight <= 0:
        parser.error("--dense-weight muss größer als 0 sein")
    if args.bm25_weight <= 0:
        parser.error("--bm25-weight muss größer als 0 sein")
    if args.retriever == "hybrid" and args.top_k > args.candidate_k:
        parser.error("Bei Hybrid darf --top-k nicht größer als --candidate-k sein")
    if args.no_stemming and args.retriever == "dense":
        parser.error(
            "--no-stemming kann nur mit --retriever bm25 oder hybrid "
            "verwendet werden"
        )

    return EvaluationConfig(
        retriever=cast(RetrieverName, args.retriever),
        questions_path=args.questions,
        top_k=args.top_k,
        use_stemming=not args.no_stemming,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        collection_name=args.collection,
    )


def load_cases(path: str = DEFAULT_QUESTIONS_PATH) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Die Evaluationsdatei muss eine JSON-Liste enthalten.")

    required = {"id", "query", "answerable"}
    for case in raw:
        if not isinstance(case, dict):
            raise ValueError(f"Testfall muss ein JSON-Objekt sein: {case!r}")

        missing = required - case.keys()
        if missing:
            raise ValueError(f"Testfall unvollständig, fehlt: {missing} in {case}")
        if case["answerable"] and not case.get("expected_headings"):
            raise ValueError(
                f"Beantwortbarer Testfall ohne expected_headings: {case['id']}"
            )

    return cast(list[EvalCase], raw)


def build_retriever(config: EvaluationConfig) -> RetrieverRuntime:
    collection = get_collection(collection_name=config.collection_name)
    if collection.count() == 0:
        raise RuntimeError(
            "Der Chroma-Index ist leer. Zuerst `python main.py index` ausführen."
        )

    if config.retriever == "dense":
        model = load_embedding_model()

        def dense_retrieve(query: str, top_k: int) -> list[RetrievalResult]:
            query_embedding = embed_query(model, query)
            return list(dense_search(collection, query_embedding, top_k=top_k))

        return RetrieverRuntime(
            name="Dense",
            score_label="Ähnlichkeit",
            retrieve=dense_retrieve,
        )

    if config.retriever == "bm25":
        bm25 = BM25Retriever(
            collection,
            use_stemming=config.use_stemming,
        )

        def bm25_retrieve(query: str, top_k: int) -> list[RetrievalResult]:
            return list(bm25.search(query, top_k=top_k))

        stemming = "mit Stemming" if config.use_stemming else "ohne Stemming"
        return RetrieverRuntime(
            name=f"BM25 ({stemming})",
            score_label="BM25-Score",
            retrieve=bm25_retrieve,
        )

    stemming = "mit Stemming" if config.use_stemming else "ohne Stemming"
    model = load_embedding_model()
    hybrid = HybridRetriever(
        collection,
        model,
        use_stemming=config.use_stemming,
        candidate_k=config.candidate_k,
        rrf_k=config.rrf_k,
        dense_weight=config.dense_weight,
        bm25_weight=config.bm25_weight,
    )

    def hybrid_retrieve(query: str, top_k: int) -> list[RetrievalResult]:
        return list(hybrid.search(query, top_k=top_k))

    return RetrieverRuntime(
        name=(
            f"Hybrid RRF ({stemming}, Kandidaten={config.candidate_k}, "
            f"k={config.rrf_k}, Dense={config.dense_weight:g}, "
            f"BM25={config.bm25_weight:g})"
        ),
        score_label="RRF-Score",
        retrieve=hybrid_retrieve,
    )


def is_relevant(
    result: RetrievalResult,
    expected_headings: list[str],
) -> bool:
    actual = result.heading.casefold()
    return any(expected.casefold() in actual for expected in expected_headings)


def result_score(result: RetrievalResult) -> float:
    if isinstance(result, SearchResult):
        return result.similarity
    if isinstance(result, BM25Result):
        return result.score
    return result.rrf_score


def evaluate(
    cases: list[EvalCase],
    runtime: RetrieverRuntime,
    top_k: int,
) -> None:
    hit_at_1 = 0
    hit_at_k = 0
    reciprocal_rank_sum = 0.0
    answerable_count = 0
    negative_scores: list[float] = []

    print(f"Retriever: {runtime.name}")
    print(f"Fragen: {len(cases)}, Top-k: {top_k}")

    for case in cases:
        results = runtime.retrieve(case["query"], top_k)
        print(f"\n{case['id']}: {case['query']}")

        if not case["answerable"]:
            top_score = result_score(results[0]) if results else 0.0
            negative_scores.append(top_score)
            print(
                f"  Nicht beantwortbar, bester {runtime.score_label}: "
                f"{top_score:.3f}"
            )
            continue

        answerable_count += 1
        expected_headings = case.get("expected_headings", [])
        relevant_rank: int | None = None

        for rank, result in enumerate(results, start=1):
            relevant = is_relevant(result, expected_headings)
            marker = "✓" if relevant else " "
            print(
                f"  {marker} Rang {rank}: {result_score(result):.3f} "
                f"| {result.heading}"
            )

            if relevant and relevant_rank is None:
                relevant_rank = rank

        if relevant_rank == 1:
            hit_at_1 += 1
        if relevant_rank is not None:
            hit_at_k += 1
            reciprocal_rank_sum += 1 / relevant_rank

    print("\n--- Ergebnis ---")
    print(f"Retriever: {runtime.name}")
    if answerable_count:
        print(f"Beantwortbare Fragen: {answerable_count}")
        print(f"Hit@1: {hit_at_1 / answerable_count:.1%}")
        print(f"Hit@{top_k}: {hit_at_k / answerable_count:.1%}")
        print(f"MRR:   {reciprocal_rank_sum / answerable_count:.3f}")
    else:
        print("Keine beantwortbaren Fragen vorhanden.")

    if negative_scores:
        print(
            f"Höchster {runtime.score_label} bei unbeantwortbaren Fragen: "
            f"{max(negative_scores):.3f}"
        )


def main() -> None:
    config = parse_args()
    cases = load_cases(config.questions_path)
    runtime = build_retriever(config)
    evaluate(cases, runtime, config.top_k)


if __name__ == "__main__":
    main()
