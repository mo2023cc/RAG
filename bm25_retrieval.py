import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from chromadb.api.models.Collection import Collection
from nltk.stem.snowball import SnowballStemmer
from rank_bm25 import BM25Okapi

from config import RETRIEVAL
from text_utils import normalize_text


__all__ = ["BM25Result", "BM25Retriever", "tokenize"]

TOKEN_PATTERN = re.compile(
    r"§\s*\d+(?:\.\d+)*|\d+(?:\.\d+)*|[\wÄÖÜäöüß]+(?:-[\wÄÖÜäöüß]+)*",
    re.UNICODE,
)

GERMAN_STEMMER = SnowballStemmer("german")


def _should_preserve(token: str) -> bool:
    return token.startswith("§") or token[0].isdigit() or len(token) <= 4


def tokenize(text: str, use_stemming: bool = True) -> list[str]:
    normalized = normalize_text(text)
    tokens = [
        re.sub(r"\s+", "", token).casefold()
        for token in TOKEN_PATTERN.findall(normalized)
    ]

    if not use_stemming:
        return tokens

    return [
        token if _should_preserve(token) else GERMAN_STEMMER.stem(token)
        for token in tokens
    ]


@dataclass(frozen=True)
class BM25Result:
    chunk_id: str
    text: str
    source: str
    heading: str
    pages: str
    score: float


class BM25Retriever:
    def __init__(
        self,
        collection: Collection,
        *,
        use_stemming: bool = RETRIEVAL.use_stemming,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        records = collection.get(include=["documents", "metadatas"])

        self.ids = records["ids"]
        self.documents = [document or "" for document in records["documents"] or []]
        self.metadatas = [metadata or {} for metadata in records["metadatas"] or []]
        self.use_stemming = use_stemming

        if not self.ids:
            raise ValueError(
                "Der Chroma-Index ist leer. "
                "Zuerst `python main.py index` ausführen."
            )
        if not (
            len(self.ids) == len(self.documents) == len(self.metadatas)
        ):
            raise RuntimeError("IDs, Dokumente und Metadaten sind nicht vollständig.")

        self.index_texts = [
            self._create_index_text(document, metadata)
            for document, metadata in zip(
                self.documents,
                self.metadatas,
                strict=True,
            )
        ]
        tokenized_corpus = [
            tokenize(text, use_stemming=self.use_stemming)
            for text in self.index_texts
        ]

        if not any(tokenized_corpus):
            raise ValueError("Die Chroma-Dokumente enthalten keinen indexierbaren Text.")

        self.bm25 = BM25Okapi(
            tokenized_corpus,
            k1=k1,
            b=b,
            epsilon=epsilon,
        )

    @staticmethod
    def _create_index_text(
        document: str,
        metadata: Mapping[str, object],
    ) -> str:
        heading = metadata.get("heading") or ""
        return f"{heading}\n{document}" if heading else document

    def search(
        self,
        query: str,
        top_k: int = RETRIEVAL.context_top_k,
    ) -> list[BM25Result]:
        if top_k < 1:
            raise ValueError("top_k muss mindestens 1 sein.")

        query_tokens = tokenize(query, use_stemming=self.use_stemming)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(-scores, kind="stable")

        results: list[BM25Result] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0:
                continue

            metadata = self.metadatas[index]
            results.append(
                BM25Result(
                    chunk_id=self.ids[index],
                    text=self.documents[index],
                    source=str(metadata.get("source", "")),
                    heading=str(metadata.get("heading", "")),
                    pages=str(metadata.get("pages", "")),
                    score=score,
                )
            )

            if len(results) == top_k:
                break

        return results
