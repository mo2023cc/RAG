from hashlib import sha256
from pathlib import Path
from typing import Sequence
from text_utils import normalize_text

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Metadata
from docling_core.transforms.chunker.doc_chunk import DocChunk
from numpy.typing import NDArray


DEFAULT_CHROMA_PATH = ".chroma"
DEFAULT_COLLECTION = "localrag-documents"


def get_collection(
    persist_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION,
) -> Collection:
    client = chromadb.PersistentClient(path=str(Path(persist_path)))
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )


def _page_numbers(chunk: DocChunk) -> list[int]:
    return sorted(
        {
            provenance.page_no
            for item in chunk.meta.doc_items
            for provenance in (item.prov or [])
        }
    )


def _chunk_id(source: str, position: int, text: str) -> str:
    value = f"{source}\0{position}\0{text}".encode("utf-8")
    return sha256(value).hexdigest()


def index_chunks(
    collection: Collection,
    chunks: Sequence[DocChunk],
    embeddings: NDArray,
    source: str,
) -> int:
    if len(chunks) != len(embeddings):
        raise ValueError("Für jeden Chunk muss genau ein Embedding vorhanden sein.")
    if not chunks:
        return 0

    ids = [_chunk_id(source, i, chunk.text) for i, chunk in enumerate(chunks)]
    documents = [chunk.text for chunk in chunks]
    metadatas: list[Metadata] = [
        {
            "source": source,
            "chunk_index": i,
            "heading": normalize_text(" > ".join(chunk.meta.headings or [])),
            "pages": ", ".join(str(page) for page in _page_numbers(chunk)),
        }
        for i, chunk in enumerate(chunks)
    ]

    collection.delete(where={"source": source})
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )
    return len(ids)
