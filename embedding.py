from typing import cast
from docling_core.transforms.chunker.doc_chunk import DocChunk
from docling_core.types.doc.document import DoclingDocument
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from text_utils import normalize_text

from config import CHUNKING
from chunk import create_chunks, contextualize_chunk


QUERY_INSTRUCTION = (
    "Instruct: Given a search query, retrieve relevant passages that answer the query\n"
    "Query: "
)


def load_embedding_model(
    model_name: str = CHUNKING.embedding_model,
) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_chunks(model: SentenceTransformer, chunks: list[DocChunk]) -> NDArray:
    texts = [normalize_text(contextualize_chunk(c)) for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return cast(NDArray, embeddings)


def embed_query(model: SentenceTransformer, query: str) -> NDArray:
    query = normalize_text(query)

    embedding = model.encode(
        QUERY_INSTRUCTION + query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return cast(NDArray, embedding)


def process_document(
    model: SentenceTransformer,
    doc: DoclingDocument,
    *,
    max_tokens: int = CHUNKING.max_tokens,
    overlap_tokens: int = CHUNKING.overlap_tokens,
):
    chunks = create_chunks(
        doc,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    embeddings = embed_chunks(model, chunks)
    return chunks, embeddings
