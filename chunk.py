import re
from typing import cast

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.doc_chunk import DocChunk, DocMeta
from docling_core.types.doc.document import DoclingDocument
from transformers import AutoTokenizer

from config import CHUNKING


SKIP_HEADINGS = {"inhaltsverzeichnis", "änderungshistorie"}


def _normalize_heading(heading: str) -> str:
    normalized = heading.strip().casefold()
    return re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", normalized)


def _is_skipped(chunk: DocChunk, skip_headings: set[str]) -> bool:
    headings = getattr(chunk.meta, "headings", None) or []
    return any(_normalize_heading(heading) in skip_headings for heading in headings)


def _with_overlap(
    chunks: list[DocChunk],
    tokenizer: HuggingFaceTokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> list[DocChunk]:
    if overlap_tokens == 0 or len(chunks) < 2:
        return chunks

    hf_tokenizer = tokenizer.get_tokenizer()
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=False)
    result = [chunks[0]]

    for previous, current in zip(chunks, chunks[1:]):
        if previous.meta.headings != current.meta.headings:
            result.append(current)
            continue

        previous_ids = hf_tokenizer.encode(
            previous.text,
            add_special_tokens=False,
        )
        overlap_ids = previous_ids[-overlap_tokens:]
        overlapped: DocChunk | None = None

        while overlap_ids:
            decoded = cast(
                str,
                hf_tokenizer.decode(
                    overlap_ids,
                    skip_special_tokens=True,
                ),
            )
            prefix = decoded.strip()
            metadata = DocMeta(
                doc_items=previous.meta.doc_items + current.meta.doc_items,
                headings=current.meta.headings,
                captions=current.meta.captions,
                origin=current.meta.origin,
            )
            candidate = DocChunk(
                text=f"{prefix}\n{current.text}",
                meta=metadata,
            )
            contextualized = chunker.contextualize(candidate)
            if tokenizer.count_tokens(contextualized) <= max_tokens:
                overlapped = candidate
                break
            overlap_ids = overlap_ids[1:]

        result.append(overlapped or current)

    return result


def create_chunks(
    doc: DoclingDocument,
    model_name: str = CHUNKING.embedding_model,
    max_tokens: int = CHUNKING.max_tokens,
    overlap_tokens: int = CHUNKING.overlap_tokens,
    merge_peers: bool = True,
    skip_headings: set[str] = SKIP_HEADINGS,
) -> list[DocChunk]:
    if max_tokens < 1:
        raise ValueError("max_tokens muss mindestens 1 sein.")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens darf nicht negativ sein.")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens muss kleiner als max_tokens sein.")

    base_max_tokens = max_tokens - overlap_tokens
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        max_tokens=base_max_tokens,
    )

    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=merge_peers,
    )

    chunks = [
        cast(DocChunk, chunk)
        for chunk in chunker.chunk(dl_doc=doc)
        if not _is_skipped(
            cast(DocChunk, chunk),
            skip_headings,
        )
    ]
    return _with_overlap(
        chunks,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )


def contextualize_chunk(chunk: DocChunk) -> str:
    headings = chunk.meta.headings or []
    if not headings:
        return chunk.text
    return f"{' > '.join(headings)}\n{chunk.text}"
