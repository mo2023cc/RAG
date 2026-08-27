import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from vector_store import get_collection


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    heading: str
    pages: str


def _to_chunk(
    chunk_id: str,
    document: str | None,
    metadata: Mapping[str, object] | None,
) -> StoredChunk:
    metadata = metadata or {}
    return StoredChunk(
        chunk_id=chunk_id,
        text=document or "",
        source=str(metadata.get("source", "")),
        chunk_index=int(metadata.get("chunk_index", -1)),
        heading=str(metadata.get("heading", "")),
        pages=str(metadata.get("pages", "")),
    )


def load_chunks(source: str | None = None) -> list[StoredChunk]:
    collection = get_collection()
    if source is None:
        records = collection.get(include=["documents", "metadatas"])
    else:
        records = collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )

    documents = records["documents"] or []
    metadatas = records["metadatas"] or []
    chunks = [
        _to_chunk(chunk_id, document, metadata)
        for chunk_id, document, metadata in zip(
            records["ids"],
            documents,
            metadatas,
            strict=True,
        )
    ]
    return sorted(chunks, key=lambda chunk: (chunk.source, chunk.chunk_index))


def print_chunk_list(chunks: Sequence[StoredChunk]) -> None:
    if not chunks:
        print("Keine Chunks gefunden.")
        return

    for chunk in chunks:
        print(
            f"Index: {chunk.chunk_index:<3} | "
            f"Seiten: {chunk.pages or '-':<8} | "
            f"Heading: {chunk.heading or '-'} | "
            f"ID: {chunk.chunk_id[:12]}"
        )

    print(f"\n{len(chunks)} Chunk(s) gefunden.")


def print_chunk(chunk: StoredChunk) -> None:
    print(f"ID:      {chunk.chunk_id}")
    print(f"Quelle:  {chunk.source or '-'}")
    print(f"Index:   {chunk.chunk_index}")
    print(f"Heading: {chunk.heading or '-'}")
    print(f"Seiten:  {chunk.pages or '-'}")
    print("\n--- Text ---")
    print(chunk.text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Indexierte Chunks aus Chroma auflisten und untersuchen"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list",
        help="Übersicht der indexierten Chunks anzeigen",
    )
    list_parser.add_argument(
        "--source",
        help="Optional auf eine Quelldatei begrenzen",
    )

    show_parser = subparsers.add_parser(
        "show",
        help="Vollständigen Text eines Chunks anzeigen",
    )
    show_parser.add_argument("chunk_index", type=int)
    show_parser.add_argument(
        "--source",
        help="Optional auf eine Quelldatei begrenzen",
    )

    range_parser = subparsers.add_parser(
        "range",
        help="Vollständige Chunks innerhalb eines Indexbereichs anzeigen",
    )
    range_parser.add_argument("from_index", type=int, help="Erster Chunk-Index")
    range_parser.add_argument("to_index", type=int, help="Letzter Chunk-Index")
    range_parser.add_argument(
        "--source",
        help="Optional auf eine Quelldatei begrenzen",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = load_chunks(source=args.source)

    if args.command == "list":
        print_chunk_list(chunks)
        return

    if args.command == "show":
        matches = [
            chunk
            for chunk in chunks
            if chunk.chunk_index == args.chunk_index
        ]
        missing_message = f"Kein Chunk mit Index {args.chunk_index}"
    else:
        if args.from_index > args.to_index:
            raise SystemExit(
                "Der Startindex darf nicht größer als der Endindex sein."
            )
        matches = [
            chunk
            for chunk in chunks
            if args.from_index <= chunk.chunk_index <= args.to_index
        ]
        missing_message = (
            f"Keine Chunks von Index {args.from_index} "
            f"bis {args.to_index}"
        )

    if not matches:
        source_hint = f" in {args.source}" if args.source else ""
        raise SystemExit(f"{missing_message}{source_hint} gefunden.")

    for position, chunk in enumerate(matches):
        if position:
            print("\n" + "=" * 80 + "\n")
        print_chunk(chunk)


if __name__ == "__main__":
    main()
