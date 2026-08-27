from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument


def _build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        mode=TableFormerMode.ACCURATE,
        do_cell_matching=True,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _cache_path(pdf_path: Path, cache_dir: Path) -> Path:
    return cache_dir / f"{pdf_path.stem}.json"


def _load_or_convert(
    pdf_path: Path,
    converter: DocumentConverter,
    cache_dir: Path,
    use_cache: bool,
) -> DoclingDocument:
    cache_file = _cache_path(pdf_path, cache_dir)

    if use_cache and cache_file.exists() and cache_file.stat().st_mtime > pdf_path.stat().st_mtime:
        print(f"Aus Cache: {pdf_path.name}")
        return DoclingDocument.load_from_json(cache_file)

    print(f"Konvertiere: {pdf_path.name}")
    document = converter.convert(pdf_path).document

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        document.save_as_json(cache_file)

    return document


def load_documents(
    folder_path: str,
    use_cache: bool = True,
    cache_dir: str = ".cache",
):
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Ordner nicht gefunden: {folder}")

    converter = _build_converter()
    cache_path = Path(cache_dir)

    documents = []

    for pdf_path in folder.glob("*.pdf"):
        document = _load_or_convert(pdf_path, converter, cache_path, use_cache)
        documents.append({"document": document, "source": pdf_path.name})

    return documents