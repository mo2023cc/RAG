# RAG

This is a local Hybrid Retrieval-Augmented Generation system for structured PDF documents. 
It combines semantic and lexical retrieval and uses Llama 3.2 to generate answers grounded in retrieved document chunks.

The project intentionally avoids orchestration frameworks such as LangChain or LlamaIndex so that ingestion, chunking, retrieval, fusion, and generation remain transparent and easier to understand, modify, and evaluate.

## Features

- structured PDF extraction with Docling
- cached Docling parsing to avoid repeated PDF processing
- token-aware, structure-aware chunking with section-bound overlap
- local embeddings with `Qwen3-Embedding-0.6B`
- persistent Chroma vector storage
- Dense semantic retrieval
- BM25 lexical retrieval with German Snowball stemming
- Weighted Reciprocal Rank Fusion
- local generation with `llama3.2:3b` through Ollama
- Streamlit interface with sources and retrieval scores
- retrieval evaluation with Hit@1, Hit@5, and MRR

Every user query passes through retrieval before generation.. The LLM is instructed to answer only
from the retrieved PDF context.

## Architecture

The architecture follows a simple Hybrid RAG pipeline.

PDF documents are parsed with Docling and split into structured chunks. The chunks are embedded with Qwen3-Embedding-0.6B and stored in ChromaDB.

For each query, Dense Retrieval and BM25 search for relevant chunks. Their results are combined using Weighted Reciprocal Rank Fusion (RRF).

The top-ranked chunks are then passed as context to Llama 3.2, which generates the final answer based only on the retrieved information.


## Quick Start

### Requirements

- Python with `venv`
- Ollama

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
```

### Build the Index

```bash
python main.py index
```

This command converts all PDFs in `data/`, creates chunks and embeddings, and
stores them in ChromaDB. 
No PDFs are included in this repository. Feel free to use your own PDFs.

### Start the Application

Run Ollama and Streamlit in separate terminals:

```bash
ollama serve
```

```bash
streamlit run ui.py
```

### Test Retrieval in the Terminal

```bash
python main.py query \
  "What are the main requirements described in the documents?" \
  --retriever hybrid \
  --top-k 5
```


## Evaluation and Inspection

Evaluation questions and expected headings are defined in eval/questions.json.
`evaluation.py` evaluates Dense, BM25, and Hybrid retrieval using:

- Hit@1
- Hit@5
- Mean Reciprocal Rank
- top scores for unanswerable questions

```bash
python evaluation.py --retriever dense
python evaluation.py --retriever bm25
python evaluation.py --retriever hybrid
```

Indexed chunks can be inspected directly:

```bash
python inspect_chunks.py list
python inspect_chunks.py show 42
python inspect_chunks.py range 40 50
```

Different chunk sizes can be evaluated in separate Chroma collections:

```bash
python chunk_experiment.py \
  --sizes 256 384 512 768 \
  --overlap-tokens 32 \
  --retriever hybrid
```

## Main Files

| File | Purpose |
|---|---|
| `config.py` | central configuration |
| `ingestion.py` | PDF conversion and Docling cache |
| `chunk.py` | structure-aware chunking and overlap |
| `embedding.py` | document and query embeddings |
| `vector_store.py` | Chroma storage and indexing |
| `dense_retrieval.py` | semantic retrieval |
| `bm25_retrieval.py` | lexical retrieval |
| `hybrid_retrieval.py` | Weighted RRF |
| `generation.py` | context construction and RAG prompt |
| `llm_client.py` | Ollama communication |
| `ui.py` | Streamlit interface |
| `evaluation.py` | retrieval evaluation |

## Current Limitations and Next Steps

- RRF is the final ranking stage; no neural reranker is implemented yet.
- Multiple top results may come from the same document or section.
- Metadata does not yet include document IDs or section types.
- The context budget is character-based rather than token-based.
- Source citations are requested but not automatically validated.
- Generated answers are not evaluated systematically.
- Responses are not streamed.
- optimized for German-language documents using German Snowball stemming
