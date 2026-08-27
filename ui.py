import os
from typing import TypedDict

import streamlit as st

from bm25_retrieval import BM25Result
from config import GENERATION, RETRIEVAL
from dense_retrieval import SearchResult, search as dense_search
from embedding import embed_query, load_embedding_model
from generation import generate_answer
from hybrid_retrieval import HybridResult, HybridRetriever
from llm_client import OllamaClient, OllamaError
from vector_store import get_collection


st.set_page_config(
    page_title="localRAG",
)


RetrievalResult = SearchResult | BM25Result | HybridResult


class SourceData(TypedDict):
    heading: str
    source: str
    pages: str
    text: str


class StoredMessage(TypedDict):
    role: str
    content: str
    sources: list[SourceData]


@st.cache_resource(show_spinner="Embedding-Modell wird geladen …")
def load_resources():
    model = load_embedding_model()
    collection = get_collection()
    hybrid = HybridRetriever(collection, model)
    return model, collection, hybrid


model, collection, hybrid = load_resources()


def format_retriever_name(retriever_name: str) -> str:
    labels = {
        "hybrid": "Hybrid (Dense + BM25 mit RRF)",
        "dense": "Dense",
        "bm25": "BM25",
    }
    return labels[retriever_name]


with st.sidebar:
    st.header("Einstellungen")
    st.caption(f"{collection.count()} Chunks im Chroma-Index")
    ollama_model = st.text_input(
        "Ollama-Modell",
        value=os.getenv("OLLAMA_MODEL", GENERATION.ollama_model),
    )
    retriever = st.selectbox(
        "Retrieval-Methode",
        options=("hybrid", "dense", "bm25"),
        format_func=format_retriever_name,
    )
    top_k = st.slider(
        "Chunks für das LLM",
        min_value=1,
        max_value=10,
        value=RETRIEVAL.context_top_k,
    )
    if st.button("Chat löschen"):
        st.session_state.messages = []


def result_score(result: SearchResult | BM25Result | HybridResult) -> str:
    if isinstance(result, SearchResult):
        return f"Ähnlichkeit {result.similarity:.3f}"
    if isinstance(result, BM25Result):
        return f"BM25 {result.score:.3f}"
    return (
        f"RRF {result.rrf_score:.4f} "
        f"(Dense #{result.dense_rank or '–'}, "
        f"BM25 #{result.bm25_rank or '–'})"
    )


def render_sources(sources: list[SourceData]) -> None:
    for rank, source in enumerate(sources, start=1):
        title = f"[{rank}] {source['heading'] or 'Ohne Überschrift'}"
        with st.expander(title):
            st.caption(
                f"Quelle: {source['source'] or '-'} · "
                f"Seite(n): {source['pages'] or '-'}"
            )
            st.write(source["text"])


def search_documents(query: str) -> list[RetrievalResult]:
    if retriever == "hybrid":
        return list(hybrid.search(query, top_k=top_k))
    if retriever == "bm25":
        return list(hybrid.bm25.search(query, top_k=top_k))

    query_embedding = embed_query(model, query)
    return list(dense_search(collection, query_embedding, top_k=top_k))


if "messages" not in st.session_state:
    st.session_state.messages = []

messages: list[StoredMessage] = st.session_state.messages
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["sources"]:
            render_sources(message["sources"])

question = st.chat_input(
    "Stelle eine Frage zu den indexierten PDF-Dokumenten"
)

if question:
    user_message: StoredMessage = {
        "role": "user",
        "content": question,
        "sources": [],
    }
    messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        results: list[RetrievalResult] = []
        sources: list[SourceData] = []
        with st.spinner("Suche relevante Abschnitte …"):
            results = search_documents(question)

        sources = [
            {
                "heading": result.heading,
                "source": result.source,
                "pages": result.pages,
                "text": result.text,
            }
            for result in results
        ]

        if not results:
            answer = (
                "Dazu enthalten die indexierten Dokumente keine ausreichenden "
                "Informationen."
            )
            st.warning(answer)
        else:
            try:
                client = OllamaClient.from_environment(model=ollama_model)
                with st.spinner(f"{ollama_model} formuliert die Antwort …"):
                    answer = generate_answer(client, question, results)
                st.markdown(answer)
            except (OllamaError, ValueError) as error:
                answer = f"Die Antwort konnte nicht erzeugt werden: {error}"
                st.error(answer)

        if results:
            st.caption(
                "Verwendete Retrieval-Treffer: "
                + ", ".join(result_score(result) for result in results)
            )
            render_sources(sources)

        assistant_message: StoredMessage = {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
        messages.append(assistant_message)
