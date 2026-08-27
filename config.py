from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    max_tokens: int = 512
    overlap_tokens: int = 32
    size_candidates: tuple[int, ...] = (256, 384, 512, 768)


@dataclass(frozen=True)
class RetrievalConfig:
    candidate_k: int = 20
    context_top_k: int = 5
    rrf_k: int = 5
    dense_weight: float = 2.0
    bm25_weight: float = 1.0
    use_stemming: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    request_timeout_seconds: int = 300
    max_context_chars: int = 12_000
    temperature: float = 0.1


CHUNKING = ChunkingConfig()
RETRIEVAL = RetrievalConfig()
GENERATION = GenerationConfig()
