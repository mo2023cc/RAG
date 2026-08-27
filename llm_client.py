import json
import os
from dataclasses import dataclass
from typing import TypedDict, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import GENERATION


class ChatMessage(TypedDict):
    role: str
    content: str


class OllamaError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaClient:
    model: str = GENERATION.ollama_model
    base_url: str = GENERATION.ollama_base_url
    timeout_seconds: int = GENERATION.request_timeout_seconds
    temperature: float = GENERATION.temperature

    @classmethod
    def from_environment(cls, model: str | None = None) -> "OllamaClient":
        return cls(
            model=model or os.getenv("OLLAMA_MODEL", GENERATION.ollama_model),
            base_url=os.getenv("OLLAMA_BASE_URL", GENERATION.ollama_base_url),
        )

    def chat(self, messages: list[ChatMessage]) -> str:
        if not self.model.strip():
            raise ValueError("Der Ollama-Modellname darf nicht leer sein.")

        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.temperature},
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_response = response.read().decode("utf-8")
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise OllamaError(
                f"Ollama antwortete mit HTTP {error.code}: {details}"
            ) from error
        except URLError as error:
            raise OllamaError(
                "Ollama ist nicht erreichbar. Starte den Dienst mit "
                "`ollama serve`."
            ) from error
        except TimeoutError as error:
            raise OllamaError("Die Anfrage an Ollama hat zu lange gedauert.") from error

        try:
            data = cast(dict[str, object], json.loads(raw_response))
            message = cast(dict[str, object], data["message"])
            content = message["content"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise OllamaError("Ollama lieferte eine ungültige Antwort.") from error

        if not isinstance(content, str) or not content.strip():
            raise OllamaError("Ollama lieferte keinen Antworttext.")
        return content.strip()
