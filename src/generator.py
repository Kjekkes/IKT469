"""Generator wrapper around Ollama.

Single entry point `generate(prompt, system=...)` so we can swap providers
without touching the pipeline. Uses the official ollama python client; if
Ollama isn't running we raise a clear error.

`stream_generate` yields partial output as the model produces it; the GUI
uses this so the user sees text appearing instead of waiting 5-15 s for
the whole answer.
"""
from __future__ import annotations

from typing import Iterator, Optional

try:
    import ollama  # type: ignore
except ImportError as e:
    raise ImportError(
        "The `ollama` python package is required. Install with `pip install ollama`."
    ) from e


class OllamaGenerator:
    def __init__(self, model: str, temperature: float = 0.0, num_ctx: int = 4096):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx

    def _build_messages(self, prompt: str, system: Optional[str]) -> list[dict]:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        try:
            response = ollama.chat(
                model=self.model,
                messages=self._build_messages(prompt, system),
                options={"temperature": self.temperature, "num_ctx": self.num_ctx},
            )
        except Exception as e:
            raise RuntimeError(
                f"Ollama call failed. Is the Ollama daemon running and is "
                f"`{self.model}` pulled? Original error: {e}"
            ) from e
        return response["message"]["content"].strip()

    def stream_generate(self, prompt: str, system: Optional[str] = None) -> Iterator[str]:
        """Yield partial-text chunks as Ollama produces them.

        Each yielded value is a *cumulative* string so the caller can simply
        replace the previous one in the UI -- this matches Gradio's expected
        streaming contract for `ChatInterface`.
        """
        try:
            stream = ollama.chat(
                model=self.model,
                messages=self._build_messages(prompt, system),
                options={"temperature": self.temperature, "num_ctx": self.num_ctx},
                stream=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Ollama call failed. Is the Ollama daemon running and is "
                f"`{self.model}` pulled? Original error: {e}"
            ) from e
        buf = ""
        for chunk in stream:
            piece = chunk.get("message", {}).get("content", "")
            if not piece:
                continue
            buf += piece
            yield buf
