"""Thin wrapper around the Ollama Python client."""

import ollama


class ModelClient:
    """Synchronous chat client for a single Ollama model."""

    def __init__(self, model: str, base_url: str):
        self.model = model
        self._client = ollama.Client(host=base_url)

    def chat(self, messages: list[dict]) -> str:
        """Send a messages list and return the assistant reply text."""
        response = self._client.chat(model=self.model, messages=messages)
        return response.message.content

    def list_local_models(self) -> list[str]:
        """Return the names of all models currently pulled in Ollama."""
        result = self._client.list()
        return [m.model for m in result.models]
