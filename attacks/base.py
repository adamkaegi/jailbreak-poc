"""Abstract base class for attacks."""

from abc import ABC, abstractmethod


class BaseAttack(ABC):
    """Converts a plain behavior string into an Ollama-style messages list."""

    @abstractmethod
    def build_messages(self, behavior: str) -> list[dict]:
        """Return a messages list (role/content dicts) for the given behavior."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
