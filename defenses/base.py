"""Abstract base class for defenses."""

from abc import ABC, abstractmethod

from model_client import ModelClient


class BaseDefense(ABC):
    """
    Wraps a ModelClient.  Subclasses may modify the input before querying,
    post-process the response, or both.
    """

    def __init__(self, client: ModelClient):
        self.client = client

    @abstractmethod
    def query(self, messages: list[dict]) -> tuple[str, dict]:
        """
        Query the model and return (response_text, metadata).
        metadata may include defense-specific keys (e.g. 'guard_label').
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
