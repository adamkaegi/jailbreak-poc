"""Defense registry — add new defense classes here."""

from model_client import ModelClient
from defenses.base import BaseDefense
from defenses.none import NoneDefense
from defenses.self_reminder import SelfReminderDefense
from defenses.llama_guard import LlamaGuardDefense

REGISTRY: dict[str, type[BaseDefense]] = {
    "none": NoneDefense,
    "self_reminder": SelfReminderDefense,
    "llama_guard": LlamaGuardDefense,
}


def get_defense(
    name: str,
    client: ModelClient,
    guard_client: ModelClient | None = None,
) -> BaseDefense:
    """Instantiate a defense by name."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown defense '{name}'. Choose from: {list(REGISTRY)}")
    if name == "llama_guard":
        if guard_client is None:
            raise ValueError("The 'llama_guard' defense requires a guard_client.")
        return LlamaGuardDefense(client, guard_client)
    return REGISTRY[name](client)
