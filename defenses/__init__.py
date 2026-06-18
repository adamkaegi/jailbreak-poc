"""Defense registry — add new defense classes here."""

from model_client import ModelClient
from defenses.base import BaseDefense
from defenses.none import NoneDefense
from defenses.self_reminder import SelfReminderDefense
from defenses.llama_guard import LlamaGuardDefense
from defenses.both import BothDefense

REGISTRY: dict[str, type[BaseDefense]] = {
    "none": NoneDefense,
    "self_reminder": SelfReminderDefense,
    "llama_guard": LlamaGuardDefense,
    "both": BothDefense,
}


def get_defense(
    name: str,
    client: ModelClient,
    guard_client: ModelClient | None = None,
) -> BaseDefense:
    """Instantiate a defense by name."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown defense '{name}'. Choose from: {list(REGISTRY)}")
    if name in {"llama_guard", "both"}:
        if guard_client is None:
            raise ValueError(f"The '{name}' defense requires a guard_client.")
        return REGISTRY[name](client, guard_client)
    return REGISTRY[name](client)
