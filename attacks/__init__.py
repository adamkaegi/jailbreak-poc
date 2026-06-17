"""Attack registry — add new attack classes here."""

from attacks.base import BaseAttack
from attacks.none import NoneAttack
from attacks.template import TemplateAttack

REGISTRY: dict[str, type[BaseAttack]] = {
    "none": NoneAttack,
    "template": TemplateAttack,
}


def get_attack(name: str) -> BaseAttack:
    """Instantiate an attack by name."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown attack '{name}'. Choose from: {list(REGISTRY)}")
    return REGISTRY[name]()
