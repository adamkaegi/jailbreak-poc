"""Baseline attack: send the raw behavior with no jailbreak wrapper."""

from attacks.base import BaseAttack


class NoneAttack(BaseAttack):
    """No transformation — the behavior string becomes the user message verbatim."""

    def build_messages(self, behavior: str) -> list[dict]:
        return [{"role": "user", "content": behavior}]
