"""No-op defense: passes messages through to the model unchanged."""

from defenses.base import BaseDefense


class NoneDefense(BaseDefense):
    """Forwards messages directly to the model with no modification."""

    def query(self, messages: list[dict]) -> tuple[str, dict]:
        response = self.client.chat(messages)
        return response, {}
