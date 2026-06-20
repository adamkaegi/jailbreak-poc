"""Combined defense: self-reminder prompt plus Llama Guard output blocking."""

from defenses.base import BaseDefense
from defenses.llama_guard import BLOCKED_RESPONSE, GuardClassifier
from defenses.self_reminder import _SYSTEM_PROMPT
from model_client import ModelClient


class BothDefense(BaseDefense):
    """Adds a safety system prompt, then blocks unsafe responses with Llama Guard."""

    def __init__(self, client: ModelClient, guard_client: ModelClient):
        super().__init__(client)
        self._classifier = GuardClassifier(guard_client)

    def query(self, messages: list[dict]) -> tuple[str, dict]:
        system_msg = {"role": "system", "content": _SYSTEM_PROMPT}
        augmented = [system_msg] + list(messages)

        response, target_stats = self.client.chat_with_stats(augmented)
        label, guard_stats = self._classifier.classify_with_stats(augmented, response)
        final_response = BLOCKED_RESPONSE if label == "unsafe" else response

        return final_response, {
            "system_prompt_injected": True,
            "guard_label": label,
            "blocked_by_guard": label == "unsafe",
            "original_response": response,
            "response_replaced": label == "unsafe",
            "target_stats": target_stats,
            "guard_stats": guard_stats,
        }
