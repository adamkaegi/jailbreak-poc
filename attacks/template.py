"""Template-based jailbreak attack (DAN / refusal-suppression style)."""

from pathlib import Path

from attacks.base import BaseAttack

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "dan_template.txt"


class TemplateAttack(BaseAttack):
    """
    Wraps each behavior in a publicly documented DAN-style prompt template.
    The template is loaded from a text file at runtime; no harmful content
    is embedded in this source file.  See attacks/templates/dan_template.txt.
    """

    def __init__(self):
        self._template = _TEMPLATE_PATH.read_text()

    def build_messages(self, behavior: str) -> list[dict]:
        content = self._template.format(behavior=behavior)
        return [{"role": "user", "content": content}]
