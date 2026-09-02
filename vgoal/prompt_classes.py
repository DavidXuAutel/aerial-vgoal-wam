"""Parse fixed visual_prompt strings into detector class tokens."""
from __future__ import annotations


def classes_from_visual_prompt(prompt: str) -> list[str]:
    """Split a comma/semicolon prompt into non-empty class / phrase tokens.

    Examples:
        ``"red chair, person"`` → ``["red chair", "person"]``
        ``"chair"`` → ``["chair"]``
    """
    parts = [p.strip() for p in str(prompt).replace(";", ",").split(",")]
    return [p for p in parts if p]
