"""Standard report fields for indoor semantic nav (Method B)."""
from __future__ import annotations

from typing import Any, Dict


def semantic_nav_report_fields(
    *,
    depth_source: str,
    visual_prompt: str,
    phase: str = "P0",
) -> Dict[str, Any]:
    """Return metadata that must appear on every semantic-nav artifact."""
    return {
        "method": "semantic_nav",
        "goal_from": "vision",
        "depth_source": str(depth_source),
        "visual_prompt": str(visual_prompt),
        "phase": str(phase),
    }
