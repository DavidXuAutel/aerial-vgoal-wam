"""Unit tests for semantic nav report metadata."""

from vgoal.report_meta import semantic_nav_report_fields


def test_required_fields():
    d = semantic_nav_report_fields(
        depth_source="airsim_depth",
        visual_prompt="chair",
        phase="P0",
    )
    assert d["method"] == "semantic_nav"
    assert d["goal_from"] == "vision"
    assert d["depth_source"] == "airsim_depth"
    assert d["visual_prompt"] == "chair"
    assert d["phase"] == "P0"
