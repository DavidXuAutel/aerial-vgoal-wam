"""Unit tests for visual_prompt → class tokens."""

from vgoal.prompt_classes import classes_from_visual_prompt


def test_splits_common_prompt():
    assert classes_from_visual_prompt("red chair, person") == ["red chair", "person"]


def test_strips_empty():
    assert classes_from_visual_prompt("  chair ,,  ") == ["chair"]


def test_semicolon():
    assert classes_from_visual_prompt("door; chair") == ["door", "chair"]
