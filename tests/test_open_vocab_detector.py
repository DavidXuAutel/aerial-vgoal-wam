"""Unit tests for OpenVocabPromptDetector (inner mock; no GPU weights)."""

import numpy as np

from vgoal.detector import MockDetector, OpenVocabPromptDetector


def test_open_vocab_uses_prompt_classes_with_inner_mock():
    inner = MockDetector(target_bbox=[10, 10, 40, 40], class_name="chair")
    det = OpenVocabPromptDetector(visual_prompt="chair", inner=inner)
    out = det.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    assert out is not None
    assert out.class_name == "chair"


def test_set_visual_prompt_updates_string():
    inner = MockDetector(target_bbox=[1, 1, 2, 2], class_name="door")
    det = OpenVocabPromptDetector(visual_prompt="chair", inner=inner)
    det.set_visual_prompt("door")
    assert det.visual_prompt == "door"
