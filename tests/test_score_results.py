from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from score_results import holm_adjust


def test_holm_adjust_reorders_back_to_original_positions() -> None:
    adjusted = holm_adjust([0.04, 0.01, 0.03])
    assert adjusted == pytest.approx([0.06, 0.03, 0.06])


def test_holm_adjust_handles_ties_monotonically() -> None:
    adjusted = holm_adjust([0.01, 0.01, 0.04])
    assert adjusted == pytest.approx([0.03, 0.03, 0.04])


def test_holm_adjust_preserves_none_inputs() -> None:
    adjusted = holm_adjust([0.01, None, 0.04])
    assert adjusted[0] == pytest.approx(0.02)
    assert adjusted[1] is None
    assert adjusted[2] == pytest.approx(0.04)
