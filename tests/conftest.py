"""Shared pytest configuration.

Tests marked ``slow`` run the real models (and download them on first use).
They are skipped by default so the fast suite runs anywhere; pass
``--run-slow`` to include them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SPOKEN = "Good evening everyone. Today we look at the quarterly budget."


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests marked slow (real models, downloads)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="slow test: pass --run-slow to enable")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def speech_wav(tmp_path: Path) -> Path:
    """A few seconds of synthesised English speech: Windows SAPI or macOS ``say``; skipped elsewhere.

    The macOS branch has not been run yet (the tool was built on Windows).
    """
    out = tmp_path / "speech.wav"
    if sys.platform == "win32":
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SetOutputToWaveFile('{out}'); "
            f"$s.Speak('{SPOKEN}'); "
            "$s.Dispose()"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, capture_output=True)
    elif sys.platform == "darwin":
        subprocess.run(["say", "-o", str(out), "--data-format=LEI16@22050", SPOKEN], check=True, capture_output=True)
    else:
        pytest.skip("speech synthesis fixture needs Windows SAPI or macOS say")
    return out
