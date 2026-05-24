from __future__ import annotations

import shutil
import subprocess

import pytest


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("kd-sensing-train", "--config"),
        ("kd-sensing-evaluate", "--weights"),
        ("kd-sensing-preprocess", "--action"),
        ("kd-sensing-runs", "--outputs"),
        ("kd-sensing-export-viewer-manifest", "--cache-dir"),
        ("kd-sensing-visualize-modalities", "--cache-dir"),
    ],
)
def test_console_script_help_is_available(command: str, expected: str):
    executable = shutil.which(command)
    assert executable is not None, f"{command} console script is not installed"

    result = subprocess.run(
        [executable, "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
