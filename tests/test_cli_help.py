from __future__ import annotations

from pathlib import Path
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
        ("kd-sensing-clean-runtime-artifacts", "--manifest"),
        ("kd-sensing-export-viewer-manifest", "--cache-dir"),
        ("kd-sensing-visualize-modalities", "--cache-dir"),
        ("kd-sensing-jepa-visual-analysis", "--analysis-config"),
        ("kd-sensing-jepa-gps-shortcut-benchmark", "--manifest"),
        ("kd-sensing-mmw-town-gps-v2", "--config"),
        ("kd-sensing-plot-mmw-town-gps-v2", "--results-dir"),
        ("kd-sensing-compare-mmw-town-gps-v2", "--previous-dir"),
        ("kd-sensing-prepare-mmw-town-gps-lidar-bgam-manifest", "--topk"),
        ("kd-sensing-run-mmw-town-gps-lidar-bgam", "--topk"),
        ("kd-sensing-evaluate-mmw-town-gps-lidar-bgam", "--ckpt"),
        ("kd-sensing-prepare-deepsense6g-gps-lidar-bgam-manifest", "--topk"),
        ("kd-sensing-run-deepsense6g-gps-lidar-bgam", "--bgam-mode"),
        ("kd-sensing-evaluate-deepsense6g-gps-lidar-bgam", "--ckpt"),
        ("kd-sensing-run-amr-net-gps-image", "--claim-status"),
        ("kd-sensing-run-jepa-msac", "--pretrained-checkpoint"),
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


def test_retired_top8_residual_cli_scripts_are_not_declared():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    retired_fragments = [
        "gps-coarse-anchor",
        "gps-window",
        "top8",
        "residual",
        "camera-ae",
    ]
    violations = [fragment for fragment in retired_fragments if fragment in text]

    assert violations == []


def test_organize_runtime_outputs_cli_help_declares_confirmation_flag():
    from kd_sensing.cli.organize_runtime_outputs import build_parser

    help_text = build_parser().format_help()

    assert "--manifest" in help_text
    assert "--confirm-organize" in help_text
