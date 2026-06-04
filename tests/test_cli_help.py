from __future__ import annotations

import shutil
import subprocess
import sys

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
        ("kd-sensing-gps-window-baseline", "--execute"),
        ("kd-sensing-mmw-town-gps-v2", "--config"),
        ("kd-sensing-plot-mmw-town-gps-v2", "--results-dir"),
        ("kd-sensing-compare-mmw-town-gps-v2", "--previous-dir"),
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


@pytest.mark.parametrize(
    ("command", "module", "expected"),
    [
        (
            "kd-sensing-prepare-deepsense6g-gps-lidar-bgam-manifest",
            "kd_sensing.cli.prepare_deepsense6g_gps_lidar_bgam_manifest",
            "--support-ratio",
        ),
        (
            "kd-sensing-run-deepsense6g-gps-lidar-bgam",
            "kd_sensing.cli.run_deepsense6g_gps_lidar_bgam",
            "--bgam-mode",
        ),
        (
            "kd-sensing-evaluate-deepsense6g-gps-lidar-bgam",
            "kd_sensing.cli.evaluate_deepsense6g_gps_lidar_bgam",
            "--ckpt",
        ),
    ],
)
def test_gps_lidar_bgam_cli_help_is_available(command: str, module: str, expected: str):
    executable = shutil.which(command)
    argv = [executable, "--help"] if executable is not None else [sys.executable, "-m", module, "--help"]
    result = subprocess.run(argv, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--label-space" in result.stdout
    assert "--topk" in result.stdout
    assert expected in result.stdout
