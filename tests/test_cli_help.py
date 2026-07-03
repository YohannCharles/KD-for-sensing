import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("kd-sensing-train", "--config"),
        ("kd-sensing-evaluate", "--weights"),
        ("kd-sensing-preprocess", "--action"),
        ("kd-sensing-runs", "--outputs"),
        ("kd-sensing-clean-runtime-artifacts", "--manifest"),
        ("kd-sensing-jepa-visual-analysis", "--analysis-config"),
        ("kd-sensing-jepa-gps-shortcut-benchmark", "--manifest"),
        ("kd-sensing-wcl2025-missing-modality-audit", "--output-root"),
        ("kd-sensing-mmw-town-gps-v2", "--config"),
        ("kd-sensing-plot-mmw-town-gps-v2", "--results-dir"),
        ("kd-sensing-compare-mmw-town-gps-v2", "--previous-dir"),
        ("kd-sensing-train-beambench-image-ae-gps", "--scene"),
        ("kd-sensing-run-beambench-image-ae-gps-tableiii", "--output-root"),
        ("kd-sensing-tii-vlrg-transformer", "--execute"),
        ("kd-sensing-inspect-mmw-physics", "--max-samples"),
        ("kd-sensing-model-architecture-summary", "--config"),
    ],
)
def test_console_script_help_is_available(command: str, expected: str):
    result = subprocess.run(
        _help_command(command),
        text=True,
        capture_output=True,
        check=False,
        env=_source_env(),
    )

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


def test_retired_top8_residual_cli_scripts_are_not_declared():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    retired_fragments = [
        "gps-coarse-anchor",
        "gps-window",
        "top8",
        "residual",
        "camera-ae",
        "bgam",
        "viewer-manifest",
        "visualize-modalities",
        "run-amr-net-gps-image",
        "run-jepa-msac",
    ]
    violations = [fragment for fragment in retired_fragments if fragment in text]

    assert violations == []


def test_organize_runtime_outputs_cli_help_declares_confirmation_flag():
    from kd_sensing.cli.organize_runtime_outputs import build_parser

    help_text = build_parser().format_help()

    assert "--manifest" in help_text
    assert "--confirm-organize" in help_text


def _help_command(command: str) -> list[str]:
    executable = shutil.which(command)
    if executable is not None:
        return [executable, "--help"]
    target = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"][command]
    module_name, function_name = target.split(":", 1)
    code = f"from {module_name} import {function_name} as _main; raise SystemExit(_main(['--help']))"
    return [sys.executable, "-c", code]


def _source_env() -> dict[str, str]:
    env = dict(os.environ)
    paths = [str(SRC)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env
