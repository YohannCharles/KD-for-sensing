import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest

from kd_sensing.diagnostics.cli_surface import PUBLIC_CLI_HELP_SMOKE, PUBLIC_CLI_SURFACE


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


@pytest.mark.parametrize(
    ("command", "expected"),
    list(PUBLIC_CLI_HELP_SMOKE),
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


def test_public_cli_help_smoke_covers_pyproject_console_scripts():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    public_commands = {name for name in scripts if name.startswith("kd-sensing-")}

    assert set(PUBLIC_CLI_SURFACE) == public_commands
    assert {name for name, _expected in PUBLIC_CLI_HELP_SMOKE} == public_commands


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
        "jepa-visual-analysis",
        "jepa-gps-shortcut-benchmark",
        "training-throughput",
        "target-shot-split",
        "distribution-shift",
        "wcl2025-missing-modality-audit",
        "dataset-audit",
        "train-beambench-image-ae-gps",
        "run-beambench-image-ae-gps-tableiii",
        "tii-vlrg-transformer",
        "model-architecture-summary",
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
