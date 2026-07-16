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


def test_only_core_cli_are_declared():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]

    assert scripts == {
        "kd-sensing-train": "kd_sensing.cli.train:main",
        "kd-sensing-evaluate": "kd_sensing.cli.evaluate:main",
        "kd-sensing-preprocess": "kd_sensing.cli.preprocess:main",
    }


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
