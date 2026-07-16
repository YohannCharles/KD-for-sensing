"""Machine-readable public console script surface."""

from typing import NamedTuple


class PublicCli(NamedTuple):
    target: str
    lifecycle: str
    owner: str
    responsibility: str
    output_boundary: str
    focused_validation: str
    help_expected: str


PUBLIC_CLI_SURFACE: dict[str, PublicCli] = {
    "kd-sensing-train": PublicCli(
        "kd_sensing.cli.train:main",
        "core_workflow",
        "kd_sensing.engine.trainer",
        "config-driven training entrypoint",
        "ignored outputs/ and logs/ run roots",
        "conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q",
        "--config",
    ),
    "kd-sensing-evaluate": PublicCli(
        "kd_sensing.cli.evaluate:main",
        "core_workflow",
        "kd_sensing.engine.evaluation_pass",
        "checkpoint evaluation entrypoint",
        "ignored evaluation/output roots or user path",
        "conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q",
        "--weights",
    ),
    "kd-sensing-preprocess": PublicCli(
        "kd_sensing.cli.preprocess:main",
        "core_workflow",
        "kd_sensing.preprocessing",
        "config-driven preprocessing entrypoint",
        "dataset preparation targets or ignored cache/output roots",
        "conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q",
        "--action",
    ),
}

PUBLIC_CLI_HELP_SMOKE: tuple[tuple[str, str], ...] = tuple(
    (command, spec.help_expected) for command, spec in PUBLIC_CLI_SURFACE.items()
)

PUBLIC_CLI_LIFECYCLES = ("core_workflow",)


__all__ = ["PUBLIC_CLI_HELP_SMOKE", "PUBLIC_CLI_LIFECYCLES", "PUBLIC_CLI_SURFACE", "PublicCli"]
