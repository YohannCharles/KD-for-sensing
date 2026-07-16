import argparse
import json
from typing import Iterable

from kd_sensing.config import load_config


def collect_overrides(namespace: argparse.Namespace, unknown: Iterable[str]) -> list[str]:
    values = [*(getattr(namespace, "override", []) or []), *unknown]
    invalid = [item for item in values if not item or item.startswith("-") or "=" not in item]
    if invalid:
        raise ValueError(f"Overrides must use key=value without an option prefix: {', '.join(invalid)}")
    return values


def parse_cli_args(parser: argparse.ArgumentParser, argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    args, unknown = parser.parse_known_args(argv)
    invalid = [item for item in unknown if item.startswith("-") or "=" not in item]
    if invalid:
        parser.error(f"unrecognized arguments: {' '.join(invalid)}")
    return args, unknown


def load_cli_config(
    args: argparse.Namespace,
    unknown: Iterable[str],
    *,
    parser: argparse.ArgumentParser | None = None,
) -> dict:
    try:
        return load_config(args.config, collect_overrides(args, unknown))
    except ValueError as exc:
        if parser is not None:
            parser.error(str(exc))
        raise


def print_result(result: dict) -> None:
    print(json.dumps(result, indent=2))
