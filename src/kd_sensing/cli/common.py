import argparse
import json
from typing import Iterable

from kd_sensing.config import load_config


def collect_overrides(namespace: argparse.Namespace, unknown: Iterable[str]) -> list[str]:
    return [*(getattr(namespace, "override", []) or []), *(item for item in unknown if "=" in item)]


def load_cli_config(args: argparse.Namespace, unknown: Iterable[str]) -> dict:
    return load_config(args.config, collect_overrides(args, unknown))


def print_result(result: dict) -> None:
    print(json.dumps(result, indent=2))
