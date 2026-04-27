from __future__ import annotations

import argparse
import json
from typing import Iterable

from kd_sensing.config import load_config


def collect_overrides(namespace: argparse.Namespace, unknown: Iterable[str]) -> list[str]:
    overrides = []
    for item in getattr(namespace, "override", []) or []:
        overrides.append(item)
    overrides.extend(item for item in unknown if "=" in item)
    return overrides


def print_result(result: dict) -> None:
    print(json.dumps(result, indent=2))


def load_cli_config(args: argparse.Namespace, unknown: Iterable[str]) -> dict:
    return load_config(args.config, collect_overrides(args, unknown))

