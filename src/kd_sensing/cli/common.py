import argparse
import json
from pathlib import Path
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


def bind_cli_mmw_protocol(
    cfg: dict,
    *,
    split_seed: int | None = None,
) -> None:
    dataset_type = str(cfg.get("data", {}).get("dataset", {}).get("type", "")).strip().lower()
    if dataset_type != "mmw":
        if split_seed is not None:
            raise ValueError("--split-seed is only valid for the MMW dataset.")
        return
    from kd_sensing.data.mmw.trajectory_protocol import (
        TRAJECTORY_PROTOCOL_MODE,
        bind_trajectory_config,
        trajectory_manifest_path,
    )

    data = cfg.setdefault("data", {})
    seed = int(data.get("split_seed", 0) if split_seed is None else split_seed)
    if seed < 0:
        raise ValueError("--split-seed must be non-negative.")
    configured = Path(str(data.get("split_manifest", "outputs"))).resolve()
    if configured.parent.name == TRAJECTORY_PROTOCOL_MODE and configured.parent.parent.name == "splits":
        manifest = trajectory_manifest_path(configured.parents[2], seed)
    elif configured.suffix.lower() == ".json":
        manifest = configured
    else:
        manifest = trajectory_manifest_path(configured, seed)
    data.update(split_seed=seed, split_manifest=str(manifest))
    bind_trajectory_config(cfg, manifest)
