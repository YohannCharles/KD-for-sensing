import argparse
import json
from pathlib import Path
from typing import Any

from kd_sensing.config.io import deep_merge, parse_overrides
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.engine.mmw_town_gps_v2 import run_mmw_town_gps_v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MMW Town GPS-only v2 circular scene adapter experiments.")
    parser.add_argument("--config", "-c", default="configs/mmw_town_gps_adapter_v2.yaml")
    parser.add_argument("--label-space", choices=("mapping_enabled", "mapping_disabled"), default=None)
    parser.add_argument("--target-scene", default=None, help="Scene name or slug; comma-separated values are accepted.")
    parser.add_argument("--support-ratio", type=float, default=None)
    parser.add_argument("--support-num", type=int, default=None)
    parser.add_argument("--support-mode", choices=("temporal_first", "angle_coverage", "random", "trajectory"), default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-logits", action="store_true", help="Write gps_logits.npy and gps_logits_index.csv.")
    parser.add_argument("--save-prior-probs", action="store_true", help="Also write gps_prior_probs.npy when logits are saved.")
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help="Override config value using dotted key=value syntax. Can be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    result = run_main(argv)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    overrides = list(args.override or []) + [item for item in unknown if "=" in item]
    cfg = _load_config(args.config, overrides)
    return run_mmw_town_gps_v2(
        cfg,
        label_space=args.label_space,
        target_scene=args.target_scene,
        support_ratio=args.support_ratio,
        support_num=args.support_num,
        support_mode=args.support_mode,
        output_dir=args.output_dir,
        save_logits=args.save_logits or None,
        save_prior_probs=args.save_prior_probs or None,
    )


def _load_config(path: str | Path, overrides: list[str]) -> dict[str, Any]:
    payload = safe_load_yaml(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"MMW Town GPS v2 config must be a mapping: {path}")
    if overrides:
        payload = deep_merge(payload, parse_overrides(overrides))
    return payload


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
