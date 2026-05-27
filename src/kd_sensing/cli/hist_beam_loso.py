from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kd_sensing.config import load_config
from kd_sensing.data.loso import default_loso_folds, resolve_loso_fold
from kd_sensing.engine.hist_beam_loso_execution import (
    DEFAULT_QUICK_BUDGETS,
    DEFAULT_QUICK_SEEDS,
    DEFAULT_QUICK_VARIANTS,
    execute_loso_run_plan,
)
from kd_sensing.utils.paths import output_dir as resolve_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or run HiST-Beam LOSO cross-scene adaptation.")
    parser.add_argument(
        "--config",
        "-c",
        default="configs/hist_beam/quick_smoke.yaml",
        help="Path to a HiST-Beam LOSO YAML config.",
    )
    parser.add_argument("--target-scene", type=int, help="Run a single target scene fold.")
    parser.add_argument("--source-scenes", help="Comma-separated explicit source scenes for a single fold.")
    parser.add_argument("--skip-scenes", default=None, help="Comma-separated scenes to skip, e.g. 34.")
    parser.add_argument("--variants", default=None, help="Comma-separated variant names.")
    parser.add_argument("--budgets", default=None, help="Comma-separated label budgets.")
    parser.add_argument("--seeds", default=None, help="Comma-separated random seeds.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit the planned matrix to the first N runs.")
    parser.add_argument("--output-dir", default=None, help="Directory for plan/execution metadata.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting metadata files.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing execution metadata where supported.")
    parser.add_argument("--execute", action="store_true", help="Execute stages instead of only writing a run plan.")
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
    print(json.dumps(result, indent=2))
    return 0


def run_main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    overrides = list(args.override or []) + [item for item in unknown if "=" in item]
    cfg = load_config(args.config, overrides)
    result = run_hist_beam_loso(cfg, args=args)
    return result


def run_hist_beam_loso(
    cfg: dict[str, Any],
    *,
    args: argparse.Namespace | None = None,
    stage_executor: Any | None = None,
) -> dict[str, Any]:
    loso_cfg = cfg.get("loso", {}) if isinstance(cfg.get("loso"), dict) else {}
    args = args or argparse.Namespace()
    target_scene = getattr(args, "target_scene", None)
    target_scenes = None if target_scene is not None else _parse_int_list(None, default=loso_cfg.get("target_scenes"))
    plan = build_loso_run_plan(
        cfg,
        target_scene=target_scene,
        target_scenes=target_scenes,
        source_scenes=_parse_optional_int_list(getattr(args, "source_scenes", None)),
        skip_scenes=_parse_int_list(getattr(args, "skip_scenes", None), default=loso_cfg.get("skip_scenes", [])),
        variants=_parse_str_list(getattr(args, "variants", None), default=loso_cfg.get("variants", DEFAULT_QUICK_VARIANTS)),
        budgets=_parse_int_list(getattr(args, "budgets", None), default=loso_cfg.get("budgets", DEFAULT_QUICK_BUDGETS)),
        seeds=_parse_int_list(getattr(args, "seeds", None), default=loso_cfg.get("seeds", DEFAULT_QUICK_SEEDS)),
        max_runs=_max_runs_value(getattr(args, "max_runs", None), loso_cfg.get("max_runs")),
    )
    out_dir = Path(getattr(args, "output_dir", None) or loso_cfg.get("output_dir") or resolve_output_dir("outputs/hist_beam_loso"))
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "loso_plan.json"
    if plan_path.exists() and not getattr(args, "overwrite", False):
        plan_path = _unique_path(plan_path)
    payload = {
        "mode": "execute" if getattr(args, "execute", False) else "plan_only",
        "config_name": cfg.get("experiment", {}).get("name"),
        "skip_scenes": plan["skip_scenes"],
        "run_count": len(plan["runs"]),
        "planned_run_count": plan.get("planned_run_count", len(plan["runs"])),
        "max_runs": plan.get("max_runs"),
        "runs": plan["runs"],
        "checkpoint_reuse": {
            "enabled": bool(loso_cfg.get("reuse_source_checkpoint", True)),
            "behavior": "reuse_existing_when_config_allows",
        },
    }
    with plan_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    payload["plan_path"] = str(plan_path)
    if getattr(args, "execute", False):
        payload["execution"] = execute_loso_run_plan(
            plan,
            cfg,
            output_dir=out_dir,
            overwrite=bool(getattr(args, "overwrite", False)),
            resume=bool(getattr(args, "resume", False)),
            stage_executor=stage_executor,
            plan_path=plan_path,
        )
    return payload


def build_loso_run_plan(
    cfg: dict[str, Any],
    *,
    target_scene: int | None = None,
    target_scenes: list[int] | None = None,
    source_scenes: list[int] | None = None,
    skip_scenes: list[int] | None = None,
    variants: list[str] | None = None,
    budgets: list[int] | None = None,
    seeds: list[int] | None = None,
    max_runs: int | None = None,
) -> dict[str, Any]:
    skipped = set(skip_scenes or [])
    if target_scene is not None:
        folds = [resolve_loso_fold(target_scene=target_scene, source_scenes=source_scenes)]
    elif target_scenes:
        folds = [resolve_loso_fold(target_scene=scene) for scene in target_scenes]
    else:
        folds = default_loso_folds()
    folds = [fold for fold in folds if fold.target_scene not in skipped and not set(fold.source_scenes) & skipped]
    runs = []
    for fold in folds:
        for variant in variants or ["v3_decoupled"]:
            for budget in budgets or [0]:
                for seed in seeds or [0]:
                    runs.append(
                        {
                            "fold": fold.fold_id,
                            "target_scene": fold.target_scene,
                            "source_scenes": list(fold.source_scenes),
                            "variant": str(variant),
                            "budget": int(budget),
                            "seed": int(seed),
                            "stages": [
                                "source_train",
                                "source_only_target_test_eval",
                                "target_adaptation",
                                "adapted_target_test_eval",
                                "summary",
                            ],
                            "target_test_for_training": False,
                        }
                    )
    planned_run_count = len(runs)
    if max_runs is not None:
        if int(max_runs) < 0:
            raise ValueError("max_runs must be non-negative.")
        runs = runs[: int(max_runs)]
    return {
        "skip_scenes": sorted(skipped),
        "runs": runs,
        "planned_run_count": planned_run_count,
        "max_runs": int(max_runs) if max_runs is not None else None,
        "enabled_modalities": list(cfg.get("model", {}).get("modalities", ["image", "radar", "gps"])),
    }


def _parse_int_list(raw: str | None, *, default: Any = None) -> list[int]:
    if raw is None:
        return list(default or [])
    if not str(raw).strip():
        return []
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def _parse_optional_int_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return _parse_int_list(raw)


def _parse_str_list(raw: str | None, *, default: Any = None) -> list[str]:
    if raw is None:
        return [str(item) for item in (default or [])]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _max_runs_value(cli_value: int | None, config_value: Any) -> int | None:
    value = cli_value if cli_value is not None else config_value
    if value is None or value == "":
        return None
    return int(value)


def _unique_path(path: Path) -> Path:
    index = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


if __name__ == "__main__":
    main()
