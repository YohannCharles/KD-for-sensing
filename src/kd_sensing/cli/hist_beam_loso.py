from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kd_sensing.config import load_config
from kd_sensing.data.loso import default_loso_folds, resolve_loso_fold
from kd_sensing.data.mmw.protocol import (
    build_mmw_folds,
    load_mmw_data_availability,
    mmw_scene_csv_names,
    mmw_scene_data_roots,
)
from kd_sensing.engine.hist_beam_loso_execution import (
    DEFAULT_QUICK_BUDGETS,
    DEFAULT_QUICK_SEEDS,
    DEFAULT_QUICK_VARIANTS,
    SENSOR_ASSISTED_QUICK_BUDGETS,
    SENSOR_ASSISTED_QUICK_SEEDS,
    SENSOR_ASSISTED_QUICK_VARIANTS,
    execute_loso_run_plan,
)
from kd_sensing.engine.modality_resolution import (
    SENSOR_ASSISTED_DISALLOWED_MODALITIES,
    SENSOR_ASSISTED_PROFILE,
    sensor_assisted_profile_enabled,
    resolve_enabled_modalities,
)
from kd_sensing.modalities import normalize_modalities
from kd_sensing.utils.paths import output_dir as resolve_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or run HiST-Beam LOSO cross-scene adaptation.")
    parser.add_argument(
        "--config",
        "-c",
        default="configs/hist_beam/quick_smoke.yaml",
        help="Path to a HiST-Beam LOSO YAML config.",
    )
    parser.add_argument("--target-scene", help="Run a single target scene fold. MMW accepts scenario slugs.")
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
    sensor_assisted = sensor_assisted_profile_enabled(cfg)
    args = args or argparse.Namespace()
    target_scene = _parse_scene_value(getattr(args, "target_scene", None))
    target_scenes = None if target_scene is not None else _parse_scene_list(None, default=loso_cfg.get("target_scenes"))
    plan = build_loso_run_plan(
        cfg,
        target_scene=target_scene,
        target_scenes=target_scenes,
        source_scenes=_parse_optional_scene_list(getattr(args, "source_scenes", None)),
        skip_scenes=_parse_scene_list(getattr(args, "skip_scenes", None), default=loso_cfg.get("skip_scenes", [])),
        variants=_parse_str_list(
            getattr(args, "variants", None),
            default=loso_cfg.get("variants", SENSOR_ASSISTED_QUICK_VARIANTS if sensor_assisted else DEFAULT_QUICK_VARIANTS),
        ),
        budgets=_parse_int_list(
            getattr(args, "budgets", None),
            default=loso_cfg.get("budgets", SENSOR_ASSISTED_QUICK_BUDGETS if sensor_assisted else DEFAULT_QUICK_BUDGETS),
        ),
        seeds=_parse_int_list(
            getattr(args, "seeds", None),
            default=loso_cfg.get("seeds", SENSOR_ASSISTED_QUICK_SEEDS if sensor_assisted else DEFAULT_QUICK_SEEDS),
        ),
        max_runs=_max_runs_value(getattr(args, "max_runs", None), loso_cfg.get("max_runs")),
        matrix_overrides=_matrix_override_metadata(args),
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
        "profile": plan.get("profile"),
        "matrix_scope": plan.get("matrix_scope"),
        "quick_validation": plan.get("quick_validation"),
        "modality_profile": plan.get("modality_profile"),
        "matrix": plan.get("matrix"),
        "matrix_overrides": plan.get("matrix_overrides", {}),
        "enabled_modalities": plan.get("enabled_modalities", []),
        "excluded_sensitive_fields": plan.get("excluded_sensitive_fields", []),
        "runs": plan["runs"],
        "dataset_family": plan.get("dataset_family", "DeepSense6G"),
        "claim_scope": plan.get("claim_scope", "cross_scene"),
        "cross_scene_claim_allowed": plan.get("cross_scene_claim_allowed", True),
        "data_availability": plan.get("data_availability"),
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
    target_scene: Any | None = None,
    target_scenes: list[Any] | None = None,
    source_scenes: list[Any] | None = None,
    skip_scenes: list[Any] | None = None,
    variants: list[str] | None = None,
    budgets: list[int] | None = None,
    seeds: list[int] | None = None,
    max_runs: int | None = None,
    matrix_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if _is_mmw_loso(cfg):
        return _build_mmw_run_plan(
            cfg,
            target_scene=target_scene,
            target_scenes=target_scenes,
            source_scenes=source_scenes,
            skip_scenes=skip_scenes,
            variants=variants,
            budgets=budgets,
            seeds=seeds,
            max_runs=max_runs,
            matrix_overrides=matrix_overrides,
        )
    try:
        enabled_modalities = list(resolve_enabled_modalities(cfg))
    except Exception:
        enabled_modalities = list(cfg.get("model", {}).get("modalities", ["image", "radar", "gps"]))
    profile = _matrix_profile(cfg)
    resolved_variants = list(variants or ["v3_decoupled"])
    resolved_budgets = list(budgets or [0])
    resolved_seeds = list(seeds or [0])
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
        for variant in resolved_variants:
            for budget in resolved_budgets:
                for seed in resolved_seeds:
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
                            "profile": profile,
                            "modality_profile": profile,
                            "enabled_modalities": enabled_modalities,
                            "excluded_sensitive_fields": _excluded_sensitive_fields(cfg),
                            "matrix_scope": "quick_validation" if profile else "standard",
                            "quick_validation": bool(profile),
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
        "enabled_modalities": enabled_modalities,
        "profile": profile,
        "modality_profile": {"profile": profile, "enabled_modalities": enabled_modalities},
        "matrix_scope": "quick_validation" if profile else "standard",
        "quick_validation": bool(profile),
        "matrix": _matrix_metadata(
            resolved_variants,
            resolved_budgets,
            resolved_seeds,
            matrix_scope="quick_validation" if profile else "standard",
        ),
        "matrix_overrides": dict(matrix_overrides or {}),
        "excluded_sensitive_fields": _excluded_sensitive_fields(cfg),
    }


def _is_mmw_loso(cfg: dict[str, Any]) -> bool:
    dataset_cfg = cfg.get("data", {}).get("dataset", {}) if isinstance(cfg.get("data"), dict) else {}
    loso_cfg = cfg.get("loso", {}) if isinstance(cfg.get("loso"), dict) else {}
    return str(dataset_cfg.get("type", "")).strip().lower() == "mmw" or str(loso_cfg.get("dataset_family", "")).upper() == "MMW"


def _build_mmw_run_plan(
    cfg: dict[str, Any],
    *,
    target_scene: Any | None,
    target_scenes: list[Any] | None,
    source_scenes: list[Any] | None,
    skip_scenes: list[Any] | None,
    variants: list[str] | None,
    budgets: list[int] | None,
    seeds: list[int] | None,
    max_runs: int | None,
    matrix_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loso_cfg = cfg.get("loso", {}) if isinstance(cfg.get("loso"), dict) else {}
    availability = load_mmw_data_availability(loso_cfg.get("data_availability_path"))
    folds = build_mmw_folds(availability, protocol=str(loso_cfg.get("protocol", "scenario_loso")))
    folds = _filter_mmw_folds(
        folds,
        target_scene=target_scene,
        target_scenes=target_scenes,
        source_scenes=source_scenes,
        skip_scenes=skip_scenes,
    )
    cfg.setdefault("loso", {})
    cfg["loso"].setdefault("scene_data_roots", mmw_scene_data_roots(availability))
    cfg["loso"].setdefault("scene_csv_names", mmw_scene_csv_names(availability))
    profile = _matrix_profile(cfg)
    sensor_assisted = sensor_assisted_profile_enabled(cfg)
    enabled_modalities = _resolve_loso_enabled_modalities(cfg)
    excluded_sensitive_fields = _excluded_sensitive_fields(cfg)
    matrix_scope = "quick_validation" if sensor_assisted else "full_or_configured"
    resolved_variants = list(variants or (SENSOR_ASSISTED_QUICK_VARIANTS if sensor_assisted else ["v3_decoupled"]))
    resolved_budgets = list(budgets or (SENSOR_ASSISTED_QUICK_BUDGETS if sensor_assisted else [0]))
    resolved_seeds = list(seeds or (SENSOR_ASSISTED_QUICK_SEEDS if sensor_assisted else [0]))
    runs = []
    for fold in folds:
        metadata = fold.metadata()
        for variant in resolved_variants:
            for budget in resolved_budgets:
                for seed in resolved_seeds:
                    runs.append(
                        {
                            "fold": fold.fold_id,
                            "target_scene": fold.target_scene,
                            "source_scenes": list(fold.source_scenes),
                            "variant": str(variant),
                            "budget": int(budget),
                            "seed": int(seed),
                            "dataset_family": "MMW",
                            "condition": fold.condition,
                            "town": fold.town,
                            "protocol": fold.protocol,
                            "claim_scope": fold.claim_scope,
                            "cross_scene_claim_allowed": fold.cross_scene_claim_allowed,
                            "stages": [
                                "source_train",
                                "source_only_target_test_eval",
                                "target_adaptation",
                                "adapted_target_test_eval",
                                "summary",
                            ],
                            "target_test_for_training": False,
                            "fold_metadata": metadata,
                            "profile": profile,
                            "modality_profile": profile,
                            "enabled_modalities": enabled_modalities,
                            "excluded_sensitive_fields": excluded_sensitive_fields,
                            "matrix_scope": matrix_scope,
                            "quick_validation": bool(sensor_assisted),
                        }
                    )
    planned_run_count = len(runs)
    if max_runs is not None:
        if int(max_runs) < 0:
            raise ValueError("max_runs must be non-negative.")
        runs = runs[: int(max_runs)]
    return {
        "skip_scenes": [],
        "runs": runs,
        "planned_run_count": planned_run_count,
        "max_runs": int(max_runs) if max_runs is not None else None,
        "enabled_modalities": enabled_modalities,
        "profile": profile,
        "modality_profile": {
            "profile": profile,
            "enabled_modalities": enabled_modalities,
            "excluded_sensitive_fields": excluded_sensitive_fields,
            "sensor_assisted": bool(sensor_assisted),
        },
        "matrix_scope": matrix_scope,
        "quick_validation": bool(sensor_assisted),
        "matrix": _matrix_metadata(resolved_variants, resolved_budgets, resolved_seeds, matrix_scope=matrix_scope),
        "matrix_overrides": dict(matrix_overrides or {}),
        "excluded_sensitive_fields": excluded_sensitive_fields,
        "dataset_family": "MMW",
        "claim_scope": availability.get("claim_scope", "unavailable"),
        "cross_scene_claim_allowed": bool(availability.get("cross_scene_claim_allowed", False)),
        "data_availability": {
            "ready_scenario_count": int(availability.get("ready_scenario_count", 0) or 0),
            "unavailable_reason": availability.get("unavailable_reason"),
        },
    }


def _filter_mmw_folds(
    folds: list[Any],
    *,
    target_scene: Any | None,
    target_scenes: list[Any] | None,
    source_scenes: list[Any] | None,
    skip_scenes: list[Any] | None,
) -> list[Any]:
    skipped = {str(scene) for scene in (skip_scenes or [])}
    targets = None
    if target_scene is not None:
        targets = {str(target_scene)}
    elif target_scenes:
        targets = {str(scene) for scene in target_scenes}

    selected = [
        fold
        for fold in folds
        if (targets is None or str(fold.target_scene) in targets)
        and str(fold.target_scene) not in skipped
        and not ({str(scene) for scene in fold.source_scenes} & skipped)
    ]
    if targets is not None:
        found = {str(fold.target_scene) for fold in selected}
        missing = sorted(targets - found)
        if missing:
            available = sorted(str(fold.target_scene) for fold in folds)
            raise ValueError(f"MMW target scene(s) {missing} are not ready. Available targets: {available}.")

    if source_scenes is None:
        return selected
    requested_sources = tuple(str(scene) for scene in source_scenes)
    if len(selected) != 1:
        raise ValueError("MMW --source-scenes requires exactly one selected target scene.")
    if not requested_sources:
        raise ValueError("MMW --source-scenes must contain at least one source scene.")
    target = str(selected[0].target_scene)
    if target in requested_sources:
        raise ValueError("MMW source/target scene must not overlap.")
    available_sources = {str(source) for fold in folds for source in fold.source_scenes}
    available_sources.update(str(fold.target_scene) for fold in folds)
    unknown = sorted(set(requested_sources) - available_sources)
    if unknown:
        raise ValueError(f"MMW source scene(s) {unknown} are not ready. Available scenes: {sorted(available_sources)}.")
    fold = selected[0]
    return [
        type(fold)(
            fold_id=fold.fold_id,
            target_scene=fold.target_scene,
            source_scenes=requested_sources,
            condition=fold.condition,
            town=fold.town,
            protocol=fold.protocol,
            claim_scope=fold.claim_scope,
            cross_scene_claim_allowed=fold.cross_scene_claim_allowed,
        )
    ]


def _parse_int_list(raw: str | None, *, default: Any = None) -> list[int]:
    if raw is None:
        return list(default or [])
    if not str(raw).strip():
        return []
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def _parse_scene_value(raw: Any) -> Any | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _parse_scene_list(raw: str | None, *, default: Any = None) -> list[Any]:
    if raw is None:
        return [_parse_scene_value(item) for item in (default or [])]
    if not str(raw).strip():
        return []
    return [_parse_scene_value(part.strip()) for part in str(raw).split(",") if part.strip()]


def _parse_optional_scene_list(raw: str | None) -> list[Any] | None:
    if raw is None:
        return None
    return _parse_scene_list(raw)


def _parse_str_list(raw: str | None, *, default: Any = None) -> list[str]:
    if raw is None:
        return [str(item) for item in (default or [])]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _max_runs_value(cli_value: int | None, config_value: Any) -> int | None:
    value = cli_value if cli_value is not None else config_value
    if value is None or value == "":
        return None
    return int(value)


def _matrix_override_metadata(args: argparse.Namespace) -> dict[str, Any]:
    result = {}
    for key in ("variants", "budgets", "seeds", "max_runs", "target_scene", "source_scenes", "skip_scenes"):
        value = getattr(args, key, None)
        if value is not None:
            result[key] = value
    return result


def _resolve_loso_enabled_modalities(cfg: dict[str, Any]) -> list[str]:
    if sensor_assisted_profile_enabled(cfg):
        return list(resolve_enabled_modalities(cfg))
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    for context, raw in (
        ("model.modalities", model_cfg.get("modalities")),
        ("model.student.modalities", model_cfg.get("student", {}).get("modalities") if isinstance(model_cfg.get("student"), dict) else None),
        ("model.teacher.modalities", model_cfg.get("teacher", {}).get("modalities") if isinstance(model_cfg.get("teacher"), dict) else None),
    ):
        if raw:
            return list(normalize_modalities(raw, context=context))
    return list(resolve_enabled_modalities(cfg))


def _matrix_profile(cfg: dict[str, Any]) -> str | None:
    loso_cfg = cfg.get("loso", {}) if isinstance(cfg.get("loso"), dict) else {}
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    dataset_cfg = cfg.get("data", {}).get("dataset", {}) if isinstance(cfg.get("data"), dict) else {}
    for value in (
        loso_cfg.get("profile"),
        loso_cfg.get("matrix_profile"),
        hist_cfg.get("profile"),
        dataset_cfg.get("modality_profile"),
    ):
        if value not in (None, ""):
            return str(value)
    if sensor_assisted_profile_enabled(cfg):
        return SENSOR_ASSISTED_PROFILE
    return None


def _excluded_sensitive_fields(cfg: dict[str, Any]) -> list[str]:
    if sensor_assisted_profile_enabled(cfg):
        return list(SENSOR_ASSISTED_DISALLOWED_MODALITIES)
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    configured = hist_cfg.get("excluded_sensitive_fields", [])
    return [str(item) for item in configured] if isinstance(configured, list) else []


def _matrix_metadata(
    variants: list[str],
    budgets: list[int],
    seeds: list[int],
    *,
    matrix_scope: str,
) -> dict[str, Any]:
    return {
        "variants": [str(item) for item in variants],
        "budgets": [int(item) for item in budgets],
        "seeds": [int(item) for item in seeds],
        "scope": matrix_scope,
        "is_full_budget_seed_sweep": matrix_scope != "quick_validation",
    }


def _unique_path(path: Path) -> Path:
    index = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


if __name__ == "__main__":
    main()
