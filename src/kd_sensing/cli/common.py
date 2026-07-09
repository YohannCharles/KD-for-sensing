import argparse
import json
from typing import Iterable

from kd_sensing.config import load_config
from kd_sensing.data.scenes import DEEPSENSE_SCENES, resolve_deepsense_scene

DEFAULT_COMPARE_SCENES = (9, 32)


def collect_overrides(namespace: argparse.Namespace, unknown: Iterable[str]) -> list[str]:
    overrides = []
    overrides.extend(_scene_selection_overrides(getattr(namespace, "scenes", None)))
    for item in getattr(namespace, "override", []) or []:
        overrides.append(item)
    overrides.extend(item for item in unknown if "=" in item)
    return overrides


def print_result(result: dict) -> None:
    print(json.dumps(result, indent=2))


def load_cli_config(args: argparse.Namespace, unknown: Iterable[str]) -> dict:
    return load_config(args.config, collect_overrides(args, unknown))


def add_temporal_window_missing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--history_window", "--history-window", type=int, default=None)
    parser.add_argument("--prediction_window", "--prediction-window", type=int, default=None)
    parser.add_argument(
        "--temporal_aggregation",
        "--temporal-aggregation",
        choices=("last", "mean", "masked_mean", "flatten"),
        default=None,
    )
    parser.add_argument(
        "--temporal_missing_mode",
        "--temporal-missing-mode",
        choices=("none", "frame_bernoulli", "modality_frame_bernoulli", "block", "stratified_modality_temporal"),
        default=None,
    )
    parser.add_argument("--mask_sampler", "--mask-sampler", default=None)
    parser.add_argument("--train_missing_drop_counts", "--train-missing-drop-counts", default=None)
    parser.add_argument("--train_temporal_missing_rates", "--train-temporal-missing-rates", default=None)
    parser.add_argument("--train_temporal_missing_types", "--train-temporal-missing-types", default=None)
    parser.add_argument("--temporal_missing_prob", "--temporal-missing-prob", type=float, default=None)
    parser.add_argument("--temporal_missing_block_len", "--temporal-missing-block-len", type=int, default=None)
    parser.add_argument(
        "--temporal_missing_apply",
        "--temporal-missing-apply",
        choices=("train", "eval", "both"),
        default=None,
    )
    parser.add_argument("--temporal_missing_seed", "--temporal-missing-seed", type=int, default=None)
    parser.add_argument("--ensure_at_least_one_frame", "--ensure-at-least-one-frame", type=_bool_arg, default=None)
    parser.add_argument("--ensure_at_least_one_cell", "--ensure-at-least-one-cell", type=_bool_arg, default=None)
    parser.add_argument("--ensure_at_least_one_modality", "--ensure-at-least-one-modality", type=_bool_arg, default=None)
    parser.add_argument(
        "--ensure_at_least_one_modality_per_frame",
        "--ensure-at-least-one-modality-per-frame",
        type=_bool_arg,
        default=None,
    )


def apply_temporal_window_missing_cli_args(cfg: dict, args: argparse.Namespace) -> None:
    changed = False
    temporal = cfg.setdefault("temporal_missing", {})
    dataset = cfg.setdefault("data", {}).setdefault("dataset", {})
    model = cfg.setdefault("model", {})
    primary = model.setdefault("primary", {})
    if args.history_window is not None:
        changed = True
        value = int(args.history_window)
        temporal["history_window"] = value
        dataset["history_window"] = value
        dataset["seq_len"] = value
        model["history_window"] = value
        model["seq_length"] = value
        primary["history_window"] = value
        primary["seq_length"] = value
    if args.prediction_window is not None:
        changed = True
        value = int(args.prediction_window)
        temporal["prediction_window"] = value
        dataset["prediction_window"] = value
        dataset["num_pred"] = value
        model["prediction_window"] = value
        model["num_pred"] = value
        primary["prediction_window"] = value
        primary["num_pred"] = value
    mapping = {
        "temporal_aggregation": "temporal_aggregation",
        "temporal_missing_mode": "mode",
        "temporal_missing_prob": "prob",
        "temporal_missing_block_len": "block_len",
        "temporal_missing_apply": "apply",
        "temporal_missing_seed": "seed",
        "mask_sampler": "mask_sampler",
        "train_missing_drop_counts": "train_missing_drop_counts",
        "train_temporal_missing_rates": "train_temporal_missing_rates",
        "train_temporal_missing_types": "train_temporal_missing_types",
        "ensure_at_least_one_cell": "ensure_at_least_one_cell",
        "ensure_at_least_one_frame": "ensure_at_least_one_frame",
        "ensure_at_least_one_modality": "ensure_at_least_one_modality",
        "ensure_at_least_one_modality_per_frame": "ensure_at_least_one_modality_per_frame",
    }
    for attr, key in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            changed = True
            temporal[key] = value
    if temporal.get("mode", "none") != "none" or float(temporal.get("prob", 0.0) or 0.0) > 0.0:
        temporal["enabled"] = True
    if changed:
        from kd_sensing.config.normalization import normalize_temporal_window_missing_config
        from kd_sensing.data.difficulty.schema import _temporal_missing_profile, normalize_difficulty_profiles

        normalize_temporal_window_missing_config(cfg)
        profile = _temporal_missing_profile(cfg)
        if profile is not None:
            difficulty = cfg.setdefault("difficulty", {})
            profiles = [
                item
                for item in difficulty.get("profiles", [])
                if not isinstance(item, dict) or str(item.get("id", "")) != "temporal_missing"
            ]
            profiles.extend(
                profile.to_dict()
                for profile in normalize_difficulty_profiles(
                    [profile],
                    default_seed=cfg.get("experiment", {}).get("seed", 0),
                )
            )
            difficulty["enabled"] = bool(profiles)
            difficulty["profiles"] = profiles


def _bool_arg(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}.")


def _scene_selection_overrides(raw: str | None) -> list[str]:
    if raw is None or not str(raw).strip():
        return []
    scenes = _parse_scene_selection(raw)
    if len(scenes) == 1:
        return [
            f"data.dataset.scene={scenes[0]}",
            "diagnostics.visualization.compare_scenes=null",
        ]
    return [f"diagnostics.visualization.compare_scenes={json.dumps(scenes)}"]


def _parse_scene_selection(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        if token.lower() == "all":
            values.extend(scene for scene in DEFAULT_COMPARE_SCENES if scene in DEEPSENSE_SCENES)
            continue
        values.append(resolve_deepsense_scene(token).scene_id)
    if not values:
        raise ValueError("--scenes must include at least one scene, for example --scenes 9,32.")
    unique: list[int] = []
    for scene_id in values:
        if scene_id not in unique:
            unique.append(scene_id)
    return unique
