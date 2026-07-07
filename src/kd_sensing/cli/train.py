import argparse

from kd_sensing.cli.common import load_cli_config, print_result
from kd_sensing.engine.trainer import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a sensing or beam prediction model from a YAML config.")
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML config file.")
    parser.add_argument("--resume", help="Resume from an explicit checkpoint path.")
    parser.add_argument("--auto-resume", "--auto_resume", action="store_true", help="Resume from this run's checkpoints/last.pth.")
    parser.add_argument("--num-workers", "--num_workers", type=int, help="Set train/test DataLoader workers.")
    parser.add_argument("--prefetch-factor", "--prefetch_factor", type=int, help="Set train/test DataLoader prefetch factor.")
    parser.add_argument("--persistent-workers", "--persistent_workers", action="store_true", help="Enable persistent DataLoader workers.")
    parser.add_argument("--no-persistent-workers", "--no_persistent_workers", action="store_true", help="Disable persistent DataLoader workers.")
    parser.add_argument("--pin-memory", "--pin_memory", action="store_true", help="Enable DataLoader pin_memory.")
    parser.add_argument("--no-pin-memory", "--no_pin_memory", action="store_true", help="Disable DataLoader pin_memory.")
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help="Override config value using dotted key=value syntax. Can be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one synthetic epoch by overriding dataset, epochs, workers, and output settings.",
    )
    parser.add_argument("--fusion", choices=("weighted_sum", "average", "raw_conf_gate", "bprr", "pcpg", "supervised_router"))
    parser.add_argument("--router_supervision", "--router-supervision", choices=("oracle", "none", "pattern_best"))
    parser.add_argument("--router_distill_weight", "--router-distill-weight", type=float)
    parser.add_argument("--router_focus_patterns", "--router-focus-patterns")
    parser.add_argument("--router_fuse_level", "--router-fuse-level", default=None)
    for name in (
        "router_use_pattern_features",
        "router_use_reliability_features",
        "router_use_prototype_margin",
        "router_use_entropy",
        "router_use_confidence",
        "router_use_logit_norm",
        "use_beam_prototype_alignment",
        "use_modality_prototype_loss",
        "use_circular_soft_targets",
        "use_gaussian_beam_targets",
        "use_jepa",
        "branch_aux_loss",
        "radar_protect_loss",
    ):
        parser.add_argument(f"--{name}", f"--{name.replace('_', '-')}", type=_bool_arg, default=None)
    parser.add_argument("--beam_proto_align_weight", "--beam-proto-align-weight", type=float)
    parser.add_argument("--modality_proto_weight", "--modality-proto-weight", type=float)
    parser.add_argument("--head_type", "--head-type", choices=("prototype", "classifier"))
    parser.add_argument("--hard_subset_weighting", "--hard-subset-weighting", choices=("none", "static", "soft_static"))
    parser.add_argument("--jepa_weight", "--jepa-weight", type=float)
    parser.add_argument("--unimodal_aux_weight", "--unimodal-aux-weight", type=float)
    parser.add_argument("--radar_aux_weight", "--radar-aux-weight", type=float)
    parser.add_argument("--bprr_calibration", "--bprr-calibration", choices=("none", "temperature"))
    return parser


def run(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    cfg = load_cli_config(args, unknown)
    _apply_training_cli_shortcuts(cfg, args)
    cfg.setdefault("runtime", {})["cli_config_path"] = args.config
    if args.dry_run:
        cfg["data"]["dataset"]["type"] = "synthetic"
        for key in ("train_scenes", "test_scenes", "eval_scenes", "validation_scenes"):
            cfg["data"]["dataset"].pop(key, None)
        cfg["data"]["dataset"]["length"] = 2
        cfg["data"]["dataloader"]["train_batch_size"] = 1
        cfg["data"]["dataloader"]["test_batch_size"] = 1
        cfg["data"]["dataloader"]["num_workers"] = 0
        cfg["training"]["epochs"] = 1
        cfg["training"]["use_early_stopping"] = False
        cfg["output"]["run_name"] = f"{cfg['experiment']['name']}_dry_run"
    result = train(cfg)
    print_result(result)
    return result


def _apply_training_cli_shortcuts(cfg: dict, args: argparse.Namespace) -> None:
    training = cfg.setdefault("training", {})
    loader = cfg.setdefault("data", {}).setdefault("dataloader", {})
    if args.resume and args.auto_resume:
        raise ValueError("Use either --resume PATH or --auto-resume, not both.")
    if args.resume:
        training["resume"] = args.resume
    elif args.auto_resume:
        training["resume"] = True
    if args.num_workers is not None:
        loader["num_workers"] = int(args.num_workers)
        loader["train_num_workers"] = int(args.num_workers)
        loader["test_num_workers"] = int(args.num_workers)
    if args.prefetch_factor is not None:
        loader["prefetch_factor"] = int(args.prefetch_factor)
        loader["train_prefetch_factor"] = int(args.prefetch_factor)
        loader["test_prefetch_factor"] = int(args.prefetch_factor)
    if args.persistent_workers and args.no_persistent_workers:
        raise ValueError("Use either --persistent-workers or --no-persistent-workers, not both.")
    if args.persistent_workers or args.no_persistent_workers:
        value = bool(args.persistent_workers)
        loader["persistent_workers"] = value
        loader["train_persistent_workers"] = value
        loader["test_persistent_workers"] = value
    if args.pin_memory and args.no_pin_memory:
        raise ValueError("Use either --pin-memory or --no-pin-memory, not both.")
    if args.pin_memory or args.no_pin_memory:
        value = bool(args.pin_memory)
        loader["pin_memory"] = value
        loader["train_pin_memory"] = value
        loader["test_pin_memory"] = value
    _apply_final_ablation_cli_flags(cfg, args)


def _apply_final_ablation_cli_flags(cfg: dict, args: argparse.Namespace) -> None:
    primary = cfg.setdefault("model", {}).setdefault("primary", {})
    loss = cfg.setdefault("loss", {})
    training = cfg.setdefault("training", {})
    if args.fusion is not None:
        primary["fusion_type"] = args.fusion
    if args.bprr_calibration is not None:
        primary["bprr_calibration"] = args.bprr_calibration
    for key in (
        "router_supervision",
        "router_distill_weight",
        "router_focus_patterns",
        "router_fuse_level",
        "router_use_pattern_features",
        "router_use_reliability_features",
        "router_use_prototype_margin",
        "router_use_entropy",
        "router_use_confidence",
        "router_use_logit_norm",
    ):
        value = getattr(args, key, None)
        if value is not None:
            primary[key] = value
            loss[key] = value
    if args.head_type is not None:
        primary["head_type"] = args.head_type
    for key in (
        "use_beam_prototype_alignment",
        "use_modality_prototype_loss",
        "use_circular_soft_targets",
        "use_gaussian_beam_targets",
    ):
        value = getattr(args, key, None)
        if value is not None:
            primary[key] = value
            training[key] = value
    if args.beam_proto_align_weight is not None:
        training["beam_proto_align_weight"] = float(args.beam_proto_align_weight)
        training["lambda_proto"] = float(args.beam_proto_align_weight)
    if args.modality_proto_weight is not None:
        training["modality_proto_weight"] = float(args.modality_proto_weight)
        training["lambda_modality_proto"] = float(args.modality_proto_weight)
    if args.hard_subset_weighting is not None:
        mode = str(args.hard_subset_weighting)
        loss["hard_subset_weighting"] = {"enabled": mode != "none", "mode": mode}
        loss.setdefault("pcpg_radar_balance", {})["enabled"] = mode != "none" or bool(loss.get("pcpg_radar_balance", {}).get("enabled", False))
    for key in ("use_jepa", "branch_aux_loss", "radar_protect_loss"):
        value = getattr(args, key, None)
        if value is not None:
            loss[key] = value
            if key == "use_jepa":
                primary["use_jepa_loss"] = value
    for key in ("jepa_weight", "unimodal_aux_weight", "radar_aux_weight"):
        value = getattr(args, key, None)
        if value is not None:
            loss[key] = float(value)
    if any(getattr(args, key, None) is not None for key in ("branch_aux_loss", "radar_protect_loss", "unimodal_aux_weight", "radar_aux_weight", "jepa_weight")):
        loss.setdefault("pcpg_radar_balance", {})["enabled"] = True


def _bool_arg(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}.")


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
