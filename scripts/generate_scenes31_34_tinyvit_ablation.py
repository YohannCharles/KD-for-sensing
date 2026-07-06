#!/usr/bin/env python3

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from kd_sensing.config.io import load_config

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scene31_generator_common import rel, truthy  # noqa: E402


DEFAULT_OUT_DIR = "outputs/scenes31_34_tinyvit_lmdb/generated_configs"
DEFAULT_OUTPUT_DIR = "outputs/scenes31_34_tinyvit_lmdb"
DEFAULT_SCENES = [31, 32, 33, 34]
DEFAULT_TINYVIT_ENCODER = "tinyvit_5m_scratch_rgb"
PRETRAIN_EPOCHS = 100
DOWNSTREAM_EPOCHS = 40
IMAGE_PRETRAIN_BASE = ROOT / "configs/fusion/experiments/m2beam_single_modal_scene31/image.yaml"
LIDAR_PRETRAIN_BASE = ROOT / "configs/fusion/experiments/m2beam_single_modal_scene31/lidar.yaml"
DOWNSTREAM_BASE = ROOT / "configs/scene31/templates/main_v3_proto_es20_base.yaml"
MODALITIES = ["image", "radar", "gps", "lidar"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Scene31-34 TinyViT encoder ablation configs.")
    parser.add_argument("--out-dir", "--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-dir", "--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenes", default="31,32,33,34")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tinyvit-encoder", default=DEFAULT_TINYVIT_ENCODER)
    parser.add_argument("--pretrain-epochs", type=int, default=PRETRAIN_EPOCHS)
    parser.add_argument("--downstream-epochs", type=int, default=DOWNSTREAM_EPOCHS)
    parser.add_argument("--overwrite", default="false")
    args = parser.parse_args(argv)

    scenes = _parse_scenes(args.scenes)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _build_rows(
        out_dir=out_dir,
        output_dir=str(args.output_dir),
        scenes=scenes,
        seed=int(args.seed),
        tinyvit_encoder=str(args.tinyvit_encoder),
        pretrain_epochs=int(args.pretrain_epochs),
        downstream_epochs=int(args.downstream_epochs),
    )
    _write_configs(rows, overwrite=truthy(args.overwrite))
    _write_manifest(out_dir, rows)
    _write_budget_manifest(
        out_dir,
        output_dir=str(args.output_dir),
        scenes=scenes,
        rows=rows,
        pretrain_epochs=int(args.pretrain_epochs),
        downstream_epochs=int(args.downstream_epochs),
    )
    print(f"Wrote {len(rows)} Scene31-34 TinyViT ablation manifest rows to {out_dir}.")
    return 0


def _parse_scenes(value: str) -> list[int]:
    scenes = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not scenes:
        raise ValueError("--scenes must contain at least one scene id")
    return scenes


def _build_rows(
    *,
    out_dir: Path,
    output_dir: str,
    scenes: list[int],
    seed: int,
    tinyvit_encoder: str,
    pretrain_epochs: int,
    downstream_epochs: int,
) -> list[dict[str, Any]]:
    image_run = f"scenes31_34_tinyvit_image_pretrain_seed{seed}"
    lidar_run = f"scenes31_34_tinyvit_lidar_pretrain_seed{seed}"
    plain_run = f"scenes31_34_proto_randomdrop_subset_tinyvit_es40_seed{seed}"
    jepa_run = f"scenes31_34_proto_randomdrop_subset_tinyvit_jepa_es40_seed{seed}"
    image_checkpoint = _checkpoint_path(output_dir, scenes, image_run)
    lidar_checkpoint = _checkpoint_path(output_dir, scenes, lidar_run)
    specs = [
        {
            "run_name": image_run,
            "group": "tinyvit_pretrain",
            "stage": "pretrain",
            "seed": seed,
            "method_tags": "scenes31_34,tinyvit,image_pretrain,single_modal",
            "expected_epochs": pretrain_epochs,
            "priority": "high",
            "execution_mode": "train",
            "config": _pretrain_config(
                base_path=IMAGE_PRETRAIN_BASE,
                run_name=image_run,
                output_dir=output_dir,
                scenes=scenes,
                seed=seed,
                modality="image",
                tinyvit_encoder=tinyvit_encoder,
                epochs=pretrain_epochs,
            ),
        },
        {
            "run_name": lidar_run,
            "group": "tinyvit_pretrain",
            "stage": "pretrain",
            "seed": seed,
            "method_tags": "scenes31_34,tinyvit,lidar_pretrain,single_modal",
            "expected_epochs": pretrain_epochs,
            "priority": "high",
            "execution_mode": "train",
            "config": _pretrain_config(
                base_path=LIDAR_PRETRAIN_BASE,
                run_name=lidar_run,
                output_dir=output_dir,
                scenes=scenes,
                seed=seed,
                modality="lidar",
                tinyvit_encoder=tinyvit_encoder,
                epochs=pretrain_epochs,
            ),
        },
        {
            "run_name": plain_run,
            "group": "tinyvit_downstream",
            "stage": "downstream",
            "seed": seed,
            "method_tags": "scenes31_34,proto,randomdrop_subset,tinyvit,es40",
            "expected_epochs": downstream_epochs,
            "priority": "high",
            "execution_mode": "train",
            "requires": f"{image_checkpoint};{lidar_checkpoint}",
            "config": _downstream_config(
                run_name=plain_run,
                output_dir=output_dir,
                scenes=scenes,
                seed=seed,
                tinyvit_encoder=tinyvit_encoder,
                image_checkpoint=image_checkpoint,
                lidar_checkpoint=lidar_checkpoint,
                epochs=downstream_epochs,
                use_jepa=False,
            ),
        },
        {
            "run_name": jepa_run,
            "group": "tinyvit_downstream_jepa",
            "stage": "downstream",
            "seed": seed,
            "method_tags": "scenes31_34,proto,randomdrop_subset,tinyvit,jepa,es40",
            "expected_epochs": downstream_epochs,
            "priority": "high",
            "execution_mode": "train",
            "requires": f"{image_checkpoint};{lidar_checkpoint}",
            "config": _downstream_config(
                run_name=jepa_run,
                output_dir=output_dir,
                scenes=scenes,
                seed=seed,
                tinyvit_encoder=tinyvit_encoder,
                image_checkpoint=image_checkpoint,
                lidar_checkpoint=lidar_checkpoint,
                epochs=downstream_epochs,
                use_jepa=True,
            ),
        },
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        config_path = out_dir / f"{spec['run_name']}.yaml"
        rows.append(
            {
                **spec,
                "config_path": rel(config_path),
                "path": config_path,
            }
        )
    return rows


def _pretrain_config(
    *,
    base_path: Path,
    run_name: str,
    output_dir: str,
    scenes: list[int],
    seed: int,
    modality: str,
    tinyvit_encoder: str,
    epochs: int,
) -> dict[str, Any]:
    cfg = load_config(base_path)
    cfg["experiment"]["name"] = run_name
    cfg["experiment"]["seed"] = int(seed)
    cfg["model"]["modalities"] = [modality]
    cfg["model"]["primary"]["modalities"] = [modality]
    cfg["model"]["primary"]["encoders"] = {modality: _tinyvit_encoder_cfg(tinyvit_encoder, trainable=True)}
    cfg["training"].update(
        {
            "epochs": int(epochs),
            "max_epochs": int(epochs),
            "save_best_metric": "val_top1",
            "validation": {"interval_epochs": 1},
        }
    )
    cfg["data"]["dataset"].update(_scene_dataset_cfg(scenes, seq_len=8, num_pred=3, cache_enabled=False))
    if modality == "image":
        cfg["data"]["dataset"].update({"image_profile": "rgb_imagenet", "image_size": [224, 224]})
    if modality == "lidar":
        cfg["data"]["dataset"].update(
            {
                "use_lidar": True,
                "lidar_bev_size": [224, 224],
                "lidar_normalize": False,
                "lidar_normalization": {"enabled": False, "mode": "none"},
            }
        )
        cfg["model"]["primary"]["lidar_channels"] = 3
    cfg["output"]["dir"] = str(output_dir)
    cfg["output"]["run_name"] = run_name
    cfg["output"]["group_by_scene"] = True
    return cfg


def _downstream_config(
    *,
    run_name: str,
    output_dir: str,
    scenes: list[int],
    seed: int,
    tinyvit_encoder: str,
    image_checkpoint: str,
    lidar_checkpoint: str,
    epochs: int,
    use_jepa: bool,
) -> dict[str, Any]:
    cfg = load_config(DOWNSTREAM_BASE)
    primary = cfg["model"]["primary"]
    training = cfg["training"]
    loss = cfg.setdefault("loss", {}).setdefault("u_mask_beam_jepa", {})
    cfg["experiment"]["name"] = run_name
    cfg["experiment"]["seed"] = int(seed)
    cfg["data"]["dataset"].update(_scene_dataset_cfg(scenes, seq_len=2, num_pred=1, cache_enabled=False))
    cfg["output"]["dir"] = str(output_dir)
    cfg["output"]["run_name"] = run_name
    cfg["output"]["group_by_scene"] = True
    cfg["model"]["num_pred"] = 1
    cfg["model"]["seq_length"] = 2
    primary["ablation_id"] = run_name
    primary["fusion_type"] = "weighted_sum"
    primary["use_beam_prototype_alignment"] = True
    primary["use_full_to_partial_kd"] = False
    primary["kd_teacher_mode"] = "disabled"
    primary["encoders"]["image"] = _tinyvit_encoder_cfg(tinyvit_encoder, trainable=True)
    primary["encoders"]["lidar"] = _tinyvit_encoder_cfg(tinyvit_encoder, trainable=True)
    primary.setdefault("encoder_checkpoint_paths", {})
    primary["encoder_checkpoint_paths"].update({"image": image_checkpoint, "lidar": lidar_checkpoint})
    training.update(
        {
            "epochs": int(epochs),
            "max_epochs": int(epochs),
            "random_modality_dropout": _randomdrop_subset(),
            "use_beam_prototype_alignment": True,
            "lambda_proto": 0.2,
            "lambda_modality_proto": 0.1,
            "use_full_to_partial_kd": False,
            "kd_teacher_mode": "disabled",
            "save_best_metric": "val_top1",
            "validation": {"interval_epochs": 1},
        }
    )
    if use_jepa:
        primary["use_jepa_loss"] = True
        loss.update(
            {
                "enabled": True,
                "use_jepa_loss": True,
                "use_beam_prototype_alignment": True,
                "use_full_to_partial_kd": False,
                "kd_teacher_mode": "disabled",
                "lambda_teacher": 0.0,
                "lambda_jepa": 0.1,
                "lambda_jepa_global": 0.1,
                "lambda_modality_nll": 0.1,
                "lambda_proto": 0.2,
                "lambda_modality_proto": 0.1,
            }
        )
    else:
        primary["use_jepa_loss"] = False
        loss.update(
            {
                "enabled": False,
                "use_jepa_loss": False,
                "use_beam_prototype_alignment": True,
                "use_full_to_partial_kd": False,
                "kd_teacher_mode": "disabled",
            }
        )
    return cfg


def _tinyvit_encoder_cfg(name: str, *, trainable: bool) -> dict[str, Any]:
    return {
        "type": name,
        "output_dim": 64,
        "pretrained": False,
        "freeze_backbone": not trainable,
        "unfreeze_stages": [],
        "unfreeze_last_n_stages": 0,
        "dropout": 0.1,
    }


def _scene_dataset_cfg(scenes: list[int], *, seq_len: int, num_pred: int, cache_enabled: bool) -> dict[str, Any]:
    return {
        "type": "deepsense6g",
        "scene": int(scenes[0]),
        "scenes": list(scenes),
        "train_scenes": list(scenes),
        "validation_scenes": list(scenes),
        "test_scenes": list(scenes),
        "split_protocol": "stratified_80_10_10",
        "split_strategy": "stratified_by_target_beam_per_scene",
        "split_seed": 42,
        "split_source_splits": ["train", "test"],
        "split_fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "train_csv_name": "train_seqs_RA_GPS_LIDAR.csv",
        "test_csv_name": "test_seqs_RA_GPS_LIDAR.csv",
        "seq_len": int(seq_len),
        "num_pred": int(num_pred),
        "sample_cache": {"enabled": bool(cache_enabled)},
    }


def _randomdrop_subset() -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": "random_nonempty_subset",
        "modalities": MODALITIES,
        "ensure_at_least_one_modality": True,
    }


def _checkpoint_path(output_dir: str, scenes: list[int], run_name: str) -> str:
    return str(Path(output_dir) / _scene_slug(scenes) / run_name / "checkpoints" / "best_top1.pth")


def _scene_slug(scenes: list[int]) -> str:
    ordered = sorted(int(scene) for scene in scenes)
    if len(ordered) == 1:
        return f"scene{ordered[0]}"
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"scenegroup_s{ordered[0]}_s{ordered[-1]}"
    return "scenegroup_" + "_".join(f"s{scene}" for scene in ordered)


def _write_configs(rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    for row in rows:
        path = Path(row["path"])
        if path.exists() and not overwrite:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = copy.deepcopy(row["config"])
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_manifest(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_name",
        "group",
        "stage",
        "config_path",
        "seed",
        "method_tags",
        "expected_epochs",
        "priority",
        "execution_mode",
        "requires",
    ]
    manifest_rows = [{key: row.get(key, "") for key in fieldnames} for row in rows]
    with (out_dir / "experiment_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    (out_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_budget_manifest(
    out_dir: Path,
    *,
    output_dir: str,
    scenes: list[int],
    rows: list[dict[str, Any]],
    pretrain_epochs: int,
    downstream_epochs: int,
) -> None:
    payload = {
        "workflow": "scenes31_34_tinyvit_encoder_ablation",
        "change_id": "add-scenes31-34-tinyvit-encoder-ablation",
        "config_manifest": rel(out_dir / "experiment_manifest.csv"),
        "dataset_family": "DeepSense6G",
        "scenes": list(scenes),
        "reads_real_dataset": True,
        "gpu_plan": "2 parallel pretrain workers, then 2 parallel downstream workers",
        "outputs_root": str(output_dir),
        "checkpoint_plan": "writes pretrain and downstream checkpoints under ignored outputs root",
        "cache_plan": "sample_cache disabled in generated configs; dataloader may read existing source data only",
        "stop_condition": "runner exits after both stages finish or a stage reports failed/missing_checkpoint",
        "artifact_boundary": "outputs/logs/cache/checkpoints are local runtime artifacts and must not be committed",
        "epochs": {"pretrain": int(pretrain_epochs), "downstream": int(downstream_epochs)},
        "runs": [
            {
                "run_name": row["run_name"],
                "stage": row["stage"],
                "config_path": row["config_path"],
                "requires": row.get("requires", ""),
            }
            for row in rows
        ],
    }
    (out_dir / "run_budget_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
