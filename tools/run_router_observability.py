#!/usr/bin/env python3
"""Run the frozen-U0 Router observability causal screen.

Stage 1 caches the mask-independent frozen representations once per setting.
Stage 2 trains only the router (and, for Q2/Q3, a small quality branch) on those
cached tensors, so every arm shares bit-identical representations.

The screen answers one question -- does the router need pre-projection quality
information that its current scalars do not expose -- and nothing else.  It does
not train encoders and does not answer the end-to-end joint-training question.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from kd_sensing.baselines import router_observability as ro
from kd_sensing.baselines.full_pool_bt_scl import sha256_file, write_json
from kd_sensing.baselines.full_pool_common import atomic_csv
from kd_sensing.baselines.prototype_decision_adapter import (
    _amp,
    _batch_ids,
    _inputs,
    _sequential,
    checkpoint_normalization_overrides,
    load_frozen_u0,
    preflight,
)
from kd_sensing.data.corruption_conditions import apply_batch_conditions, condition_table
from kd_sensing.engine.data_factory import build_dataloaders, shutdown_dataloader_workers
from kd_sensing.models.router_quality_branch import ARM_DESCRIPTIONS, ARMS, uses_quality_branch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/router_observability"
U0_ROOT = ROOT / "outputs/full_pool_capacity/u0_seed1"
U0_CONFIG = U0_ROOT / "final_config.yaml"
U0_CHECKPOINT = U0_ROOT / "checkpoints/last.pth"
U0_SHA256 = "ed909406a37ec4ccd2b08bd1fb65ab66fc437cec226a526fdaf7ada1407ba8cf"
# float32 replay against a float32 live forward is bit-exact in practice; this is
# a guard against algebraic drift, not a precision allowance.
EQUIVALENCE_TOLERANCE = 1e-4


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    # Union of keys in first-seen order: aggregate tables mix arms that carry
    # different columns (only q2/q3 report an ablation delta), and taking the
    # first row's keys would silently drop the rest.
    fields: dict[str, None] = {}
    for row in rows:
        fields.update(dict.fromkeys(row))
    atomic_csv(path, rows, list(fields))


def _selection(raw: str | None, allowed: Sequence[str], label: str) -> list[str]:
    """Parse a comma-separated shard selector, failing closed on anything unknown.

    A typo here would quietly shrink the screen -- the shard would report success
    while its arms were never trained -- so an unrecognised entry is an error
    rather than a silently dropped item.
    """
    if raw is None:
        return list(allowed)
    chosen = [token.strip() for token in raw.split(",") if token.strip()]
    unknown = [token for token in chosen if token not in allowed]
    if unknown or not chosen:
        raise SystemExit(f"Unknown {label} selection {unknown or raw!r}; allowed: {list(allowed)}")
    return [value for value in allowed if value in chosen]


def _cache_path(root: Path, setting: str, split: str) -> Path:
    return root / f"cache/setting_{setting}/{split}.npz"


# --------------------------------------------------------------------------
# stage 1: frozen representation cache
# --------------------------------------------------------------------------


def build_cache(root: Path, setting: str, *, force: bool = False, max_batches: int | None = None) -> dict[str, Any]:
    if setting not in ro.SETTINGS:
        raise ValueError(f"Unknown setting: {setting!r}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol_dir = root / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    cfg, audit = preflight(U0_CONFIG, U0_CHECKPOINT, U0_SHA256, migration_root=protocol_dir)
    model, structure = load_frozen_u0(cfg, U0_CHECKPOINT, device)
    loaders = build_dataloaders(cfg, normalization_overrides=checkpoint_normalization_overrides(U0_CHECKPOINT))
    manifest: dict[str, Any] = {
        "setting": setting,
        "description": ro.SETTING_DESCRIPTIONS[setting],
        "u0_checkpoint": str(U0_CHECKPOINT),
        "u0_sha256": U0_SHA256,
        "preflight": audit,
        "u0_structure": structure,
        "preprojection_dims": ro.preprojection_dims(model),
        "hook_targets": dict(ro.HOOK_TARGETS),
        "latent_targets": dict(ro.LATENT_TARGETS),
        # float32 end to end; the published A0 row was measured under bfloat16
        # autocast, so absolute Top-1 is not directly comparable to it.
        "cache_dtype": str(ro.CACHE_DTYPE).removeprefix("torch."),
        "autocast": False,
        "comparable_to_published_bfloat16_rows": False,
        "encoders_trained": False,
        "outer_test_accessed": False,
        "splits": {},
    }
    if setting == "C":
        manifest["condition_table"] = condition_table()
        manifest["condition_draw_seed"] = ro.CONDITION_DRAW_SEED
    try:
        for split in ("train", "validation"):
            destination = _cache_path(root, setting, split)
            if destination.is_file() and not force:
                raise FileExistsError(f"Representation cache already exists (fail closed): {destination}")
            payload = _encode_split(model, loaders[split], cfg, device, setting, max_batches=max_batches)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp.npz")
            np.savez(temporary, **payload)
            temporary.replace(destination)
            manifest["splits"][split] = {
                "path": str(destination),
                "sha256": sha256_file(destination),
                "sample_count": int(len(payload["sample_id"])),
                "latent_shape": list(payload["latent_sequence"].shape),
            }
            print(json.dumps({"event": "router_cache", "setting": setting, "split": split, **manifest["splits"][split]}), flush=True)
        equivalence = verify_equivalence(model, loaders["validation"], cfg, device, root, setting)
        manifest["equivalence"] = equivalence
        if not equivalence["passed"]:
            raise ValueError(f"Cache replay does not reproduce the frozen U0 forward: {equivalence}")
    finally:
        for loader in loaders.values():
            shutdown_dataloader_workers(loader)
    manifest["created_at"] = now()
    write_json(root / f"cache/setting_{setting}/manifest.json", manifest)
    return manifest


def _encode_split(
    model: Any,
    loader: Any,
    cfg: Mapping[str, Any],
    device: torch.device,
    setting: str,
    *,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    sequential = _sequential(loader)
    ids: list[str] = []
    domains: list[str] = []
    weathers: list[str] = []
    conditions: list[str] = []
    labels: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    forced: list[np.ndarray] = []
    preprojection: dict[str, list[np.ndarray]] = {name: [] for name in ro.MODALITY_ORDER}
    with ro.EncoderCapture(model) as capture, torch.no_grad():
        for batch_index, batch in enumerate(sequential):
            batch_ids = _batch_ids(batch)
            inputs = _inputs(batch, cfg, device)
            ro.assert_dense_temporal_inputs(inputs)
            batch_forced = torch.zeros((len(batch_ids), len(ro.MODALITY_ORDER)), dtype=torch.bool, device=device)
            batch_conditions = ["clean"] * len(batch_ids)
            if setting == "C":
                batch_conditions = ro.draw_conditions(batch_ids)
                inputs, batch_forced = apply_batch_conditions(inputs, batch_conditions, seed=ro.CONDITION_DRAW_SEED)
            # No autocast: see router_observability.CACHE_DTYPE for why float32 is
            # the reference here rather than the usual bfloat16 evaluation path.
            model(**inputs)
            steps = int(cfg["model"]["seq_length"])
            latent, preproj = capture.collect(len(batch_ids), steps)
            latents.append(ro.pack_cache_array(latent))
            for name in ro.MODALITY_ORDER:
                preprojection[name].append(ro.pack_cache_array(preproj[name]))
            labels.append(torch.as_tensor(batch["target_beam"]).reshape(-1).cpu().numpy().astype(np.int64))
            forced.append(batch_forced.cpu().numpy())
            metadata = batch["metadata"]
            weather = [str(value) for value in metadata["condition"]]
            scenario = [str(value) for value in metadata["scenario"]]
            ids.extend(batch_ids)
            weathers.extend(weather)
            domains.extend(f"{a}/{b}" for a, b in zip(weather, scenario))
            conditions.extend(batch_conditions)
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    shutdown_dataloader_workers(sequential)
    payload = {
        "sample_id": np.asarray(ids, dtype=str),
        "label": np.concatenate(labels),
        "domain": np.asarray(domains, dtype=str),
        "weather": np.asarray(weathers, dtype=str),
        "condition": np.asarray(conditions, dtype=str),
        "latent_sequence": np.concatenate(latents),
        "forced_missing": np.concatenate(forced),
    }
    for name in ro.MODALITY_ORDER:
        payload[f"preprojection_{name}"] = np.concatenate(preprojection[name])
    if len(payload["sample_id"]) != len(set(ids)):
        raise ValueError("Router observability cache contains duplicate sample identities.")
    return payload


@torch.no_grad()
def verify_equivalence(
    model: Any,
    loader: Any,
    cfg: Mapping[str, Any],
    device: torch.device,
    root: Path,
    setting: str,
    *,
    batches: int = 3,
) -> dict[str, Any]:
    """Replayed fusion must reproduce a live frozen-U0 forward for every mask.

    Both sides run in float32 and the cached side is quantised exactly as it will
    be stored, so this gate covers the storage precision as well as the replay
    algebra.  Comparing against a bfloat16 live forward instead would measure the
    gap between two different roundings, neither of which is the reference.
    """
    head = ro.FrozenU0Head(model)
    sequential = _sequential(loader)
    worst = 0.0
    compared = 0
    patterns = [pattern for _, pattern in ro.MASK_PATTERNS]
    with ro.EncoderCapture(model) as capture:
        for batch_index, batch in enumerate(sequential):
            batch_ids = _batch_ids(batch)
            inputs = _inputs(batch, cfg, device)
            ro.assert_dense_temporal_inputs(inputs)
            model(**inputs)
            latent, preproj = capture.collect(len(batch_ids), int(cfg["model"]["seq_length"]))
            latent = ro.quantize_for_cache(latent)
            preproj = {name: ro.quantize_for_cache(value) for name, value in preproj.items()}
            for pattern in patterns:
                mask = ro.expand_mask(pattern, len(batch_ids), device)
                live = model(**inputs, missing_mask=mask)
                replayed = head(latent, preproj, mask)
                reference = head.reference_logits(replayed)
                difference = (reference.float() - live["logits"][:, 0, :].float()).abs().max()
                worst = max(worst, float(difference))
                compared += len(batch_ids)
            if batch_index + 1 >= batches:
                break
    shutdown_dataloader_workers(sequential)
    return {
        "max_absolute_difference": worst,
        "tolerance": EQUIVALENCE_TOLERANCE,
        "compared_samples": compared,
        "masks": len(patterns),
        "passed": bool(worst <= EQUIVALENCE_TOLERANCE),
        "setting": setting,
    }


# --------------------------------------------------------------------------
# stage 2: arms
# --------------------------------------------------------------------------


class Workspace:
    """Load the frozen U0 and each setting's cache once, then reuse across arms."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg, self.audit = preflight(U0_CONFIG, U0_CHECKPOINT, U0_SHA256, migration_root=root / "protocol")
        model, self.structure = load_frozen_u0(self.cfg, U0_CHECKPOINT, self.device)
        self.head = ro.FrozenU0Head(model)
        self._caches: dict[tuple[str, str], ro.RepresentationCache] = {}

    def cache(self, setting: str, split: str) -> ro.RepresentationCache:
        key = (setting, split)
        if key not in self._caches:
            self._caches[key] = ro.load_cache(_cache_path(self.root, setting, split), self.device)
        return self._caches[key]


def run_arm(
    root: Path,
    run: ro.ArmRun,
    *,
    force: bool = False,
    epochs: int = ro.EPOCHS,
    workspace: Workspace | None = None,
) -> dict[str, Any]:
    run_dir = root / run.directory
    if (run_dir / "metrics.json").is_file() and not force:
        raise FileExistsError(f"Router arm already has results (fail closed): {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = workspace or Workspace(root)
    head = workspace.head
    train_cache = workspace.cache(run.setting, "train")
    validation_cache = workspace.cache(run.setting, "validation")

    started = time.monotonic()
    router = ro.build_model(run.arm, head, train_cache, seed=run.seed)
    history = ro.train_arm(router, head, train_cache, seed=run.seed, epochs=epochs)
    ro.fit_quality_mean(router, head, train_cache)
    summary = ro.evaluate_arm(router, head, validation_cache, seed=run.seed)
    payload: dict[str, Any] = {
        **asdict(run),
        "arm_description": ARM_DESCRIPTIONS[run.arm],
        "trainable_parameters": int(router.parameter_count()),
        "router_feature_count": int(router.feature_count),
        "epochs": epochs,
        "final_train_loss": float(history[-1]["loss"]),
        "wall_seconds": time.monotonic() - started,
        "encoders_trained": False,
        "outer_test_accessed": False,
        **{key: value for key, value in summary.items() if key != "per_mask"},
    }
    if uses_quality_branch(run.arm):
        ablated = ro.evaluate_arm(router, head, validation_cache, seed=run.seed, ablate_quality=True)
        payload["ablation"] = {key: value for key, value in ablated.items() if key != "per_mask"}
    torch.save(
        {"state_dict": router.state_dict(), "arm": run.arm, "setting": run.setting, "seed": run.seed},
        run_dir / "router.pt",
    )
    write_json(run_dir / "metrics.json", {**payload, "per_mask": summary["per_mask"]})
    _atomic_csv(run_dir / "training_curve.csv", history)
    _atomic_csv(
        run_dir / "per_mask_metrics.csv",
        [{"mask": key, **values} for key, values in summary["per_mask"].items()],
    )
    write_json(run_dir / "status.json", {"status": "passed", **asdict(run), "created_at": now()})
    print(json.dumps({"event": "router_arm", **{k: payload[k] for k in ("setting", "arm", "seed", "full_top1", "all14_top1")}}), flush=True)
    return payload


def reference_row(root: Path, setting: str, *, workspace: Workspace | None = None) -> dict[str, Any]:
    """Frozen U0's own router, evaluated on the same cache; a sanity reference, not a gate."""
    workspace = workspace or Workspace(root)
    summary = ro.evaluate_arm(None, workspace.head, workspace.cache(setting, "validation"), seed=0)
    return {
        "setting": setting,
        "arm": "u0_frozen_router",
        "seed": 0,
        "arm_description": "frozen U0 router, not retrained (reference only)",
        "trainable_parameters": 0,
        **{key: value for key, value in summary.items() if key != "per_mask"},
    }


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def aggregate(root: Path, *, force: bool = False, workspace: Workspace | None = None) -> dict[str, Any]:
    target = root / "combined_metrics.csv"
    if target.is_file() and not force:
        raise FileExistsError(f"Aggregate already exists (fail closed): {target}")
    rows: list[dict[str, Any]] = []
    ablations: list[dict[str, Any]] = []
    settings = [setting for setting in ro.SETTINGS if (root / f"cache/setting_{setting}").is_dir()]
    workspace = workspace or Workspace(root)
    for setting in settings:
        rows.append(reference_row(root, setting, workspace=workspace))
        for arm in ARMS:
            for seed in ro.ROUTER_SEEDS:
                path = root / ro.ArmRun(setting, arm, seed).directory / "metrics.json"
                if not path.is_file():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows.append({key: value for key, value in payload.items() if key not in {"per_mask", "ablation"}})
                if "ablation" in payload:
                    ablations.append(
                        {
                            "setting": setting,
                            "arm": arm,
                            "seed": seed,
                            **{f"trained_{k}": payload[k] for k in ro.PRIMARY_METRICS},
                            **{f"ablated_{k}": payload["ablation"][k] for k in ro.PRIMARY_METRICS},
                            **{
                                f"delta_{k}": float(payload[k]) - float(payload["ablation"][k])
                                for k in ro.PRIMARY_METRICS
                            },
                        }
                    )
    if not rows:
        raise ValueError("No completed router arms found; run the arms first.")
    _atomic_csv(target, [{k: v for k, v in row.items() if not isinstance(v, dict)} for row in rows])
    if ablations:
        _atomic_csv(root / "inference_ablation.csv", ablations)

    gates: list[dict[str, Any]] = []
    verdicts: dict[str, bool] = {}
    for setting in settings:
        summaries = {
            arm: [row for row in rows if row.get("setting") == setting and row.get("arm") == arm]
            for arm in ARMS
        }
        if any(len(summaries[arm]) != len(ro.ROUTER_SEEDS) for arm in ("q1", "q2", "q3")):
            continue
        setting_gates = ro.evaluate_gates(summaries)
        for gate in setting_gates:
            gates.append({"setting": setting, **gate})
        verdicts[setting] = ro.direction_survives(setting_gates)
    if gates:
        _atomic_csv(root / "success_gates.csv", gates)
    write_json(
        root / "aggregate_status.json",
        {
            "settings": settings,
            "direction_survives": verdicts,
            "recommend_joint_training_proposal": bool(verdicts and any(verdicts.values())),
            "backbone_seeds": 1,
            "router_seeds": list(ro.ROUTER_SEEDS),
            "encoders_trained": False,
            "outer_test_accessed": False,
            "created_at": now(),
        },
    )
    write_report(root, rows, gates, verdicts, ablations)
    return {"rows": len(rows), "gates": len(gates), "verdicts": verdicts}


def _span(rows: Sequence[Mapping[str, Any]], setting: str, arm: str, field: str) -> tuple[float, float] | None:
    values = [float(row[field]) for row in rows if row["setting"] == setting and row["arm"] == arm and row.get(field) not in (None, "")]
    return (min(values), max(values)) if values else None


def _failure_anatomy(
    rows: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    ablations: Sequence[Mapping[str, Any]],
    verdicts: Mapping[str, bool],
) -> list[str]:
    """Say how the screen failed, not just that it failed.

    A bare pass/fail invites the wrong follow-up.  BTMA failed because it could
    not beat a zero-cost control at all; if this route failed differently, the
    difference belongs in the archive next to the verdict -- and so does the
    reason the difference is not grounds for reopening it under this proposal.
    """
    if not verdicts:
        return []
    lines = ["", "## 失败结构", ""]

    lines += [
        "本轮全程 float32，既有 A0 行是 bfloat16 autocast 下测得，因此绝对值不与已发表行直接可比；",
        "所有比较都在本表内部完成，冻结 U0 参照行也经同一条 float32 路径重算。",
        "",
    ]

    for setting in verdicts:
        reference = _span(rows, setting, "u0_frozen_router", "full_top1")
        q0 = _span(rows, setting, "q0", "full_top1")
        if reference and q0:
            lines.append(
                f"- 设定 {setting} 重训对照：冻结 U0 参照 Full Top-1 {reference[0]:.4f}，"
                f"仅重训 router 的 q0 为 [{q0[0]:.4f}, {q0[1]:.4f}]，即 **单靠重训 router 就有约 "
                f"{(sum(q0) / 2 - reference[0]) * 100:+.1f} pp**。任何增益都必须先扣掉这一项，"
                "不能记在 quality 分支名下。"
            )

    for setting in verdicts:
        top1 = [g for g in gates if g["setting"] == setting and "beats" in str(g["gate"])]
        regression = [g for g in gates if g["setting"] == setting and "no_regression" in str(g["gate"])]
        passed_top1 = [g for g in top1 if bool(g["passed"])]
        failed_regression = [g for g in regression if not bool(g["passed"])]
        if passed_top1 and failed_regression:
            names = "、".join(str(g["gate"]).removeprefix("q2_no_regression_") for g in failed_regression)
            lines.append(
                f"- 设定 {setting} 的失败形态是**取舍，不是无效**：{len(passed_top1)}/{len(top1)} 个区分门槛通过，"
                f"但 {names} 反向。q2 把精确波束选得更准，邻域却更差 —— "
                "Top-1 上去而 within-3/MAE 下来，说明它牺牲了错判时的角度邻近性。"
                "在波束预测里 within-3 与 MAE 直接对应吞吐损失，因此这不是可以忽略的副作用。"
            )

    contributions = [
        (row["setting"], float(row["delta_full_top1"]))
        for row in ablations
        if row["arm"] == "q2"
    ]
    nulls = [float(row["delta_full_top1"]) for row in ablations if row["arm"] == "q3"]
    if contributions and nulls:
        by_setting = {
            setting: [delta for key, delta in contributions if key == setting]
            for setting in dict.fromkeys(key for key, _ in contributions)
        }
        summary = "，".join(
            f"设定 {setting} 下降 {min(deltas) * 100:.2f}~{max(deltas) * 100:.2f} pp"
            for setting, deltas in by_setting.items()
        )
        null_span = f"{min(nulls) * 100:+.2f}~{max(nulls) * 100:+.2f} pp"
        lines.append(
            f"- 机制在推理期确实起作用：冻结权重后把 quality embedding 换成训练集均值，Full Top-1 {summary}；"
            f"跨样本置换的 q3 对照做同一消融只变动 {null_span}，即容量本身不解释这一增益。"
            "这与 BTMA 的失败方式不同 —— BTMA 连零成本对照都没打过。"
        )

    lines += [
        "",
        "**但这不构成放宽门槛的理由。** 门槛是跑之前定死的，取舍属于预注册里明确列为不可接受的一类结果；"
        "本轮就此判死，不调参、不加 seed、不访问 outer test。",
        "若日后要再碰这条线，必须是一份**新的**预注册提案，并把 within-3/MAE 取舍作为首要风险写在前面、",
        "预先给出可接受边界，而不是事后把它解释掉。",
    ]
    return lines


def write_report(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    verdicts: Mapping[str, bool],
    ablations: Sequence[Mapping[str, Any]],
) -> Path:
    lines = [
        "# Router 可观测性因果筛选",
        "",
        "> 冻结 Full-pool U0，仅重训 router；encoder 未参与训练。",
        "> **骨干仍是单 seed**；router 为 3 seed。开发集，未访问 outer test。",
        "> 本轮只回答 routing 输入问题，不回答端到端联合训练问题。",
        "",
        "可辩护的主张是 **router 输入设计不足**，不是 prototype 破坏了质量信息 ——",
        "后者已在诊断中检验且聚合层面不支持。",
        "",
        "## 路线",
        "",
        "| arm | router 输入 |",
        "|---|---|",
        *[f"| {arm} | {ARM_DESCRIPTIONS[arm]} |" for arm in ARMS],
        "",
        "## 结果",
        "",
        "| 设定 | arm | seed | 可训练参数 | Full Top-1 | All-14 Top-1 | All-14 Within-3 | All-14 MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('setting')} | {row.get('arm')} | {row.get('seed')} | {row.get('trainable_parameters', 0)} | "
            f"{float(row['full_top1']):.4f} | {float(row['all14_top1']):.4f} | "
            f"{float(row['all14_within3']):.4f} | {float(row['all14_mae']):.3f} |"
        )

    lines += ["", "## 预注册门槛", "", "| 设定 | gate | 处理组 | 对照 | 通过 |", "|---|---|---|---|:---:|"]
    for gate in gates:
        lines.append(
            f"| {gate['setting']} | {gate['gate']} | "
            f"{float(gate['treatment_mean']):.4f} [{float(gate['treatment_min']):.4f}, {float(gate['treatment_max']):.4f}] | "
            f"{float(gate['control_mean']):.4f} [{float(gate['control_min']):.4f}, {float(gate['control_max']):.4f}] | "
            f"{'是' if bool(gate['passed']) else '否'} |"
        )

    if ablations:
        lines += [
            "",
            "## 冻结权重推理期消融",
            "",
            "把每模态 quality embedding 换成训练集均值嵌入，不更新任何参数。",
            "",
            "| 设定 | arm | seed | Full Top-1（训练） | Full Top-1（消融） | 差值 |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in ablations:
            lines.append(
                f"| {row['setting']} | {row['arm']} | {row['seed']} | "
                f"{float(row['trained_full_top1']):.4f} | {float(row['ablated_full_top1']):.4f} | "
                f"{float(row['delta_full_top1']):+.4f} |"
            )

    lines += ["", "## 结论", ""]
    for setting, survives in verdicts.items():
        description = ro.SETTING_DESCRIPTIONS[setting]
        if survives:
            lines.append(f"- 设定 {setting}（{description}）：**全部预注册门槛通过**，可另行预注册端到端联合训练提案。")
        else:
            failed = [gate["gate"] for gate in gates if gate["setting"] == setting and not bool(gate["passed"])]
            lines.append(f"- 设定 {setting}（{description}）：**未通过**，失败门槛 {failed}。按预注册规则判死，不调参、不加 seed、不访问 outer test。")
    if not verdicts:
        lines.append("- 尚无完整的 Q1/Q2/Q3 三 seed 结果，未做门槛判定。")
    lines += _failure_anatomy(rows, gates, ablations, verdicts)
    lines += ["", f"生成时间：{now()}", ""]
    path = root / "router_observability_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--setting", choices=ro.SETTINGS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument(
        "--arms",
        help="Comma-separated subset for --run-all, so the 24 arms can be sharded across GPUs. "
        "Every arm is an independent training run on the same frozen cache, so a shard is "
        "numerically identical to the same arm inside a single sequential pass.",
    )
    parser.add_argument("--seeds", help="Comma-separated router-seed subset for --run-all.")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=ro.EPOCHS)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args(argv)

    root = args.output_root / "smoke_tests" if args.smoke else args.output_root
    root.mkdir(parents=True, exist_ok=True)
    force = args.force or args.smoke
    workspace: Workspace | None = None
    try:
        if args.build_cache:
            settings = [args.setting] if args.setting else list(ro.SETTINGS)
            for setting in settings:
                build_cache(root, setting, force=force, max_batches=args.max_batches or (2 if args.smoke else None))
        if args.arm:
            if not args.setting or args.seed is None:
                parser.error("--arm requires --setting and --seed.")
            run_arm(root, ro.ArmRun(args.setting, args.arm, int(args.seed)), force=force, epochs=args.epochs)
        if args.run_all:
            settings = [args.setting] if args.setting else list(ro.SETTINGS)
            arms = _selection(args.arms, ARMS, "arm")
            seeds = [int(value) for value in _selection(args.seeds, [str(seed) for seed in ro.ROUTER_SEEDS], "seed")]
            workspace = workspace or Workspace(root)
            for run in ro.all_runs(settings=settings, arms=arms, seeds=seeds):
                run_arm(root, run, force=force, epochs=args.epochs, workspace=workspace)
        if args.aggregate:
            workspace = workspace or Workspace(root)
            print(json.dumps({"event": "router_aggregate", **aggregate(root, force=force, workspace=workspace)}), flush=True)
        if not any((args.build_cache, args.arm, args.run_all, args.aggregate)):
            parser.error("Select at least one of --build-cache/--arm/--run-all/--aggregate.")
    except Exception as exc:
        write_json(root / "status.json", {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(), "outer_test_accessed": False})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
