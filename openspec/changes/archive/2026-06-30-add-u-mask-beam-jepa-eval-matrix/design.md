## Overview

本 change 以独立 `kd_sensing.eval` 包承载 U-MaskBeamJEPA 评估矩阵。runner 显式构造 `missing_mask` 并传给模型，避免触发训练 extension 的随机 mask；CLI 只做配置、dataloader、model、checkpoint 和导出粘合。

## Constraints

- 不修改 `src/kd_sensing/models/u_mask_beam_jepa.py`。
- 不修改 U-MaskBeamJEPA training loss、训练 extension 或已有训练行为。
- 不新增根目录脚本、旧入口或兼容聚合层。
- 所有项目 Python 验证命令使用 `conda run -n kd_mm_beam ...`。

## Approach

- `missing_patterns.py` 生成稳定 pattern 名称，并复用 `kd_sensing.data.missing_mask.sample_missing_mask` 做 random eval mask。
- `metrics.py` 只处理 `[B, K]` logits；runner 对 `[B, T, K]` 通过 `prediction_index` 选择一个 prediction slot，默认 `last`。
- `u_mask_beam_jepa_eval_matrix.py` 对每个 fixed pattern 和 random `p_missing` 遍历 dataloader，聚合 loss、Top-K、confidence、global reliability、modality reliability、available-modality reliability 和 ECE。
- runner 在无 `cfg` 时直接调用 `model(batch, missing_mask=...)` 或 `model(missing_mask=..., **batch)`，便于 fake model 测试；CLI 传入 `cfg` 时复用 `run_model_step` 走真实 batch/model 适配。
- `export.py` 使用标准库输出 CSV、JSON 和 Markdown。
- CLI 使用现有 `load_cli_config`、`build_dataloaders`、`build_model`、`build_device` 和 `load_model_state`，保持薄入口。

## Risks

- 真实 checkpoint eval 仍依赖 config 能构建对应 split 和归一化状态；复杂 registry artifact 解析继续由现有 `kd-sensing-evaluate` 承担。
- reliability diagnostics 是评估统计，不表示模型已经经过校准训练。
- 本 change 不定义 corrupted modality protocol；后续如需图像噪声、GPS 偏移等扰动，应另开 change。
