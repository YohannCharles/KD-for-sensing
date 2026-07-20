## Why

首轮八个动态 Router 候选均能在 Joint40/60/80 上取得正向但不足材料性的 static-prior 增益，其中 H2R 能稳定降权受损时间块，却仍不能在 normalized gain 上超过 frozen Current Router。训练日志同时显示 fused expected utility 已接近 `0.99`，而 quality/monotonic loss 随校准推进反而上升，说明当前连续期望效用已饱和且与最终 argmax 通信指标错位。

## What Changes

- 在既有 same-availability Joint control/corrupt 配对上，为 fused logits 增加互斥、可审计的决策对齐目标：Joint hard-label CE、beam-power soft CE 和 beam-power top-choice margin ranking。
- 保留当前 continuous expected-utility 作为对照，不修改预注册 Gate；所有物理 power 目标继续只用于训练 loss，不进入模型 forward。
- 固定 PATR 与 H2R 两个最小候选架构，形成 `2 architecture × 4 objective` 的八卡 seed1 矩阵；不继续扩展 CoRe/Unified 组合。
- 使用成熟 CurrentControl checkpoint、同一 240-entry Joint training panel、batch64、40 epoch 和相同 optimizer/freeze 边界，完成夜间训练；后续评估仍复用固定 81-condition Joint cache。
- 训练日志必须分别记录主任务 CE、各 Joint 决策目标、active beam-pair 比率及最终 fused utility，避免再次只观察总 loss。

## Capabilities

### New Capabilities

- `dynamic-router-physical-utility-alignment`: 定义 Joint fused-logit 的 hard CE、power soft CE、top-choice power ranking、互斥配置与八卡筛选证据边界。

### Modified Capabilities

- `u-mask-beam-jepa`: 候选 Router 的配对训练可在不泄漏 corruption metadata 的前提下对 Joint fused logits施加声明的决策目标。
- `training-evaluation-runtime`: 夜间筛选必须冻结 source/panel/objective 身份并维持 inner-only claim 边界。

## Impact

- 主要影响 `src/kd_sensing/losses/router_reliability.py`、现有 UMaskBeamJEPA loss 配置解析、一个新的夜间筛选 launcher及聚焦测试。
- 不新增第三方依赖，不修改 canonical T2/S1/baseline recipe，不重训冻结 expert，不更改现有 Gate 或历史输出。
- 新 checkpoint、resolved config、日志、manifest 和后续评估均写入 ignored `outputs/`。
