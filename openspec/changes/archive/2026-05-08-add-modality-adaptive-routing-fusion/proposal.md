## Why

当前 Scene32 all-modal fusion 的强 baseline 是 teacher encoder 初始化后的 `teacher_init_no_prior`，而固定 prior 或 prior residual gate 主要改善解释性，clean all-modal Top-1 未体现稳定收益；Stage 3 解冻强模态 encoder 还出现过下降。需要从“标量 gate 调权”升级为“样本级、horizon 级动态路由”，在不写死 GPS/mmWave 强、Image/LiDAR/Radar 弱的前提下，同时提升 clean all-modal 表现和模态缺失/组合鲁棒性。

## What Changes

- 新增 MARF（Modality-Adaptive Routing Fusion）fusion student：复用现有五模态 encoder、teacher registry、`force_modality_mask` 和 dict diagnostics 输出契约，增加 horizon-wise anchor router、anchor cross-attention fusion、conditional residual adapter 和 beam classifier。
- 将 teacher prior 从固定规则改为 router 的弱 bias：prior 来自现有 teacher registry，可关闭，可缩放，不决定某个模态必须是 anchor 或 residual。
- 新增 subset-aware training：每个 batch 先做 all-modal forward，再按 prior 自动采样 `top_prior`、`random_with_top_prior` 等子集，用 subset CE 和 all-modal 内部 KD 训练模态缺失鲁棒性。
- 新增 MARF 专用 loss 与 diagnostics：支持 residual norm、anchor prior regularization、可选 anchor entropy，并记录每模态、每 horizon 的 anchor/residual 权重。
- 修正并扩展 subset validation：`all` 子集必须与官方 validation 使用同一路径并保持 Top-1 一致；`strong_only/weak_only` 不再写死 GPS/mmWave，而按 teacher prior 自动解析。
- 新增配置和 ablation：主配置、subset training 配置、no residual、no prior bias、no subset training 等对照，保留 `teacher_init_no_prior` 作为 clean all-modal 强 baseline。
- 新增调试/评估脚本：评估一致性检查、模态 subset 评估、shuffle/zero 扰动评估，用于判断 router 是否真的依赖有效模态。

## Capabilities

### New Capabilities
- `modality-adaptive-routing-fusion`: 覆盖 MARF 模型构建、router/anchor/residual 输出契约、teacher prior bias、subset-aware training、MARF diagnostics、subset/perturbation evaluation 和 ablation 配置。

### Modified Capabilities
- `teacher-prior-gated-craf`: teacher registry、encoder 初始化/冻结和 prior 应用能力扩展为 CRAF 与 MARF 共享；MARF 仍不得改变既有 CRAF 默认行为。
- `experiment-workflow`: modality subset validation 从固定强弱模态集合扩展为 prior-driven 集合，并要求 `all` subset 与官方 validation 一致。

## Impact

- 主要代码：`src/kd_sensing/models/fusion/`、`src/kd_sensing/engine/trainer.py`、`src/kd_sensing/engine/validator.py`、`src/kd_sensing/engine/teacher_loader.py`、`src/kd_sensing/distillation/`、`src/kd_sensing/engine/batch.py`。
- 配置：新增 `configs/fusion/scene32_marf*.yaml` 系列，沿用现有 `experiment.task: fusion`、`model.student`、`teacher.registry_path`、`evaluation.modality_subsets` 结构。
- 测试：新增/扩展 MARF forward shape、mask、anchor softmax、prior bias、subset training loss、subset all consistency、配置加载和 synthetic train/eval smoke tests。
- 非目标：不全量解冻 teacher encoders，不默认使用 focal gamma=2，不把 GPS/mmWave 写死为强模态，也不移除现有 CRAF 与 teacher-prior baseline。
