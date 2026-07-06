## Why

TinyViT 强化 image/lidar 后，full、image_only、lidar_only 提升但 radar_only 下降，说明问题更像是训练阶段共享表示和共享预测头被强模态主导，而不是推理阶段简单的 mask 权重归一化失效。需要在不改变默认行为的前提下，新增一组显式 opt-in 的缺失模态鲁棒性机制和本地实验编排，用来验证动态融合、radar 分支保护、hard subset 目标和 checkpoint selection 的真实收益。

## What Changes

- 新增 Pattern-Conditioned Prototype Gate（PCPG）作为 opt-in fusion 选项，优先实现 logits/prototype-score 融合，并记录 gate diagnostics。
- 新增 Radar-Protected Branch-Balanced Training 的 opt-in 训练辅助项，覆盖 unimodal auxiliary CE、radar auxiliary CE，并为 prototype distance loss 保留可扩展接口。
- 新增 Hard-Subset Aware Objective 的静态 loss reweighting，默认只在显式 flag 启用时提高 radar_only、missing_image、miss3 等 hard subset 权重。
- 新增可选 JEPA latent alignment 接入，默认关闭，只服务组合实验。
- 新增 missing-aware checkpoint selection 选项：保留默认 val_acc 行为，额外支持 avg_missing_top1 和 worst_pattern_top1。
- 新增 eval-only oracle gate，用 ground-truth upper bound 选择可用 unimodal 分支并明确标注 oracle。
- 新增 6 组实验矩阵、最多 8 进程/4 GPU/每 GPU 2 进程的本地 launcher，以及结果汇总脚本；输出限定在 `outputs/pcpg_radar_balance_v1/`。
- 新增 focused tests 覆盖 PCPG mask、hard subset weighting、checkpoint selection 和 launcher dry-run。
- 不新增旧入口、不复制训练框架、不提交训练输出、checkpoint、日志或 generated config。

## Capabilities

### New Capabilities
- `pcpg-radar-balance-robustness`: 覆盖 PCPG 动态融合、radar-protected branch-balanced training、hard subset weighting、oracle gate、missing-aware checkpoint selection、6 组实验矩阵、本地 launcher 和汇总产物边界。

### Modified Capabilities
- 无；本 change 以新增 opt-in capability 表达，现有训练、评估、配置和脚本生命周期默认行为保持不变。

## Impact

- 主要影响 `src/kd_sensing/models/` 中的 U-MaskBeamJEPA/缺失模态融合路径、`src/kd_sensing/engine/` 中的训练 extension、checkpoint selection 与 diagnostics 写出，以及 `scripts/` 下的本地实验 launcher/summary helper。
- 配置和 CLI 通过现有 `kd-sensing-train` / `kd-sensing-evaluate` 的 config override 方式启用，不新增 package console script。
- 新增脚本为 local/manual experiment surface，必须支持 dry-run 或只读 summary，并将日志、manifest 和汇总写入 ignored `outputs/pcpg_radar_balance_v1/`。
- 不引入新的运行时依赖；所有项目 Python 验证仍通过 `conda run -n kd_mm_beam ...` 执行。
