## Why

S1 在现有 Scene31-34 H5/P1 完整验证上优于 S2-S4、AMBER 和 RMBP-MM，但其 `masked_mean` 会丢失观测顺序、最近观测、连续缺失和证据覆盖量，且 Drop60 到 Drop80 出现明显性能拐点。当前证据已经满足重新评估单一轻量 temporal route 的条件，需要在不恢复旧 S1-S4 wrapper suite、不改变 final C2 默认行为的前提下，验证时序聚合、partial 稳定和极端稀疏路由三类明确假设。

## What Changes

- 重新批准一个显式 opt-in 的 S1 temporal-then-modality 行为，但不恢复已退役的 `temporal_router_type=s1_*` 名称；继续拒绝 S2、S3、S4，并复用当前 U-Mask encoder、supervised router、共享 batch/runtime 和 `ModelOutput` 适配路径。
- 增加 mask statistics 和 gap-aware residual temporal pooling 两个独立可消融的聚合选项；默认保持 `masked_mean`，残差门初始化为恒等退化。
- 扩展现有 online-full teacher 为 temporal superset teacher，并增加置信度门控 soft-logit KD 与 circular beam risk monotonic ranking；不新增 distiller registry，不启用强 feature L2。
- 增加显式 opt-in 的 coverage-aware uniform router shrinkage，只在稀疏输入下向可用模态均匀先验收缩，完整输入保持零回退。
- 参数化扩展现有 H5/P1 launcher、固定 mask evaluator 和 summary，使其能运行 S1 baseline、T1/T2/A1/A3 与按证据进入的联合实验；不恢复已删除的 S1-S4 wrapper。
- 首轮筛选固定使用 seed1，并在 GPU0-7 上每卡最多一个训练进程并行；只有通过 Drop0 guardrail 且改善主要选择指标的候选才补齐 seeds 1/2/3。
- 基于本机 8xA40 吞吐基准，为 S1 lightweight profile 使用 batch64、12 个 PyTorch CPU 线程和 persistent DataLoader workers；默认 H5/P1 profile 保持原资源配置。
- 所有训练、checkpoint、日志、评估 cache 和 summary 继续写入 ignored `outputs/`，不纳入源码变更或正式 claim。

## Capabilities

### New Capabilities

- `s1-lightweight-temporal-robustness`: 定义单一 S1 route、gap-aware residual pooling、temporal superset KD、beam monotonic ranking、coverage-aware router shrinkage、消融门禁和实验产物契约。

### Modified Capabilities

- `u-mask-beam-jepa`: 新增与历史 S1 语义等价但使用新 `temporal_pooling` 配置的 opt-in 行为，继续拒绝旧 S1-S4 `temporal_router_type`，并扩展 current U-Mask teacher/loss metadata。
- `temporal-window-missing`: 扩展现有 H5/P1 local/manual workflow 的方法矩阵、GPU0-7 调度、固定 mask 评估与阶段门禁，不新增派生 wrapper。
- `training-evaluation-runtime`: 将现有 same-model online-full stabilization 扩展为 temporal superset consistency、置信度门控 KL 和 circular beam-risk monotonic ranking，并保持 training extension 边界。
- `distillation-free-project-surface`: 消除 current spec 漂移，只为同一 primary model 的在线 stop-gradient superset consistency 划定窄例外；外部/冻结 checkpoint teacher、distiller registry、旧 KD 路径和 `distillation.*` 继续禁止。

## Impact

- 主要影响 `src/kd_sensing/models/u_mask_beam_jepa.py`、`src/kd_sensing/losses/u_mask_beam_jepa.py`、共享 temporal metadata 透传、`scripts/launch_h5_p1_temporal_models_v1.py`、评估/summary 脚本和 focused tests。
- 不新增完整模型注册名、长期 package CLI、外部依赖或独立训练循环；final C2 与普通 U-Mask 配置默认行为不变。
- 实验执行使用 `conda run -n kd_mm_beam ...`；本地 GPU0-7 每卡最多一个训练进程，运行产物只保存在 ignored `outputs/`。
