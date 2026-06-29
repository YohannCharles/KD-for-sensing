## Why

现有 physics-informed MMW baseline 的 `partial` CSI 输入只是从当前完整 `csi_target` 裁剪固定左上角，适合作为 debug/ablation，但不足以支撑“低开销稀疏无线观测 + 多模态感知 -> 路径推断 -> 信道重构 -> 波束选择”的主线论证。需要新增更接近实际 pilot acquisition 的 `sparse_pilot` CSI 输入契约，并把当前完整 CSI 继续限制为训练监督。

## What Changes

- 新增 `data.csi_input_mode=sparse_pilot`，从当前 clean CSI target 生成带观测 mask 的稀疏 pilot 观测。
- 支持 `pilot_subcarrier_stride`、`pilot_antenna_stride`、`pilot_pattern` 和 `pilot_random_seed`，默认使用结构化 comb/grid 采样。
- adapter 输出 `csi_input`、`csi_observation_mask` 和 metadata，模型仍只消费 `csi_input`，loss 继续使用完整 `csi_target` 做 reconstruction supervision。
- 保留 `partial` 作为历史 ablation/debug，不再作为推荐主线配置。
- 新增 `physics_informed_mmw_sparse_pilot_multimodal.yaml` 配置和 focused tests。

## Capabilities

### New Capabilities
- 无。

### Modified Capabilities
- `physics-informed-mmw-beam-baseline`: 增加 sparse pilot CSI 输入模式、mask metadata 和主线配置边界。
- `csi-channel-data`: 扩展受限 CSI 输入契约，允许带 `csi_observation_mask` 的 sparse pilot observation 作为模型输入。
- `experiment-workflow`: 增加 sparse-pilot physics-informed MMW 配置入口，并保留 oracle/partial 的结论边界。

## Impact

- 影响 `src/kd_sensing/data/datasets/mmw_physics_adapter.py`、MMW physics 配置、focused tests 和少量文档/规格。
- 不新增训练入口、根目录脚本或新依赖。
- 不改变完整 CSI 作为 `csi_target` 的泄漏边界；`oracle_full` 仍只能作为 upper-bound。
