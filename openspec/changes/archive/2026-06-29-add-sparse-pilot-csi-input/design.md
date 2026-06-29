## Overview

新增 `sparse_pilot` 作为 physics-informed MMW baseline 的推荐受限 CSI 输入。它仍从已准备好的 clean CSI tensor 派生，用于离线实验模拟，但语义从“裁剪完整 CSI”改为“只观测 pilot 位置，其余位置为 0，并显式输出 observation mask”。这使训练逻辑能表达 sparse wireless observation + full CSI supervision 的区别。

## Design

### Adapter

`PhysicsAdapterConfig` 增加：

- `pilot_subcarrier_stride`
- `pilot_antenna_stride`
- `pilot_pattern`: `grid` 或 `random`
- `pilot_random_seed`

`csi_input_mode=sparse_pilot` 时：

- `csi_target` 仍是当前完整 clean CSI。
- `csi_input` 与 `csi_target` shape 相同，未观测位置置零。
- `csi_observation_mask` shape 为 `csi_target.shape[:-1]`，标记哪些 subcarrier/antenna pilot 被观测。
- metadata 记录 input source、pattern、stride 和 observed fraction。

### Batch/model

当前模型只消费 `csi_input`，不需要改 forward。`csi_observation_mask` 先作为 batch/metadata 留存，后续若引入 mask-aware encoder 再消费。这样 diff 最小，也不会改变现有训练循环。

### 配置

新增 `configs/fusion/physics_informed_mmw_sparse_pilot_multimodal.yaml`，启用 image + CSI，`csi_input_mode=sparse_pilot`。`partial` 配置保留，但文档和实验矩阵应把 sparse pilot 作为更合理的受限 CSI 主线。

### 非目标

- 不实现 diffusion/flow matching。
- 不实现 mask-aware CSI transformer。
- 不把 sparse pilot 结果声明为真实硬件采集，只声明为低开销 pilot observation proxy。
