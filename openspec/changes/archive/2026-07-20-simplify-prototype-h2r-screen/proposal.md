## Why

当前 H2R-JointCE 在 Joint40/60/80 上取得最佳 H2R ADBA，但其 prototype 因果依赖尚未与普通置信度融合分离，且 Router 校准同时使用多项 power/排序/锚定损失和 40 epoch 预算。需要用固定 checkpoint、固定 Joint panel 和 seed1 八卡矩阵判断：prototype-topology 证据是否必要，以及更短、更少损失的校准能否保留收益。

## What Changes

- 为 H2R 增加 `full`、`generic_confidence`、`prototype_topology` 三种显式 evidence profile；三者保持相同输入宽度和 Router 参数量，只屏蔽不同证据组。
- 允许 H2R 在 JointCE 能直接训练帧健康门控时关闭 frame-rank 辅助项，不改变既有配置默认行为。
- 新增固定八卡 seed1 开发 launcher，比较 Full-40、Full-10、Lite-10、Lite+TopologyMonotonic-10，以及 `generic/prototype × JointCE/JointCE+TopologyMonotonic` 四个因果候选。
- 所有候选复用同一 CurrentControl checkpoint、240-entry Joint panel、训练/评估 mask identity 和 batch64；生成配置、manifest、日志与 checkpoint 只写入 ignored `outputs/`。
- 本变更只生成 inner development evidence；不得修改 canonical T2/S1 recipe，不得直接补 seed2--5 或形成正式 TWC claim。

## Capabilities

### New Capabilities

- `prototype-h2r-simplification-screen`: 定义 H2R prototype 因果与训练复杂度八卡筛选矩阵、固定身份和晋级边界。

### Modified Capabilities

- `u-mask-beam-jepa`: 增加 H2R evidence profile 和可由 JointCE 单独训练帧门控的配置语义。
- `training-evaluation-runtime`: 增加该 seed1 筛选的不可变 manifest、GPU 映射、产物和 claim 边界。

## Impact

- 影响 `src/kd_sensing/models/prototype_health_router.py`、`src/kd_sensing/models/u_mask_beam_jepa.py`、动态 Router 配置解析、一个独立 launcher 和聚焦测试。
- 不新增第三方依赖，不修改数据集、历史 checkpoint、canonical 配置、公共 CLI 或正式 claim 文档。
