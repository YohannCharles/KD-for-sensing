## ADDED Requirements

### Requirement: Geometry-aware transferable knowledge
HiST-Beam MUST 显式建模可迁移知识，包括 coarse angular/beam semantics、angular neighborhood continuity、RSU-CAV relative geometry 和 cross-modal geometric consistency。每项知识 MUST 在配置、模型输出或 loss diagnostics 中有可追踪字段。

#### Scenario: 输出 transferable diagnostics
- **WHEN** geometry-aware HiST-Beam forward 完成
- **THEN** 模型输出 MUST 包含 coarse logits、beam-level logits、shared geometry representation 和 geometry diagnostics
- **AND** diagnostics MUST 至少记录启用的 geometry fields、可用性 mask 和 direct/proxy 标记

#### Scenario: coarse head 绑定 shared geometry
- **WHEN** geometry-aware shared/private 模式启用
- **THEN** coarse head MUST 只读取 shared geometry representation 或其投影
- **AND** private scene representation MUST 不直接作为 coarse head 输入

### Requirement: Scene-private knowledge as explicit refinement
HiST-Beam MUST 将 town/scene layout、RSU pose/local coordinate frame、local scatterer/occluder proxy 和 coarse sector 内 fine beam mapping 作为 scene-private refinement 处理。scene-private 分支 MUST 服务 fine mapping adapter，而不是替代 coarse shared semantics。

#### Scenario: fine head 读取 private refinement
- **WHEN** 模型启用 scene-private branch
- **THEN** fine head MUST 读取 shared geometry representation 与 adapted private representation 的组合
- **AND** adapted private representation MUST 可由 coarse sector embedding 或 coarse context 条件化

#### Scenario: proxy 不伪装成真实标签
- **WHEN** private branch 使用 occluder 或 scatterer 相关输入
- **THEN** 模型 diagnostics MUST 将这些输入标记为 proxy
- **AND** summary MUST 不将 proxy 字段报告为真实 scene semantics 标签

### Requirement: Angular smoothing loss
HiST-Beam MUST 支持 angular smoothing loss，用 beam/codebook 邻接关系构造 soft target，以约束相邻角度 beam 的连续性。该 loss MUST 支持按配置选择 linear ULA 邻接或 circular 邻接。

#### Scenario: 线性 codebook smoothing
- **WHEN** 配置 `angular_smoothing.enabled: true` 且 codebook topology 为 `linear`
- **THEN** 系统 MUST 按 beam index 或 steering angle 的非循环距离构造 soft target
- **AND** beam 0 与最后一个 beam 不得被视为相邻，除非配置显式启用 circular topology

#### Scenario: angular loss diagnostics
- **WHEN** angular smoothing loss 参与训练
- **THEN** loss diagnostics MUST 包含 angular loss 数值、sigma 或温度参数、topology 和有效样本数

### Requirement: Multimodal geometry consistency loss
HiST-Beam MUST 支持 multimodal geometry consistency loss，用于约束 GPS/IMU、CAV/RSU pose、LiDAR、depth、bbox、radar point cloud 和 channel-derived geometry 之间的一致性。该 loss MUST 对缺失模态使用 mask，并 MUST 记录 coverage。

#### Scenario: 可用模态计算 geometry consistency
- **WHEN** batch 同时包含 relative pose 与至少一个可几何对齐的视觉、LiDAR、depth、bbox、radar 或 channel 字段
- **THEN** 系统 MUST 计算配置启用的 geometry consistency 子 loss
- **AND** diagnostics MUST 记录每个子 loss 的 coverage

#### Scenario: 缺失模态跳过子 loss
- **WHEN** 某个 geometry consistency 子 loss 所需模态缺失
- **THEN** 系统 MUST 跳过该子 loss
- **AND** diagnostics MUST 将对应 coverage 记录为 0 或 unavailable reason

### Requirement: Private prototype alignment must be effective
Adapter+prototype 变体 MUST 对齐 coarse sector 条件下的 private/adapter representation，而不是只对 shared representation 做无差别对齐。prototype loss MUST 具有可诊断的 confidence、coverage、used sample count 和非零权重路径；否则该 run MUST 被标记为 prototype no-op。

#### Scenario: private prototype loss 使用 adapter representation
- **WHEN** `v5_adapter_proto` 或等价 prototype 变体执行 target adaptation
- **THEN** prototype consistency MUST 使用 adapted private representation 或配置指定的 private projection
- **AND** prototype target MUST 按 coarse sector 和 confidence threshold 选择
- **AND** shared-only prototype alignment MUST NOT 作为默认实现

#### Scenario: prototype no-op 可诊断
- **WHEN** prototype loss 权重为 0、prototype artifact 缺失、coverage 为 0 或没有样本超过 confidence threshold
- **THEN** adaptation metrics MUST 标记 prototype status 为 `no_op` 或 `unavailable`
- **AND** quick validation conclusion MUST 不把该 run 描述为有效 prototype variant

#### Scenario: v4 与 v5 对比记录 prototype 差异
- **WHEN** 同一 fold、budget、seed 下存在 adapter-only 和 adapter+prototype run
- **THEN** summary MUST 比较两者 accuracy 与 prototype diagnostics
- **AND** 若两者 prediction 完全一致，summary MUST 记录 `prototype_prediction_delta: 0` 或等价诊断

### Requirement: Geometry-aware HiST-Beam 指标
Geometry-aware HiST-Beam evaluation MUST 输出传统 beam 指标以及角度、几何和 prototype 指标。若某项指标缺少必要数据，系统 MUST 标记 unavailable reason，不得伪造数值。

#### Scenario: 输出角度和几何指标
- **WHEN** geometry-aware HiST-Beam evaluation 完成
- **THEN** metrics MUST 包含 Top-1、Top-3、Top-5、coarse accuracy、fine accuracy 和 mean angular error
- **AND** 若启用 geometry loss，metrics MUST 包含 geometry loss coverage 或 unavailable reason

#### Scenario: 输出 prototype 指标
- **WHEN** prototype alignment 启用
- **THEN** metrics MUST 包含 prototype coverage、confidence mean、used sample count 和 prototype loss mean
- **AND** 这些字段 MUST 被 LOSO summary 汇总
