## REMOVED Requirements

### Requirement: Scenario D 图像可观测性等级
**Reason**: D0-D7 benchmark naming 无 current config/runner consumer。
**Migration**: 通用 image observability operator/metadata 保留，不保留 Scenario-D preset。
#### Scenario: D-level 退出
- **WHEN** config 请求 D0-D7
- **THEN** current parser MUST 拒绝

### Requirement: Cx-Dy 二维鲁棒性矩阵
**Reason**: Scenario-CxD matrix 产品面退役。
**Migration**: Current evaluation 使用 U-Mask/missing-stress owners。
#### Scenario: CxD matrix 退出
- **WHEN** current benchmark 被枚举
- **THEN** 5x8 matrix MUST 不被要求

### Requirement: Scenario D 指标和论文产物
**Reason**: 专属 metrics/plots 无 current claim consumer。
**Migration**: 历史产物留 ignored outputs/archive。
#### Scenario: 论文产物退出
- **WHEN** current paper export 运行
- **THEN** 它 MUST 不要求 Scenario-D files

### Requirement: Scenario D 复现与产物边界
**Reason**: Workflow 整体删除。
**Migration**: 通用 runtime artifact boundary 仍适用。
#### Scenario: Manifest 退出
- **WHEN** current workflow 被规划
- **THEN** Scenario-D manifest MUST 不存在

### Requirement: Scenario D phase transition artifact
**Reason**: CxD phase analysis 退役。
**Migration**: 历史结果从 archive查询。
#### Scenario: Phase artifact 退出
- **WHEN** current reports 被枚举
- **THEN** phase files MUST 不被要求

### Requirement: Scenario D dominance evidence status
**Reason**: Dominance diagnostics 没有 current producer。
**Migration**: Current diagnostics 只报告实际 fields。
#### Scenario: Dominance status 退出
- **WHEN** current model 被评估
- **THEN** 它 MUST 不要求 Scenario-D dominance schema

### Requirement: Scenario D crossing boundary 可比较性
**Reason**: ResNet/JEPA crossing comparison 与 query pool均退役。
**Migration**: Current protocols管理自身 comparability。
#### Scenario: Crossing 退出
- **WHEN** current comparisons 被生成
- **THEN** Scenario-D crossing MUST 不存在

### Requirement: Scenario D benchmark suite
**Reason**: Shortcut benchmark suite adapter 退役。
**Migration**: Generic difficulty operator 可直接用于 current evaluation。
#### Scenario: Suite 退出
- **WHEN** manifest 请求 Scenario-D
- **THEN** current runner MUST 拒绝

### Requirement: Scenario D required model groups
**Reason**: 指定模型组包含已退役 JEPA lines。
**Migration**: Current model matrix 由 final C2/Scene31-34 owners定义。
#### Scenario: Model groups 退出
- **WHEN** current benchmark 被验证
- **THEN** 它 MUST 不要求 Scenario-D groups

### Requirement: Scenario D aggregation 和图表
**Reason**: 专属 heatmap/surface/phase plots 产品面退役。
**Migration**: 历史 plots 留 ignored artifacts/archive。
#### Scenario: Aggregation 退出
- **WHEN** current evaluation 完成
- **THEN** 它 MUST 不生成 Scenario-D plot contract
