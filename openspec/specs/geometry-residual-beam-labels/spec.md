# geometry-residual-beam-labels Specification

## Purpose
定义 geometry coarse beam、circular residual beam label 和 clipped residual class 的数据契约，用于把绝对 beam 预测拆解为可解释的几何先验与残差学习目标，并保证 label space 可逆、可诊断且默认不影响 absolute label 路径。
## Requirements
### Requirement: geometry residual label 路线已退役
专门用于把绝对 beam 拆成 geometry coarse beam 与 residual/delta class 的 label-space 路线不再属于当前支持能力。系统 MUST 不再要求 dataset 暴露 `beam_geo`、`beam_residual`、`residual_class`、GPS local delta class 或 geometry residual target provider。

#### Scenario: 默认数据样本不含 geometry residual 契约
- **WHEN** 用户运行当前保留训练或评估配置
- **THEN** dataset sample MUST 不要求 geometry residual 字段
- **AND** 项目 MUST 不保留专门服务 residual/delta 路线的 target provider 文档或测试

### Requirement: Geometry-residual label 字段按需加载
数据加载流程 MUST 在 `label_space.type: geometry_residual` 时按需加载或派生 geometry-residual label 字段。未启用 geometry-residual label_space 时，dataset MUST 不要求 position/geometry 字段，也不得改变现有 absolute beam label sample keys。

#### Scenario: geometry_residual 启用时返回新增标签字段
- **WHEN** 用户运行启用 `label_space.type: geometry_residual` 的配置
- **THEN** dataset 或 target provider MUST 返回 absolute beam label、geometry coarse beam 和 residual label 可用字段
- **AND** batch preparation MUST 保持这些字段 shape 可默认 collate

#### Scenario: 默认 absolute 配置不读取 geometry
- **WHEN** 用户运行现有 absolute beam classifier 配置
- **THEN** dataset MUST 继续按启用模态读取 sensing input 和 absolute beam label
- **AND** 缺少 GPS/pose/relative geometry 不得阻止 dataset 构建
