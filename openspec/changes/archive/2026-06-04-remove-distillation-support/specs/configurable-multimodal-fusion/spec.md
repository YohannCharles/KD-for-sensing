## MODIFIED Requirements

### Requirement: Fusion canonical 多模态配置矩阵
项目 MUST 为 `image`、`radar`、`gps`、`lidar`、`mmwave` 的所有必要多模态组合提供 canonical fusion supervised 配置矩阵。多模态组合 MUST 覆盖全部 10 个双模态组合、10 个三模态组合、5 个四模态组合和 1 个五模态组合。每个组合 MUST 提供可加载的 strong 和 lightweight canonical 配置路径；这些 canonical 配置 MAY 由 loader 生成，不要求每个路径都有实体 YAML 文件。

#### Scenario: 双模态 fusion 组合完整
- **WHEN** 开发者加载 `configs/fusion/<slug>_<mode>.yaml`
- **THEN** 系统 MUST 为所有合法双模态 slug 提供 `<slug>_strong.yaml` 和 `<slug>_lightweight.yaml`
- **AND** 系统 MUST 不要求提供 `<slug>_logits_kd.yaml` 或 `<slug>_rkd.yaml`

#### Scenario: 五模态 fusion 组合完整
- **WHEN** 开发者加载五模态 fusion canonical 配置
- **THEN** 系统 MUST 提供可加载的 `image_radar_gps_lidar_mmwave_strong.yaml` 和 `image_radar_gps_lidar_mmwave_lightweight.yaml`
- **AND** 系统 MUST 拒绝同 slug 的 KD 配置路径

### Requirement: Fusion canonical 配置语义
每个 canonical fusion 配置 MUST 使用固定模态顺序 `image`、`radar`、`gps`、`lidar`、`mmwave` 生成 slug。推荐/default fusion lightweight 路线 MUST 使用 `cls_token_transformer_fusion` 或当前 active fusion model。canonical 配置语义 MUST 不依赖实体 YAML 文件是否存在，且 MUST 不包含 distillation 或 frozen teacher runtime。

#### Scenario: fusion strong 配置
- **WHEN** 开发者加载 `configs/fusion/<slug>_strong.yaml`
- **THEN** 配置 MUST 设置 `experiment.task: fusion`
- **AND** 配置 MUST 将 `model.primary` 配置为 strong fusion baseline
- **AND** primary model modalities MUST 等于 slug 表示的模态集合

#### Scenario: fusion lightweight 默认配置
- **WHEN** 开发者加载 `configs/fusion/<slug>_lightweight.yaml`
- **THEN** 配置 MUST 设置 `experiment.task: fusion`
- **AND** 配置 MUST 将 `model.primary` 配置为 `cls_token_transformer_fusion` 或当前推荐 lightweight fusion 模型
- **AND** 配置 MUST 不构建 frozen teacher

## REMOVED Requirements

### Requirement: Canonical fusion virtual config 不扩展 legacy KD 模式
**Reason**: legacy KD 模式已从“收窄但可讨论”变为完全删除。
**Migration**: 使用 `<slug>_strong.yaml`、`<slug>_lightweight.yaml`、snapshot 或 active overlay recipe。

#### Scenario: KD fusion virtual config 不再接管路径
- **WHEN** 用户请求不存在实体 YAML 的 fusion `logits_kd` 或 `rkd` 配置
- **THEN** 系统 MUST 拒绝路径
- **AND** 错误信息 MUST 说明 KD support 已删除

