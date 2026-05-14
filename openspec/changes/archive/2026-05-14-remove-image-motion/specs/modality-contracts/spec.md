## ADDED Requirements

### Requirement: Image 模态仅支持 RGB/ImageNet 输入
系统 MUST 将 image modality 的输入契约固定为 RGB/ImageNet 路径。配置解析、模态契约和模型构建 MUST 拒绝 `motion_mask` profile、motion cache 能力和 motion image encoder。

#### Scenario: 默认 image profile 为 RGB/ImageNet
- **WHEN** 开发者查询 image modality 的输入契约
- **THEN** 系统 MUST 返回 RGB/ImageNet 输入语义
- **AND** 系统 MUST 返回 3 通道、224x224 的默认空间尺寸
- **AND** 系统 MUST 标记该 image 输入不使用 image motion cache

#### Scenario: motion profile 不可解析
- **WHEN** 用户配置 `image_profile: motion_mask`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 包含 `motion_mask` 已删除和可用 image 输入契约

## MODIFIED Requirements

### Requirement: 中心化模态契约
项目 MUST 提供单一来源的模态契约，用于描述所有受支持模态的规范名称、固定顺序、dataset flag、样本字段、fusion 输入字段、默认 dataset/model 字段，以及是否支持 cache 或归一化 artifact。image modality MUST 不暴露 image motion cache、motion profile 或 motion encoder 推荐。

#### Scenario: 枚举受支持模态
- **WHEN** 开发者查询模态契约
- **THEN** 系统 MUST 返回固定顺序的 `image`、`radar`、`gps`、`lidar` 和 `mmwave`
- **AND** 该顺序 MUST 被 canonical config、fusion 模态解析、dataset 构建和诊断配置复用

#### Scenario: 查询 image 模态元数据
- **WHEN** 开发者查询 `image` 模态契约
- **THEN** 系统 MUST 返回 image 对应的样本字段 `image`
- **AND** 系统 MUST 返回 fusion 输入字段 `image_batch`
- **AND** 系统 MUST 返回 RGB/ImageNet 输入契约
- **AND** 系统 MUST 不返回 image motion cache 能力

#### Scenario: 查询 radar 模态元数据
- **WHEN** 开发者查询 `radar` 模态契约
- **THEN** 系统 MUST 返回 radar 对应的样本字段 `radar_ra` 和 `radar_da`
- **AND** 系统 MUST 返回 fusion 输入字段 `radar_batch`
