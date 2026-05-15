## ADDED Requirements

### Requirement: Fusion 多任务配置入口
Fusion 配置 MUST 能声明多任务辅助监督相关选项，包括启用状态、遮挡阈值分位数、位置目标来源、辅助 head 开关和 loss 权重。默认 fusion 配置 MUST 保持 beam-only，recommended 多任务配置 MUST 显式启用五模态和 auxiliary heads。

#### Scenario: 默认 fusion 配置保持 beam-only
- **WHEN** 用户加载现有 canonical fusion 配置
- **THEN** 配置 MUST 不默认启用遮挡或位置辅助任务
- **AND** 模型和 dataset MUST 保持现有 beam-only 行为

#### Scenario: 五模态多任务推荐配置
- **WHEN** 用户加载 recommended 五模态多任务 fusion 配置或 overlay
- **THEN** 配置 MUST 设置 `modalities: ["image", "radar", "gps", "lidar", "mmwave"]`
- **AND** 配置 MUST 启用 `cls_token_transformer_fusion` 的遮挡和位置辅助头
- **AND** 配置 MUST 启用 dataset 的遮挡和位置目标生成

#### Scenario: loss 权重可配置
- **WHEN** 用户在 fusion 配置中设置 beam、遮挡或位置 loss 权重
- **THEN** 训练流程 MUST 使用配置值计算多任务总 loss
- **AND** final config MUST 记录实际使用的权重

### Requirement: Fusion 配置校验多任务依赖
系统 MUST 对多任务 fusion 配置进行显式校验。启用遮挡目标时必须能访问 beam sweep power 文件；启用位置目标时必须声明位置目标来源；启用 auxiliary loss 时模型必须支持对应辅助输出。

#### Scenario: 位置目标缺少来源
- **WHEN** 配置启用位置辅助任务但未声明合法 `position_target_source`
- **THEN** 系统 MUST 拒绝配置
- **AND** 错误信息 MUST 列出支持的 position target source

#### Scenario: 模型不支持辅助输出
- **WHEN** 配置启用遮挡或位置 loss，但 `model.student` 不支持对应 auxiliary head
- **THEN** 训练流程 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出模型输出缺少的辅助字段

#### Scenario: 遮挡目标与数据文件不匹配
- **WHEN** 配置启用遮挡目标但监督 beam 文件不是 64 维 power vector
- **THEN** dataset 构建或首次取样 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出遮挡标签生成依赖 64-beam sweep

