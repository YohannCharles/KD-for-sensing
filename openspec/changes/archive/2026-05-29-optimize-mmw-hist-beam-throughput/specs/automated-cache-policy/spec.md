## ADDED Requirements

### Requirement: MMW RGB/ImageNet image 派生缓存
系统 MUST 支持受控的 RGB/ImageNet image 派生缓存，用于 MMW/DeepSense6G image modality 的 processed image 输入。该 cache MUST 只表示当前 RGB/ImageNet image profile 的模型输入，不得复用或恢复已删除的 image motion cache。

#### Scenario: auto policy 生成 image-derived cache
- **WHEN** 用户启用 image modality 且设置 `data.cache.image.policy: auto`
- **THEN** dataset MUST 在 cache miss 时在线读取原始 image 并生成 RGB/ImageNet processed cache
- **AND** 后续访问同一 image、image size 和 transform version MUST 能复用该 cache
- **AND** 返回 image tensor 的 shape、dtype 和数值语义 MUST 与未启用 cache 时一致

#### Scenario: read-only policy 不写入 image cache
- **WHEN** 用户设置 `data.cache.image.policy: read_only`
- **THEN** dataset MUST 读取已有且 fingerprint 匹配的 image-derived cache
- **AND** cache miss 或 fingerprint 不匹配时 MUST 在线计算当前样本
- **AND** 系统 MUST 不写入新的 image-derived cache 文件

#### Scenario: image motion cache 仍被拒绝
- **WHEN** 用户配置 `image_motion_*` 或旧 image motion cache 字段
- **THEN** 配置解析 MUST 拒绝该配置
- **AND** 错误信息 MUST 说明应使用 RGB/ImageNet image-derived cache 或关闭 image cache

### Requirement: image-derived cache 可追踪
训练、评估、profile 和预热入口 MUST 记录 image-derived cache 的生效策略、cache 目录、transform version、coverage、命中/缺失统计和生成行为。未启用 image modality 时不得访问 image-derived cache。

#### Scenario: 运行产物记录 image cache 状态
- **WHEN** 一次训练或 profile 构建启用 image modality 的 dataset
- **THEN** 运行 metadata MUST 记录 image cache policy、cache dir、transform version、hit/miss 或 coverage 摘要
- **AND** metadata MUST 不包含旧 image motion cache 字段

#### Scenario: 未启用 image 不访问 image cache
- **WHEN** 用户运行 GPS+mmWave 或其它不包含 image 的配置
- **THEN** cache policy MUST 不检查、不创建、不读取、不写入 image-derived cache
- **AND** 缺失 image cache 不得阻止该任务运行

### Requirement: image-derived cache 预热
项目 MUST 提供可选的 image-derived cache 预热能力。预热 MUST 遵守 dataset split、enabled modalities、image profile、image size 和 cache policy，不得把生成的缓存纳入源码变更。

#### Scenario: 预热指定 split
- **WHEN** 用户运行 image-derived cache 预热入口并指定 MMW train split
- **THEN** 系统 MUST 为该 split 中启用 image modality 的样本生成 cache
- **AND** 预热报告 MUST 记录扫描样本数、生成数、跳过数、失败数、cache 总大小和输出目录

#### Scenario: 预热不改变样本契约
- **WHEN** 同一样本分别通过原始 image 路径和 image-derived cache 路径读取
- **THEN** 返回 tensor 的 shape、dtype、模态字段和 target 字段 MUST 保持一致
- **AND** focused tests MUST 覆盖该等价性
