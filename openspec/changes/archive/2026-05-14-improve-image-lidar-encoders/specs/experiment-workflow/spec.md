## ADDED Requirements

### Requirement: 默认实验记录 encoder 和 preprocessing profile
训练、验证和评估流程 MUST 在运行产物中记录 camera encoder 与 LiDAR preprocessing profile，使不同单模态 baseline 的结果可以横向比较。

#### Scenario: 记录 image encoder profile
- **WHEN** 一次 image-only 或包含 image 的 fusion 训练启动
- **THEN** final_config 或运行 metadata MUST 记录 image profile、image encoder 类型、是否使用预训练权重、权重名称、freeze 策略和实际可训练 stage

#### Scenario: 记录 LiDAR preprocessing profile
- **WHEN** 一次 LiDAR-only 或包含 LiDAR 的 fusion 训练启动
- **THEN** final_config 或运行 metadata MUST 记录 LiDAR normalization、cache、ROI、FoV、ground/background filter 和安全增强配置

### Requirement: 单模态 baseline 回归检查
项目 MUST 提供面向 image 和 LiDAR 默认 baseline 的回归检查，防止默认配置重新退回到从头训练 camera encoder 或 LiDAR 多数类退化路径。

#### Scenario: image 默认配置回归检查
- **WHEN** 开发者运行配置测试
- **THEN** 测试 MUST 验证默认 image teacher/no-KD 配置使用 `resnet18_imagenet_rgb`
- **AND** 测试 MUST 验证该 encoder 配置启用 ImageNet 预训练权重

#### Scenario: LiDAR 默认配置回归检查
- **WHEN** 开发者运行配置测试
- **THEN** 测试 MUST 验证默认 LiDAR teacher/no-KD 配置显式启用 LiDAR streaming stats normalization
- **AND** 测试 MUST 验证该配置记录可追踪的 BEV ROI/cache 参数

#### Scenario: LiDAR 退化报告回归检查
- **WHEN** 开发者运行 LiDAR 评估或诊断测试
- **THEN** 输出报告 MUST 包含 majority-class baseline
- **AND** 输出报告 MUST 包含 LiDAR input quality summary
- **AND** 报告 MUST 能标记模型未超过 majority-class baseline 的退化风险
