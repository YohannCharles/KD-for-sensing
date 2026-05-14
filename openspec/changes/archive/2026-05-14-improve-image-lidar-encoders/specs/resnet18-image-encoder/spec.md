## ADDED Requirements

### Requirement: Canonical camera baseline 默认使用预训练 ResNet-18
系统 MUST 将 `rgb_imagenet` 的 canonical camera baseline 配置默认绑定到 ImageNet 预训练 ResNet-18 encoder，而不是从头训练的小 CNN。该默认行为 MUST 覆盖 image-only teacher/no-KD baseline 和包含 image 的论文式 fusion teacher baseline。

#### Scenario: image teacher no-KD 默认使用 ResNet-18
- **WHEN** 用户运行默认 image teacher/no-KD 配置
- **THEN** 系统 MUST 使用 `rgb_imagenet` image profile
- **AND** 系统 MUST 构建包含 `resnet18_imagenet_rgb` encoder 的模型
- **AND** ResNet-18 encoder MUST 默认配置 `pretrained: true`
- **AND** ResNet-18 encoder MUST 默认使用 ImageNet `DEFAULT` 权重或等价 torchvision 权重枚举

#### Scenario: 包含 image 的 fusion teacher 默认使用 ResNet-18
- **WHEN** 用户运行包含 image 的 canonical fusion teacher/no-KD 配置
- **THEN** image 分支 MUST 使用与 image-only teacher baseline 等价的 `rgb_imagenet` ResNet-18 encoder profile
- **AND** 训练输出 metadata MUST 记录 image encoder 类型、预训练权重来源和实际可训练 stage

### Requirement: 旧小 CNN image 配置入口不得保留
系统 MUST NOT 在默认、canonical 或 legacy/ablation 配置入口中继续选择从头训练的小 CNN image encoder。

#### Scenario: image student/KD 不使用旧 CNN
- **WHEN** 用户加载 image student/no-KD、logits-KD 或 RKD 配置
- **THEN** 系统 MUST 构建包含 `resnet18_imagenet_rgb` encoder 的 `modular_sequence`
- **AND** 系统 MUST NOT 构建旧 `image_teacher` 或 `image_student` 小 CNN 模型

#### Scenario: 默认配置不静默回退到小 CNN
- **WHEN** 用户加载默认 image teacher/no-KD 或论文式 camera baseline 配置
- **THEN** 系统 MUST NOT 构建从头训练的小 CNN image encoder
- **AND** 如果 ResNet-18 依赖不可用，系统 MUST 抛出清晰错误，而不是静默回退到 legacy CNN
