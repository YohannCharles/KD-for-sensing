## ADDED Requirements

### Requirement: ResNet-18 RGB 路径成为当前 image 默认
新增 ResNet-18 RGB image 路径 MUST 作为默认 RGB 实验入口存在。当前 image-only 与包含 image 的 fusion 配置 MUST 使用 RGB/ImageNet preprocessing、3 通道输入和当前 checkpoint registry 语义。

#### Scenario: image 配置使用 RGB/ImageNet
- **WHEN** 开发者加载 `configs/image/teacher_no_kd.yaml`、`configs/image/student_no_kd.yaml`、`configs/image/logits_kd.yaml` 或 `configs/image/rkd.yaml`
- **THEN** 配置解析后的 image profile MUST 为 `rgb_imagenet`
- **AND** 模型 MUST 使用可接收 3 通道 image tensor 的 branch 或 encoder

#### Scenario: image fusion 配置使用 RGB/ImageNet
- **WHEN** 开发者加载包含 image 的 fusion 配置或 `image_radar_*` canonical 配置
- **THEN** 配置解析后的 image profile MUST 为 `rgb_imagenet`
- **AND** fusion teacher/student 的 image branch MUST 接收 3 通道 RGB/ImageNet tensor

### Requirement: ResNet-18 配置不复用旧 checkpoint
系统 MUST 避免把旧 image checkpoint 静默加载到 ResNet-18 RGB image 模型中，或把 ResNet-18 checkpoint 静默加载到不兼容模型中。结构不匹配时 MUST 沿用严格 checkpoint 加载错误。

#### Scenario: 旧 checkpoint 加载到 ResNet-18 被拒绝
- **WHEN** 用户使用 ResNet-18 RGB image 配置并提供不兼容 checkpoint
- **THEN** 系统 MUST 拒绝静默加载
- **AND** 错误信息 MUST 包含 checkpoint 路径、模型角色和不匹配 key

#### Scenario: ResNet-18 checkpoint 加载到不兼容模型被拒绝
- **WHEN** 用户使用不兼容 image 配置并提供 ResNet-18 RGB image checkpoint
- **THEN** 系统 MUST 拒绝静默加载
- **AND** 错误信息 MUST 包含 checkpoint 路径、模型角色和不匹配 key
