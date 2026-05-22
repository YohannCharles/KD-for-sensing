# original-code-compatibility Specification

## Purpose
定义与原始工程复现实验的 checkpoint、配置和行为兼容边界，确保迁移到包结构后仍能复核历史实验结果。
## Requirements
### Requirement: Checkpoint 加载可诊断
项目 MUST 默认严格加载 teacher、评估和 resume checkpoint。权重结构不匹配时，系统 MUST 抛出包含 checkpoint 路径、模型角色、missing keys 和 unexpected keys 的明确错误；只有用户显式选择非严格加载时，系统 MAY 继续运行。

#### Scenario: teacher 权重结构不匹配
- **WHEN** KD teacher 加载的一层 GRU checkpoint 被用于二层 GRU teacher 配置
- **THEN** 系统 MUST 拒绝静默加载
- **AND** 错误信息 MUST 包含缺失的 `GRU.weight_ih_l1`、`GRU.weight_hh_l1`、`GRU.bias_ih_l1` 或 `GRU.bias_hh_l1` 中至少一个 key

#### Scenario: 评估权重结构不匹配
- **WHEN** 用户使用评估入口加载与当前 `model.student` 结构不匹配的权重
- **THEN** 系统 MUST 拒绝静默加载
- **AND** 错误信息 MUST 指出评估权重路径和不匹配 key

#### Scenario: 显式非严格加载
- **WHEN** 用户通过配置显式请求非严格加载 checkpoint
- **THEN** 系统 MAY 调用非严格加载
- **AND** 系统 MUST 在日志或返回结果中记录 missing keys 和 unexpected keys

### Requirement: 恢复训练
训练入口 MUST 让 `training.resume` 生效，并在恢复时加载 student 模型、optimizer、scheduler、已完成 epoch 和 best validation loss。恢复训练 MUST 继续使用统一输出目录、checkpoint 保存和 early stopping 语义。

#### Scenario: 从 last checkpoint 恢复
- **WHEN** 用户设置 `training.resume: true` 且 `output.run_name` 指向已有运行目录
- **THEN** 系统 MUST 从该运行目录的 `checkpoints/last.pth` 加载 checkpoint
- **AND** 后续训练 MUST 从 checkpoint 中记录的下一轮 epoch 开始
- **AND** optimizer、scheduler 和 best validation loss MUST 被恢复

#### Scenario: 从显式路径恢复
- **WHEN** 用户设置 `training.resume` 为 checkpoint 文件路径
- **THEN** 系统 MUST 从该路径加载 checkpoint
- **AND** `training.start_epoch` MUST 仅在 checkpoint 缺少 epoch 字段时作为兜底

#### Scenario: 恢复路径不存在
- **WHEN** 用户启用 resume 但目标 checkpoint 不存在
- **THEN** 系统 MUST 在训练开始前抛出明确错误
- **AND** 错误信息 MUST 包含尝试恢复的 checkpoint 路径

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

### Requirement: 原代码兼容仅保留迁移说明
项目 MUST 不再为上游原代码入口、旧配置矩阵、旧 checkpoint fallback 或旧输入结构提供运行兼容。文档可以保留历史差异说明，但训练、评估、配置解析和 registry MUST 只支持当前 canonical 路线。

#### Scenario: 旧原代码兼容配置被拒绝
- **WHEN** 用户加载只为复现上游旧入口保留的配置路径或字段
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 指向当前 canonical 配置或要求显式提供 checkpoint

#### Scenario: 历史说明不产生运行入口
- **WHEN** 开发者阅读 README 或扩展指南中的历史差异说明
- **THEN** 文档 MUST 不推荐旧脚本、旧 config alias 或旧权重 fallback 作为可运行入口
- **AND** 文档 MUST 给出当前 canonical 训练和评估路线
