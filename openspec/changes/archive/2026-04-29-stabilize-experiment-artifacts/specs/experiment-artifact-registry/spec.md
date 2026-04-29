## ADDED Requirements

### Requirement: 最佳 checkpoint 归档
训练流程 MUST 提供统一的最佳 checkpoint 归档能力。默认归档目录 MUST 为 `outputs/best_checkpoints/`，并且可通过配置覆盖。每次训练结束时，归档目录 MUST 至少保存当前配置在验证 Top-1 accuracy 上最高的 checkpoint；当训练过程中出现新的最高验证 Top-1 accuracy 时，系统 MAY 立即更新归档。

#### Scenario: teacher no-KD 训练归档最高精度 checkpoint
- **WHEN** 用户运行 `configs/<modality>/teacher_no_kd.yaml` 或等价 teacher no-KD 配置完成训练
- **THEN** 系统 MUST 将该配置验证 Top-1 accuracy 最高的 checkpoint 复制到归档目录
- **AND** 归档文件名 MUST 包含配置 slug、`teacher_no_kd` 和 `acc_<val_top1>`，例如 `<slug>_teacher_no_kd_acc_<val_top1>.pth`
- **AND** 原运行目录下的 checkpoint MUST 保留，不得被移动或删除

#### Scenario: 同一 slug 刷新最高精度
- **WHEN** 同一配置 slug 产生新的更高验证 Top-1 accuracy checkpoint
- **THEN** 归档目录 MUST 指向或保留该最高精度 checkpoint 作为默认候选
- **AND** 系统 MUST 避免默认解析到同一 slug 的旧低精度 checkpoint

### Requirement: checkpoint 解析优先级
KD teacher 和评估权重解析 MUST 支持从最佳 checkpoint 归档目录加载匹配 checkpoint。显式传入的绝对路径或评估入口 `--weights` MUST 保持最高优先级；未显式指定时，系统 MUST 优先查找归档目录中的匹配 checkpoint，再回退到 `paths.weights_dir / distillation.teacher_model_name` 或旧评估配置路径。

#### Scenario: KD teacher 从归档目录加载
- **WHEN** 用户运行 KD 配置且未显式覆盖 teacher checkpoint 为绝对路径
- **THEN** 系统 MUST 根据配置推导对应 teacher baseline slug
- **AND** 如果归档目录存在该 slug 的最高验证 Top-1 checkpoint，系统 MUST 加载该 checkpoint 作为 frozen teacher
- **AND** `checkpoint_loads` MUST 记录最终加载路径和来源为 registry

#### Scenario: 显式权重路径覆盖 registry
- **WHEN** 用户通过评估入口 `--weights` 或配置中的绝对路径显式指定 checkpoint
- **THEN** 系统 MUST 加载该显式路径
- **AND** 系统 MUST 不用归档目录中的候选替换该显式路径

#### Scenario: registry 缺失时回退旧路径
- **WHEN** 归档目录没有匹配当前配置的 checkpoint
- **THEN** 系统 MUST 尝试使用既有 `paths.weights_dir` 和 `teacher_model_name` 解析逻辑
- **AND** 如果回退路径也不存在，系统 MUST 抛出包含 registry 候选和旧路径候选的清晰错误

### Requirement: 归档 metadata 与归一化工件关联
归档 checkpoint MUST 具备可机器读取的 metadata，用于记录源运行目录、配置 slug、模态、KD 模式、epoch、验证 Top-1 accuracy、源 checkpoint 路径、split 信息和训练归一化工件路径。启用 GPS 或 LiDAR 归一化时，metadata MUST 能让评估入口复用训练时的 scaler 或 normalizer/stats。

#### Scenario: 写入归档 sidecar
- **WHEN** 系统将 checkpoint 复制到归档目录
- **THEN** 系统 MUST 写入同名或可关联的 JSON sidecar metadata
- **AND** metadata MUST 记录验证 Top-1 accuracy、源 `run_dir`、源 checkpoint、配置 slug 和启用模态

#### Scenario: 评估复用归一化工件
- **WHEN** 用户评估一个 registry checkpoint 且 metadata 记录了 GPS scaler 或 LiDAR normalizer/stats 路径
- **THEN** 评估入口 MUST 加载 metadata 中的归一化工件
- **AND** 评估入口 MUST 不为了重新 fit 归一化状态而扫描训练 split
