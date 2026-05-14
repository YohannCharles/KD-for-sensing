# experiment-artifact-registry Specification

## Purpose
TBD - created by archiving change stabilize-experiment-artifacts. Update Purpose after archive.
## Requirements
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
KD teacher 和评估权重解析 MUST 支持从最佳 checkpoint 归档目录加载匹配 checkpoint。显式传入的绝对路径或评估入口 `--weights` MUST 保持最高优先级；未显式指定时，系统 MUST 查找归档目录中的匹配 checkpoint。归档目录缺失或无匹配时，系统 MUST 抛出清晰错误，且 MUST 不再回退到 legacy `paths.weights_dir / distillation.teacher_model_name` 或旧评估配置路径。

#### Scenario: KD teacher 从归档目录加载
- **WHEN** 用户运行 KD 配置且未显式覆盖 teacher checkpoint 为绝对路径
- **THEN** 系统 MUST 根据配置推导对应 teacher baseline slug
- **AND** 如果归档目录存在该 slug 的最高验证 Top-1 checkpoint，系统 MUST 加载该 checkpoint 作为 frozen teacher
- **AND** `checkpoint_loads` MUST 记录最终加载路径和来源为 registry

#### Scenario: 显式权重路径覆盖 registry
- **WHEN** 用户通过评估入口 `--weights` 或配置中的绝对路径显式指定 checkpoint
- **THEN** 系统 MUST 加载该显式路径
- **AND** 系统 MUST 不用归档目录中的候选替换该显式路径

#### Scenario: registry 缺失时报错
- **WHEN** 归档目录没有匹配当前配置的 checkpoint
- **THEN** 系统 MUST 抛出包含 registry 候选和显式 checkpoint 配置方式的清晰错误
- **AND** 系统 MUST 不尝试 legacy 权重目录 fallback

### Requirement: 归档 metadata 与归一化工件关联
归档 checkpoint MUST 具备可机器读取的 metadata，用于记录源运行目录、配置 slug、模态、KD 模式、epoch、验证 Top-1 accuracy、源 checkpoint 路径、split 信息和训练归一化工件路径。启用 GPS、LiDAR 或 mmWave 归一化时，metadata MUST 能让评估入口复用训练时的 scaler 或 normalizer/stats。

#### Scenario: 写入归档 sidecar
- **WHEN** 系统将 checkpoint 复制到归档目录
- **THEN** 系统 MUST 写入同名或可关联的 JSON sidecar metadata
- **AND** metadata MUST 记录验证 Top-1 accuracy、源 `run_dir`、源 checkpoint、配置 slug 和启用模态

#### Scenario: 评估复用归一化工件
- **WHEN** 用户评估一个 registry checkpoint 且 metadata 记录了 GPS scaler、LiDAR normalizer/stats 或 mmWave scaler 路径
- **THEN** 评估入口 MUST 加载 metadata 中的归一化工件
- **AND** 评估入口 MUST 不为了重新 fit 归一化状态而扫描训练 split

#### Scenario: 评估缺少 mmWave scaler 工件
- **WHEN** 用户评估启用 mmWave 归一化的 checkpoint 且 metadata 没有记录可用 mmWave scaler 路径
- **THEN** 评估入口 MUST 抛出清晰错误
- **AND** 错误信息 MUST 提示提供 mmWave scaler 或使用带 metadata 的训练 checkpoint

### Requirement: 场景隔离的最佳 checkpoint registry
最佳 checkpoint registry MUST 按 DeepSense6G 场景隔离。默认 registry 目录 MUST 位于当前场景输出分组下，例如 `outputs/scene9/best_checkpoints/` 和 `outputs/scene32/best_checkpoints/`。

#### Scenario: Scenario 9 registry 写入 scene9
- **WHEN** 用户运行 Scenario 9 teacher no-KD 训练并产生新的最高验证 Top-1 checkpoint
- **THEN** 系统 MUST 将归档 checkpoint 写入 `outputs/scene9/best_checkpoints/`
- **AND** metadata sidecar MUST 记录 `scene_id: 9` 和 `scene_slug: scene9`

#### Scenario: Scenario 32 registry 不复用 scene9
- **WHEN** 用户运行 Scenario 32 KD 配置且未显式指定绝对 teacher checkpoint
- **THEN** 系统 MUST 优先查找 `outputs/scene32/best_checkpoints/`
- **AND** 系统不得默认加载 `outputs/scene9/best_checkpoints/` 中同 slug 的 checkpoint

#### Scenario: 显式绝对 checkpoint 仍最高优先级
- **WHEN** 用户通过绝对路径显式指定 teacher checkpoint 或评估权重
- **THEN** 系统 MUST 使用该显式路径
- **AND** 场景 registry 不得替换该路径

### Requirement: Teacher reliability registry artifact
实验产物体系 MUST 支持 teacher reliability registry。该 registry MUST 按场景隔离，引用 teacher checkpoint 和指标来源，并能被 Stage 2/3 配置稳定解析。

#### Scenario: Scene32 teacher registry 写入 scene32 输出组
- **WHEN** 用户为 Scenario 32 构建 teacher reliability registry
- **THEN** 默认输出路径 MUST 位于 `outputs/scene32/`
- **AND** registry MUST 记录 `scene_id: 32` 或等价 scene metadata

#### Scenario: registry 引用 checkpoint metadata
- **WHEN** teacher checkpoint 有 checkpoint registry sidecar metadata
- **THEN** teacher reliability registry MUST 记录 checkpoint 路径
- **AND** registry MUST 保留可追溯到源 run_dir、epoch 和验证 Top-1 的 metadata 或引用

#### Scenario: Stage 2 解析 registry 路径
- **WHEN** Stage 2 配置提供相对 teacher registry 路径
- **THEN** 系统 MUST 按项目根目录解析该路径
- **AND** 如果文件不存在，错误信息 MUST 包含解析后的绝对路径

### Requirement: Teacher metrics export
单模态 teacher 训练产物 MUST 提供 teacher registry 可读取的指标文件或等价 metadata。指标 MUST 至少包含模态、best epoch、验证 Top-1、验证 Top-3、验证 Top-5、验证 ADBA 和训练 Top-1。

#### Scenario: teacher 训练完成写出 metrics
- **WHEN** 单模态 teacher 训练完成至少一个 epoch
- **THEN** 输出目录 MUST 包含可供 registry 构建脚本读取的指标数据
- **AND** 指标数据 MUST 包含 `modality`、`best_epoch`、`val_acc_top1`、`val_acc_top3`、`val_acc_top5`、`val_adba` 和 `train_acc_top1`

#### Scenario: metrics 与 checkpoint 模态不一致
- **WHEN** teacher metrics 中的 `modality` 与 registry 当前模态不一致
- **THEN** registry 构建流程 MUST 拒绝该输入
- **AND** 错误信息 MUST 包含期望模态和实际模态

### Requirement: Teacher-prior CRAF artifact compatibility
新增 teacher reliability registry MUST 不破坏现有 best checkpoint registry、normalization artifacts 和 train log 输出格式。

#### Scenario: 旧 checkpoint registry 继续可用
- **WHEN** 用户运行既有单模态 KD 或评估配置
- **THEN** 系统 MUST 继续按现有 best checkpoint registry 解析 teacher checkpoint
- **AND** 系统 MUST 不要求 teacher reliability registry 存在

#### Scenario: teacher-prior CRAF 记录 registry 引用
- **WHEN** Stage 2 或 Stage 3 使用 teacher reliability registry
- **THEN** `final_config.yaml` 或 `train_log.json` MUST 记录最终解析的 teacher registry 路径
- **AND** 训练日志 MUST 记录 registry 中每个启用模态的 checkpoint 和 prior

