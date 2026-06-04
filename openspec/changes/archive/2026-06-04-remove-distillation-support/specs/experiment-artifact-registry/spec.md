## MODIFIED Requirements

### Requirement: checkpoint 解析优先级
评估权重解析 MUST 支持从最佳 checkpoint 归档目录加载匹配 checkpoint。显式传入的绝对路径或评估入口 `--weights` MUST 保持最高优先级；未显式指定时，系统 MAY 查找归档目录中的匹配 checkpoint。训练流程 MUST 不再为了 KD teacher 加载 checkpoint，且 MUST 不读取 `distillation.teacher_model_name`。

#### Scenario: 显式权重路径覆盖 registry
- **WHEN** 用户通过评估入口 `--weights` 或配置中的绝对路径显式指定 checkpoint
- **THEN** 系统 MUST 加载该显式路径
- **AND** 系统 MUST 不用归档目录中的候选替换该显式路径

#### Scenario: 训练不解析 KD teacher checkpoint
- **WHEN** 用户启动任一受支持训练配置
- **THEN** 训练流程 MUST 不调用 KD teacher checkpoint 解析
- **AND** 配置中若出现 `distillation.teacher_model_name` MUST 在配置解析阶段失败

### Requirement: 归档 metadata 与归一化工件关联
归档 checkpoint MUST 具备可机器读取的 metadata，用于记录源运行目录、配置 slug、模态、训练模式、epoch、验证 Top-1 accuracy、源 checkpoint 路径、split 信息和训练归一化工件路径。启用 GPS、LiDAR 或 mmWave 归一化时，metadata MUST 能让评估入口复用训练时的 scaler 或 normalizer/stats。

#### Scenario: 写入归档 sidecar
- **WHEN** 系统将 checkpoint 复制到归档目录
- **THEN** 系统 MUST 写入同名或可关联的 JSON sidecar metadata
- **AND** metadata MUST 记录验证 Top-1 accuracy、源 `run_dir`、源 checkpoint、配置 slug、训练模式和启用模态
- **AND** metadata MUST 不记录 KD 模式

## REMOVED Requirements

### Requirement: Teacher checkpoint artifact compatibility
**Reason**: teacher checkpoint artifact compatibility 只服务 KD teacher 加载和旧 teacher-prior 路线。
**Migration**: 使用普通评估 `--weights`、best checkpoint registry 和 normalization artifact metadata。

#### Scenario: 旧 checkpoint registry 继续可用
- **WHEN** 用户运行旧单模态 KD 配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不解析 teacher checkpoint

