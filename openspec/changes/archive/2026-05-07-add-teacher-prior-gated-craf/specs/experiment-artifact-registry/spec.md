## ADDED Requirements

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
