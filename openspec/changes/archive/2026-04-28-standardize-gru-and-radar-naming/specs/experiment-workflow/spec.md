## MODIFIED Requirements

### Requirement: 训练与评估行为等价
结构重构后，默认 image-only、radar-only、GPS-only 和 fusion 工作流 MUST 通过新脚本保持当前算法的核心训练、验证和评估语义，包括默认序列长度、预测步数、类别数、KD 模式、teacher 权重选择、student 架构选择、early stopping、gradient clipping、checkpoint 恢复和指标计算。所有受支持默认实验配置中的 teacher 与 student `gru_params` MUST 统一为 `[64, 64, 2]`。

#### Scenario: 新配置默认参数
- **WHEN** 用户使用新脚本和默认配置启动训练或评估
- **THEN** 系统 MUST 使用从旧实现迁移而来的默认任务语义，并保持相同的任务类型
- **AND** `configs/image/*.yaml`、`configs/radar/*.yaml`、`configs/gps/*.yaml` 和 `configs/fusion/*.yaml` 中的 `model.teacher.gru_params` 与 `model.student.gru_params` MUST 为 `[64, 64, 2]`
- **AND** `src/kd_sensing/config/defaults.py` 中的默认 teacher/student `gru_params` MUST 为 `[64, 64, 2]`

#### Scenario: 默认 student 架构与 GRU 层数
- **WHEN** 用户使用默认 image-only、radar-only、GPS-only 或 fusion student 实验配置构建模型
- **THEN** 系统 MUST 为 image-only 工作流构建轻量 `image_student`
- **AND** 系统 MUST 为 radar-only 工作流构建轻量 `radar_student`
- **AND** 系统 MUST 为 GPS-only 工作流构建轻量 `gps_student`
- **AND** 系统 MUST 为 fusion 工作流构建轻量 `fusion_student`
- **AND** 默认 student 模型的 `GRU.num_layers` MUST 为 2
- **AND** 文档 MUST 说明旧的一层 GRU student checkpoint 与新的默认二层 GRU 配置不完全兼容

#### Scenario: 默认 teacher GRU 层数
- **WHEN** 用户通过默认配置构建 image、radar、GPS 或 fusion teacher 模型
- **THEN** teacher 模型的 `GRU.num_layers` MUST 为 2
- **AND** teacher 配置 MUST 使用 `gru_params: [64, 64, 2]`

#### Scenario: dry-run 训练
- **WHEN** 开发者使用 synthetic 或小比例数据运行一次短训练 smoke test
- **THEN** 训练流程 MUST 完成 forward、loss、backward、optimizer step、validation 和 checkpoint 保存的核心路径
