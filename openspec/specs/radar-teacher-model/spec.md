# radar-teacher-model Specification

## Purpose
TBD - created by archiving change add-radar-teacher-model. Update Purpose after archive.
## Requirements
### Requirement: RadarTeacher 模型结构
系统 MUST 提供已注册的 `radar_teacher` 模型，用于 radar-only beam prediction。该模型 MUST 接收 RA/DA 拼接后的雷达序列张量，使用任务特定 CNN embedding 提取每个时隙的低维特征，经过 LayerNorm、GRU temporal modeling、MHA residual prediction module 和 MLP classifier 后输出 beam logits。

#### Scenario: 按配置构建 RadarTeacher
- **WHEN** 配置中指定 `model.student.type: radar_teacher` 或 `model.teacher.type: radar_teacher`
- **THEN** 系统 MUST 通过 `MODELS` 注册表构建 RadarTeacher 实例
- **AND** 构建参数 MUST 支持 `feature_size`、`num_classes`、`gru_params`、`radar_channels` 和 `num_heads`

#### Scenario: RadarTeacher 前向输出契约
- **WHEN** RadarTeacher 接收形状为 `(batch, sequence, channels, height, width)` 的雷达输入张量
- **THEN** 模型 MUST 返回 `(pred, features, enhanced_seq_out)`
- **AND** `pred` 的形状 MUST 为 `(batch, sequence, num_classes)`
- **AND** `features` 的形状 MUST 为 `(batch, sequence, feature_size)`
- **AND** `enhanced_seq_out` 的 batch 与 sequence 维度 MUST 与输入一致

#### Scenario: MHA 参数校验
- **WHEN** `gru_hidden_size` 不能被 `num_heads` 整除
- **THEN** RadarTeacher MUST 在构建时抛出明确异常，说明 MHA head 配置无效

### Requirement: Radar-only 输入准备
系统 MUST 提供 radar-only 输入准备路径，从 batch 中读取 `radar_ra` 和 `radar_da`，在 channel 维拼接为雷达模型输入，并按现有预测窗口规则补齐未来占位帧。

#### Scenario: 准备 radar-only batch
- **WHEN** 训练、验证或评估流程处理 `experiment.task: radar`
- **THEN** 系统 MUST 使用 `radar_ra` 和 `radar_da` 构造雷达输入
- **AND** 系统 MUST 不要求图像输入参与模型 forward

#### Scenario: 雷达预测窗口对齐
- **WHEN** `seq_length` 为 8 且 `num_pred` 为 3
- **THEN** radar-only 输入 MUST 包含最近 8 个雷达时隙和 2 个未来 zero padding 时隙
- **AND** 验证和损失计算 MUST 继续使用最后 `num_pred + 1` 个输出时隙与标签对齐

### Requirement: Radar-only 基线配置
项目 MUST 提供 radar-only 配置，用于训练和评估论文表格中的 Radar 对照基线。该配置 MUST 使用 `experiment.task: radar`，并将训练主模型配置为 `radar_teacher`。

#### Scenario: 启动 radar-only no-KD 训练
- **WHEN** 用户使用 radar-only no-KD 配置运行训练入口
- **THEN** 系统 MUST 构建 `radar_teacher` 作为被优化的主模型
- **AND** 训练流程 MUST 完成 forward、task loss、backward、optimizer step、validation 和 checkpoint 保存

#### Scenario: 评估 RadarTeacher 指标
- **WHEN** 用户使用 radar-only 配置和 RadarTeacher 权重运行评估入口
- **THEN** 系统 MUST 输出 Top-K、DBA、loss 和测试报告
- **AND** 输出指标 MUST 支持计算论文对照表需要的 ATop-3、ATop-5 和 ADBA

