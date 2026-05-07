## ADDED Requirements

### Requirement: Fusion 配置选择 CRAF 模型
Fusion 配置 MUST 能显式选择 CRAF 或 CRAF baseline 模型，同时继续使用现有 `modalities` 字段描述参与融合的模态集合。

#### Scenario: 配置 CRAF fusion
- **WHEN** 用户在 fusion 配置中设置 `model.student.type: craf_fusion`
- **THEN** 系统 MUST 使用 `model.student.modalities` 构建 CRAF 模型
- **AND** 系统 MUST 继续使用 `experiment.task: fusion` 的 batch 输入准备流程

#### Scenario: 配置 token transformer fusion
- **WHEN** 用户在 fusion 配置中设置 token-only transformer baseline 的注册名
- **THEN** 系统 MUST 使用同一模态集合构建不带 reliability gate 的 token fusion baseline

#### Scenario: legacy fusion 配置不变
- **WHEN** 用户继续运行既有 `fusion_teacher` 或 `fusion_student` 配置
- **THEN** 系统 MUST 保持 early-concat fusion 行为
- **AND** 系统 MUST 不隐式启用 CRAF 训练 loss 或 diagnostics

### Requirement: CRAF 配置复用 fusion 模态校验
CRAF 和 CRAF baseline 配置 MUST 复用现有 fusion 模态标准化和校验语义。模态顺序、未知模态错误、重复模态错误和未启用模态输入行为 MUST 与当前 fusion 配置一致。

#### Scenario: 乱序模态标准化
- **WHEN** 用户在 CRAF 配置中设置 `modalities: ["lidar", "image", "gps"]`
- **THEN** 系统 MUST 将模态顺序标准化为项目固定顺序
- **AND** reliability 输出、token 输出和日志 MUST 使用标准化后的顺序

#### Scenario: 未启用模态不读取输入
- **WHEN** CRAF 配置不包含 `mmwave`
- **THEN** batch 准备和模型 forward MUST 不要求 `mmwave` 输入

### Requirement: CRAF canonical 与示例配置
项目 MUST 提供可运行的 CRAF 示例配置和 baseline 示例配置，用于当前项目数据和训练入口的 smoke test 与实验对比。

#### Scenario: all-modalities CRAF 配置
- **WHEN** 用户加载 all-modalities CRAF 示例配置
- **THEN** 配置 MUST 启用 image、radar、GPS、LiDAR 和 mmWave
- **AND** 配置 MUST 设置 CRAF 所需的模型、loss、counterfactual 和输出字段

#### Scenario: image-radar CRAF 配置
- **WHEN** 用户加载 image+radar CRAF 示例配置
- **THEN** 配置 MUST 使用与现有 image+radar fusion 可比较的 dataset split、num classes、seq length 和 num pred

#### Scenario: baseline 配置可加载
- **WHEN** 用户加载 token transformer 或 early concat transformer baseline 配置
- **THEN** 配置 MUST 通过现有 config loader 构建成功
- **AND** 配置 MUST 不要求新增训练入口

### Requirement: CRAF 稳定化配置字段
CRAF 示例配置 MUST 能表达稳定化训练策略，包括 warmup、CE-only counterfactual、ignore band、softmax gate、temperature schedule 和 auxiliary loss schedule。

#### Scenario: all-modalities 稳定化 CRAF 配置
- **WHEN** 用户加载 all-modalities CRAF 稳定化配置
- **THEN** 配置 MUST 启用五个标准模态
- **AND** 配置 MUST 设置 warmup epoch、counterfactual 起始 epoch、`context_marginal` 或等价反事实模式、CE-only delta 和 ignore band

#### Scenario: softmax gate 配置
- **WHEN** 用户在 CRAF 配置中设置 `model.student.reliability.gate_type: softmax`
- **THEN** 配置 MUST 能传递 gate temperature、temperature schedule 和 `min_gate`
- **AND** 模型构建 MUST 不影响配置为 sigmoid gate 的旧实验

#### Scenario: auxiliary 与 beam soft schedule 配置
- **WHEN** 用户配置 CRAF 附加 loss
- **THEN** 配置 MUST 能表达 warmup-only 单模态 auxiliary loss 和 beam soft loss 权重
- **AND** 权重为 0 的附加 loss MUST 保持关闭
