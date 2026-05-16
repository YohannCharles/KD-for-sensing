## ADDED Requirements

### Requirement: Snapshot fusion 配置
Fusion 配置体系 MUST 支持 snapshot next-frame no-KD baseline。该 baseline MUST 使用现有 `experiment.task: fusion` 输入路由、现有 `modalities` 标准化和现有 fusion batch 准备，但模型必须为无时序 snapshot fusion 模型。

#### Scenario: 五模态 snapshot fusion
- **WHEN** 用户加载五模态 snapshot fusion 配置
- **THEN** 配置 MUST 启用 `image`、`radar`、`gps`、`lidar` 和 `mmwave`
- **AND** dataset MUST 按现有 fusion 模态选择逻辑只读取启用模态
- **AND** 模型 MUST 对当前帧的五个模态表示执行融合
- **AND** 模型 MUST 输出 `[B, 1, num_classes]` logits

#### Scenario: 任意合法多模态 snapshot fusion
- **WHEN** 用户加载 `configs/fusion/<slug>_snapshot_next_frame_no_kd.yaml` 且 `<slug>` 是两个到五个合法模态组成的 canonical slug
- **THEN** 系统 MUST 使用 `<slug>` 表示的模态集合构建 snapshot fusion
- **AND** forward MUST 只要求该模态集合对应的输入张量
- **AND** 未启用模态缺失不得阻止该配置运行

### Requirement: Snapshot fusion 不依赖 legacy fusion GRU
Snapshot fusion baseline MUST 不使用 `fusion_teacher`、`fusion_student` 的 GRU 路线作为主模型。若配置中保留 teacher 字段用于兼容结构，训练主模型 MUST 仍是无时序 snapshot 模型。

#### Scenario: no-KD snapshot fusion 主模型
- **WHEN** 用户训练 snapshot fusion no-KD 配置
- **THEN** 可训练主模型 MUST 为无时序 snapshot 模型
- **AND** 训练流程 MUST 不构建 frozen teacher checkpoint
- **AND** `distillation.teacher_model_name` MUST 为 `null`

#### Scenario: legacy fusion GRU 不参与 snapshot forward
- **WHEN** snapshot fusion 模型执行 forward
- **THEN** forward 路径 MUST 不调用 legacy `fusion_teacher` 或 `fusion_student` 的 GRU 层
- **AND** output diagnostics 或 final config MUST 标记 `uses_temporal_core: false`
