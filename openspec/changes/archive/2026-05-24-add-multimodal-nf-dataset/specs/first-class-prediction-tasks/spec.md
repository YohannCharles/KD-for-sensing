## ADDED Requirements

### Requirement: Near-field beam selection objective
系统 MUST 支持 `experiment.objective: near_field_beam_selection`，用于 Multimodal-NF 当前 frame 的近场三维 codebook beam selection。该目标 MUST 与 DeepSense6G future beam prediction 区分，并 MUST 不计算 future-only DBA 或 future horizon 指标。

#### Scenario: 解析 near-field beam objective
- **WHEN** 用户加载 `experiment.objective: near_field_beam_selection`
- **THEN** 系统 MUST 将主 target 解析为当前 `target_beam`
- **AND** 系统 MUST 要求或解析 codebook metadata
- **AND** 默认 early stopping metric MUST 为 `val_beam_top1`
- **AND** 默认 metric mode MUST 为 `max`

#### Scenario: 拒绝 future horizon 语义
- **WHEN** Multimodal-NF 配置包含 future-only `num_pred > 1`、future beam horizon 或 DeepSense sequence-only target
- **THEN** 系统 MUST 拒绝该配置或将其标准化为 current frame 语义
- **AND** 错误信息 MUST 指向 `near_field_beam_selection` 当前 frame objective

### Requirement: 三维 codebook target schema
系统 MUST 支持 near-field beam target schema，用于描述三维 beam triplet、flattened class、Top-5 候选、beam power 和 codebook shape。target helper MUST 输出主训练 label，并保留结构化 metadata。

#### Scenario: 准备主 label
- **WHEN** batch 包含 `target_beam`
- **THEN** target helper MUST 返回形状兼容 `[B, 1]` 的 beam class labels
- **AND** labels MUST 表示 Top-1 三维 triplet flatten 后的 class id

#### Scenario: 准备 Top-5 metadata
- **WHEN** batch 包含 `beam_triplet_topk` 和 `beam_power_topk`
- **THEN** target helper 或 metrics payload MUST 保留这些字段用于诊断
- **AND** 主 loss MUST 默认只使用 `target_beam`
- **AND** 缺失 Top-5 metadata 时系统 MUST 能继续训练主 beam classification，并在 metadata 中记录不可用状态

### Requirement: Near-field beam loss 契约
系统 MUST 为 `near_field_beam_selection` 计算主分类 loss。默认 loss MUST 使用 flattened `target_beam` 与模型输出 beam logits；结构化 triplet loss 或 beam power weighting 只能在配置显式启用时参与。

#### Scenario: 默认分类 loss
- **WHEN** `experiment.objective` 为 `near_field_beam_selection`
- **THEN** 总 loss 默认 MUST 等于当前 beam classification 主 loss
- **AND** LoS、NF、position 或 beam power 辅助项 MUST 不参与反向传播，除非配置显式启用

#### Scenario: 输出维度校验
- **WHEN** 模型输出 beam logits 的类别数与 codebook flattened class 数不一致
- **THEN** 系统 MUST 拒绝训练或评估
- **AND** 错误信息 MUST 包含模型输出类别数、codebook shape 和期望类别数

### Requirement: Near-field beam 指标
系统 MUST 为 near-field beam selection 输出当前 frame Top-K 指标。指标 MUST 基于当前 frame label，不得使用 DeepSense future-only DBA 口径。

#### Scenario: Top-K 指标
- **WHEN** 验证或评估 near-field beam selection objective
- **THEN** metrics MUST 包含 `val_beam_top1`、`val_beam_top3` 和 `val_beam_top5`
- **AND** `available_metrics` MUST 包含这些字段和 `val_loss`
- **AND** validation metrics MUST NOT 产生 `val_adba`

#### Scenario: triplet Top-5 命中诊断
- **WHEN** batch 提供 `beam_triplet_topk`
- **THEN** metrics MAY 输出 triplet Top-5 命中或等价诊断字段
- **AND** 该诊断字段 MUST 明确标注为 current near-field codebook metric

### Requirement: Near-field objective 运行 metadata
系统 MUST 在训练产物、评估报告和 final config runtime metadata 中记录 near-field objective、codebook shape、flatten 规则、target schema、启用模态、input profiles 和辅助标签可用性。

#### Scenario: 记录 codebook metadata
- **WHEN** near-field beam selection 训练完成
- **THEN** final config 或 run metadata MUST 记录 codebook shape、codebook 文件路径或 fingerprint、flatten order 和 num beam classes
- **AND** checkpoint metadata MUST 记录 objective 为 `near_field_beam_selection`

#### Scenario: 记录辅助标签可用性
- **WHEN** Multimodal-NF dataset 暴露 `los_label`、`nf_label` 或 trajectory mode
- **THEN** run metadata MUST 记录这些辅助标签是否可用
- **AND** 如果辅助标签未参与主 loss，metadata MUST 明确其诊断或过滤用途
