# u-mask-beam-jepa Specification

## Purpose

定义 T2/S1、BCACL U2 与 CMSBL 所需的最小 U-MaskBeamJEPA 模型、loss 和 mask 契约。

## Requirements

### Requirement: U-MaskBeamJEPA 仅支持 current 主线

U-MaskBeamJEPA MUST 只保留 supervised router、prototype/BPA、embedded full-modal teacher、same-model temporal superset consistency、active BPA/CMA、BCACL U2 private/shared supervision 和 CMSBL M1--M3。系统 MUST 不提供 PCER、PGCD、候选动态 Router、cached reroute、BCACL teacher/quality/two-stage 或历史 compatibility fields。

#### Scenario: 构建 T2/S1

- **WHEN** retained recipe 构建 U-MaskBeamJEPA
- **THEN** forward MUST 消费统一四模态 mask 并输出 beam logits、unimodal logits 和 current Router payload
- **AND** 默认路径 MUST 不读取 outputs、cache、teacher checkpoint 或 capacity stats

### Requirement: 四模态与 temporal mask 语义稳定

模型 MUST 使用 `image,radar,gps,lidar` 顺序和 `[B,T,M]` availability mask。缺失位置 MUST 不进入 temporal pooling、prototype fusion 或 Router 归一化，且每个样本 MUST 至少有一个可用模态。

#### Scenario: 部分模态或时间缺失

- **WHEN** batch 提供 modality temporal mask
- **THEN** 缺失单元的融合权重 MUST 为零
- **AND** 可用模态权重 MUST 归一化为一

### Requirement: BCACL U2 只增加训练期单模态监督

BCACL U2 MUST 为每模态提供 projection/private head，并提供共享参数 shared head。其 loss MUST 只使用 dropout 前自然观测的 `observed_mask`，不得将 synthetic `fusion_mask` 解释为自然缺失；推理 MUST 不消费 BCACL logits。

#### Scenario: synthetic dropout

- **WHEN** 自然观测模态被训练 mask 从 fusion 中移除
- **THEN** 它 MAY 参与 U2 private/shared supervision
- **AND** 它 MUST 不进入当前样本融合预测

### Requirement: CMSBL per-sample loss 不改变 disabled path

loss owner MUST 只在 CMSBL hard-mask 启用时提供 per-sample fusion CE、fused prototype loss和按样本可用模态归一化的 modality prototype loss。CMSBL 关闭时 MUST 保持原标量归约与数值行为。

#### Scenario: 混合 availability batch

- **WHEN** CMSBL hard-mask 对 single、double 和 full 样本归约 loss
- **THEN** 每个样本的 restoration MUST 先按自身可用模态归一化
- **AND** 再使用该样本的 canonical mask weight

### Requirement: inactive training branch 不进入 optimizer

BCACL/CMSBL 关闭时 MUST 不创建额外 trainable parameter；classifier/prototype inactive head 或禁用 auxiliary MUST 不进入 optimizer trainable set。metadata MUST 声明 active head、current fusion 和 trainable parameter count。

#### Scenario: canonical T2

- **WHEN** T2 使用 prototype head且 BCACL/CMSBL disabled
- **THEN** optimizer MUST 只包含 current T2 trainable parameter
- **AND** state dict MUST 与未声明 BCACL/CMSBL 时一致
