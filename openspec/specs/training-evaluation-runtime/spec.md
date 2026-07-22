# training-evaluation-runtime Specification

## Purpose

定义 MMW T2/baseline、受限 DeepSense6G T2、BCACL U2 与 CMSBL 的共享训练、checkpoint 和评估边界。

## Requirements

### Requirement: T2 runtime 只执行 retained training objective

训练 runtime MUST 只执行 Beam supervision、BPA/CMA、Router supervision、embedded full-modal teacher CE、same-model temporal superset consistency、BCACL U2 和可选 CMSBL。Evaluation MUST 不执行 teacher/superset/BCACL/CMSBL training target forward。

#### Scenario: S1 或普通 T2

- **WHEN** 对应 objective disabled
- **THEN** trainer MUST 不保存其临时 payload或执行额外 forward
- **AND** 基础 beam、BPA 与 Router loss MUST 保持可用

### Requirement: CMSBL 状态只由训练更新并可恢复

runtime MUST 在 epoch 结束、checkpoint 写出前聚合 capacity/mask 长期状态并写出一个 JSON。resume MUST 恢复 reference identity、EMA、count 和 initialized 状态；validation/test MUST 只读。

#### Scenario: epoch-end checkpoint

- **WHEN** CMSBL 完成一个训练 epoch
- **THEN** runtime MUST 先更新状态和 JSON，再保存 `last.pth`
- **AND** resume 后 epoch accumulator MUST 为空

### Requirement: 评估使用 recipe 声明的数据集 protocol

evaluation MUST 以 recipe 声明的 MMW 或 DeepSense6G dataset、checkpoint、split 和 mask identity 产出指标。MMW fixed-mask MUST 保留 15 个非空组合及 Full/Single/Double/Triple/All-14 macro/worst。

#### Scenario: 评估 current checkpoint

- **WHEN** 用户评估 retained checkpoint
- **THEN** 输出 MUST 带 recipe、dataset、scene/domain、seed、split、mask 与 checkpoint provenance
- **AND** 不得执行 retired branch 或修改训练状态

### Requirement: runtime 保持证据与资源边界

训练和评估 MUST 使用 checkpoint 保存的 profile、GPS mode 和 train-fitted normalization artifact。默认 evaluation MUST 流式累计指标并关闭其创建的 dataloader workers；只有显式请求才捕获逐样本 prediction。

#### Scenario: 默认评估

- **WHEN** evaluator 未请求 capture
- **THEN** 只保留指标聚合状态
- **AND** 完成或失败后 MUST 关闭创建的 worker

### Requirement: development 与 partial 运行明确隔离

development 或 partial evaluation MUST 记录实际 sample/domain/mask 覆盖，标记 `development_partial=true`，并不得升级为正式 comparison evidence。

#### Scenario: 限制 batch 或 domain

- **WHEN** evaluator 使用 `max_batches` 或 `max_domains`
- **THEN** 输出 MUST 记录实际计数
- **AND** 正式 summary MUST 拒绝该输出
