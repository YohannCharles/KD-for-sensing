## MODIFIED Requirements

### Requirement: ResNet-18 image encoder 构建
系统 MUST 能构建 ImageNet 预训练 ResNet-18 image encoder，并使其输出 logits 兼容当前 beam prediction 训练、评估和诊断流程。该兼容性 MUST 不包含 distillation workflow。

#### Scenario: image encoder 训练兼容
- **WHEN** 用户运行默认 image strong 或 lightweight supervised 配置
- **THEN** 输出 logits MUST 能进入 beam supervised loss 和评估指标
- **AND** 系统 MUST 不要求 distillation loss 或 frozen teacher

