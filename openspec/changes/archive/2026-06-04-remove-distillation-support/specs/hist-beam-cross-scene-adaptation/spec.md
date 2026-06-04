## MODIFIED Requirements

### Requirement: HiST-Beam 主线默认 no-KD
HiST-Beam 跨场景适配主线 MUST 默认使用 supervised/adaptation 训练。source-only、shared/private、adapter-only、adapter+prototype、path/radio prototype、history-anchored residual、private calibration 和 full fine-tuning baseline MUST 不要求 teacher-student distillation、teacher checkpoint 或 KD loss。

#### Scenario: 默认 LOSO plan 不加载 teacher
- **WHEN** 用户运行默认 HiST-Beam quick smoke、quick validation、MMW sensor-assisted 或 history-anchored residual 配置
- **THEN** run plan MUST 不包含 KD baseline variant
- **AND** 训练流程 MUST 不加载 teacher checkpoint
- **AND** run metadata MUST 不写 distillation 字段

#### Scenario: target adaptation 不计算 KD loss
- **WHEN** HiST-Beam target adaptation 执行 support adaptation 或 target test evaluation
- **THEN** loss MUST 来自 supervised/adaptation、prototype、residual、calibration 或 workflow-specific objective
- **AND** loss diagnostics MUST 不包含 `loss/distillation`

## REMOVED Requirements

### Requirement: KD 仅作为显式 HiST-Beam baseline 或增强
**Reason**: KD baseline 和增强不再作为当前源码支持能力。
**Migration**: 使用现有 HiST-Beam supervised/adaptation variants；未来 KD 需独立提案。

#### Scenario: 显式运行 HiST-Beam KD baseline
- **WHEN** 用户选择旧 HiST-Beam KD baseline profile
- **THEN** 系统 MUST 拒绝该 profile
- **AND** summary MUST 不生成 KD baseline 分组

