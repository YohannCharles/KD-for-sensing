## MODIFIED Requirements

### Requirement: Soft target supervised loss
系统 SHALL 在 beam soft target 可用且 soft target loss 启用时，使用 soft target distribution 计算主 beam supervised loss；若 soft target 不可用，MUST 回退到 hard-label loss。该流程 MUST 不经过 distillation runtime。

#### Scenario: supervised 主 loss 使用 soft target
- **WHEN** batch 包含 `target_beam_distribution` 且 `loss.soft_targets.enabled=true`
- **THEN** supervised loss MUST 消费 soft target distribution
- **AND** `loss/beam` 和 `loss/primary` MUST 记录 soft-target supervised loss
- **AND** diagnostics MUST 不记录 `loss/distillation`

#### Scenario: validation 和 evaluation 不使用 soft target
- **WHEN** 验证 DataLoader batch 包含 `target_beam_distribution`
- **THEN** validation/evaluation loss MUST 使用 hard `target_beam`
- **AND** validation/evaluation top-k/DBA 指标 MUST 继续使用 hard `target_beam`

## REMOVED Requirements

### Requirement: KD soft target 与 beam soft target 可共存但必须分离
**Reason**: KD runtime 删除后不存在 teacher-student distillation loss 与 beam soft target 共存场景。
**Migration**: Beam soft target 保留为 supervised beam smoothing。

#### Scenario: legacy KD 同时使用 beam soft target
- **WHEN** 用户运行旧 legacy KD 加 soft target 配置
- **THEN** 系统 MUST 拒绝 legacy KD 配置

### Requirement: 历史 KD 命名迁移
**Reason**: 新代码不再兼容读取 KD 命名的 soft label 字段作为源码配置契约。
**Migration**: 历史 artifact 只读；新数据和配置使用 beam soft target 命名。

#### Scenario: 旧 kd_soft_label 字段兼容读取
- **WHEN** 新训练配置或新 dataset 字段使用 `kd_soft_label`
- **THEN** 系统 MUST 拒绝或忽略该字段并提示使用 beam soft target 命名

