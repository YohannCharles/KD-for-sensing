## ADDED Requirements

### Requirement: G2D 支持包含 CSI 的模态集合
G2D teacher ensemble、teacher confidence、ranking、SMP gradient masking 和 diagnostics MUST support any configured modality subset that is valid in the project modality registry, including subsets that contain `csi`. Existing five-modality G2D configs MUST remain valid.

#### Scenario: 构建 GPS+CSI G2D teacher ensemble
- **WHEN** G2D 配置的模态集合为 `gps` 和 `csi`
- **THEN** 系统 MUST 构建 `gps` teacher 和 `csi` teacher
- **AND** 每个 teacher MUST 使用对应单模态输入前向
- **AND** teacher checkpoint 缺失时错误信息 MUST 包含缺失的模态名

#### Scenario: CSI teacher confidence 参与排序
- **WHEN** G2D teacher logits 包含 `gps` 和 `csi` 的 `[B,H,C]` 输出且 labels 为 `[B,H]`
- **THEN** 系统 MUST 计算 `gps` 和 `csi` 对真实 label 的 teacher confidence
- **AND** weak-to-strong ranking MUST 包含 `csi`

#### Scenario: SMP 可以激活 CSI
- **WHEN** SMP 调度器将 active modalities 设置为 `["csi"]`
- **THEN** 系统 MUST 保留 CSI encoder、fusion module 和 prediction head 的梯度
- **AND** 系统 MUST 清零 inactive modality encoder 的梯度

#### Scenario: G2D diagnostics 记录 CSI
- **WHEN** G2D epoch diagnostics 写出 JSON
- **THEN** diagnostics MUST 在 teacher confidence、ranking 和 active modalities 中使用真实配置模态名
- **AND** 当配置包含 `csi` 时 diagnostics MUST 能记录 `csi` 项
