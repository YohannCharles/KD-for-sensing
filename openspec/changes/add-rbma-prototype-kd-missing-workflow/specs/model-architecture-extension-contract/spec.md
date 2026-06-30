## ADDED Requirements

### Requirement: RBMA workflow extension path
RBMA、beam prototype alignment、full-to-partial teacher stabilization 和 pattern-balanced mask MUST 作为 U-MaskBeamJEPA opt-in 增强实现。除非后续 design 证明现有 whole-model exception 无法承载，系统 MUST 不新增第二个完整模型注册名来表达同一 workflow。

#### Scenario: 不新增重复 whole-model
- **WHEN** 实现 RBMA prototype KD workflow
- **THEN** 系统 MUST 复用 `u_mask_beam_jepa` 或现有 current owner
- **AND** 系统 MUST 不新增与 U-MaskBeamJEPA 语义重复的完整 `MODELS.register(...)` 名称

#### Scenario: 普通 baseline 不消费新增 metadata
- **WHEN** 普通 supervised、AMBER full local 或非 U-MaskBeamJEPA baseline 运行
- **THEN** reliability、prototype、full-to-partial teacher 和 pattern diagnostics MUST 不是必需 forward 输入
- **AND** 这些 baseline 的 metadata MUST 能声明未消费该 workflow metadata

### Requirement: RBMA workflow metadata
RBMA workflow MUST 写出可审计训练策略 metadata，覆盖 fusion type、mask sampler、prototype alignment、teacher stabilization、JEPA loss 状态、reliability metadata consumption 和 ablation id。

#### Scenario: metadata 最小字段
- **WHEN** RBMA workflow 模型或训练 run 被构建
- **THEN** metadata MUST 包含 model type、enabled modalities、fusion type、mask sampler、use_jepa_loss、use_beam_prototype_alignment、use_full_to_partial_kd 和 reliability metadata consumption
- **AND** 缺少这些字段 MUST 被 focused tests 或 architecture summary tests 捕获

#### Scenario: checkpoint teacher 状态可审计
- **WHEN** config 声明 `kd_teacher_mode`
- **THEN** metadata MUST 记录 teacher mode、teacher checkpoint provenance 或 pending reason
- **AND** checkpoint teacher 未实现时 MUST 不被记录为已启用成功
