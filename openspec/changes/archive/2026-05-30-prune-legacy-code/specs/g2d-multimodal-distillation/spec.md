## REMOVED Requirements

### Requirement: G2D distiller construction
**Reason**: G2D 多模态蒸馏研究线退役，继续保留 distiller 会扩大训练、配置、诊断和测试维护面。
**Migration**: 使用现有 `no_kd`、`logits_kd`、`rkd` 或当前保留的标准训练流程；不要再配置 `distillation.type: g2d`。

#### Scenario: G2D distiller 不再可构建
- **WHEN** 用户配置 `distillation.type: g2d`
- **THEN** 系统 MUST 拒绝构建该 distiller
- **AND** 错误信息 MUST 指出 G2D 已退役或该 distillation type 不可用

### Requirement: G2D future-only shape contract
**Reason**: 该 shape contract 只服务于 G2D loss、teacher confidence 和 diagnostics。
**Migration**: 普通 future-only 训练和评估继续由 `experiment-workflow` 的未来标签时隙对齐要求约束。

#### Scenario: G2D shape contract 不再作为独立能力
- **WHEN** 开发者查看蒸馏能力规格
- **THEN** 系统不再要求提供 G2D 专属 `[B,H,C]` shape 校验入口
- **AND** 普通模型输出对齐仍由通用训练/评估流程处理

### Requirement: G2D teacher ensemble
**Reason**: G2D teacher ensemble 只用于退役的多 teacher 蒸馏流程。
**Migration**: 单 teacher KD 使用现有 teacher checkpoint 解析；多 teacher ensemble 需要未来新 change 重新定义。

#### Scenario: G2D teacher ensemble 入口删除
- **WHEN** 代码或测试尝试导入 G2D teacher ensemble 构建函数
- **THEN** 导入 MUST 失败或该函数 MUST 不再存在
- **AND** 核心训练入口 MUST 不构建 G2D frozen teacher ensemble

### Requirement: G2D loss components
**Reason**: supervised CE、feature KD、logit KD 的 G2D 组合只服务于退役算法。
**Migration**: 使用普通 task loss、`logits_kd` 或 `rkd` distiller。

#### Scenario: G2D loss helper 删除
- **WHEN** 自动化测试扫描 `kd_sensing.distillation`
- **THEN** 系统 MUST 不要求存在 G2D loss helper 或 `G2DDistiller`
- **AND** 旧 G2D 正向 loss 测试 MUST 被删除

### Requirement: Teacher confidence and ranking
**Reason**: teacher confidence ranking 是 G2D/SMP 的内部诊断，不再是项目保留能力。
**Migration**: 需要 teacher 置信度分析时应新建独立诊断 capability。

#### Scenario: teacher ranking 不再输出
- **WHEN** 普通训练完成一个 epoch
- **THEN** 系统 MUST 不要求生成 G2D teacher confidence 或 weak-to-strong ranking

### Requirement: Sequential Modality Prioritization
**Reason**: SMP 梯度屏蔽与 G2D-global 绑定，随 G2D 一并退役。
**Migration**: 后续如需梯度调度方法，必须重新提出训练方法 change。

#### Scenario: SMP 不再参与训练
- **WHEN** 训练流程执行 optimizer step
- **THEN** 系统 MUST 不调用 G2D SMP gradient mask
- **AND** `g2d_smp` 专用测试 MUST 不再作为验收要求

### Requirement: G2D diagnostics artifact
**Reason**: `diagnostics/g2d_epoch_<epoch>.json` 是退役算法的专属运行产物。
**Migration**: 普通训练继续写出 `train_log.json`、`metrics.json` 和当前保留能力的 diagnostics。

#### Scenario: G2D epoch diagnostics 不再生成
- **WHEN** 用户运行当前保留的训练配置
- **THEN** 系统 MUST 不要求写出 `diagnostics/g2d_epoch_<epoch>.json`
- **AND** run metadata MUST 不把缺少 G2D diagnostics 视为失败

### Requirement: G2D 支持包含 CSI 的模态集合
**Reason**: 该要求扩展的是已退役 G2D teacher ensemble、ranking、SMP 和 diagnostics。
**Migration**: CSI 模态自身保留；CSI 参与普通训练、评估或 fusion 的能力由 CSI 和 fusion 相关 specs 约束。

#### Scenario: CSI 不再触发 G2D 分支
- **WHEN** 用户运行包含 CSI 的当前保留配置
- **THEN** 系统 MUST 不构建 G2D teacher ensemble
- **AND** 配置中出现 `distillation.type: g2d` 时 MUST 失败
