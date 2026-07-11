## MODIFIED Requirements

### Requirement: RBMA workflow extension path
U-MaskBeamJEPA MUST 继续以内嵌 opt-in component 形式支持 `reliability_biased_missing_attention`、beam prototype alignment、full-to-partial teacher stabilization 和 pattern-balanced mask。系统 MUST 不恢复旧独立 RBMA/prototype-KD sweep owner，也 MUST 不新增第二个完整模型注册名表达同一行为。

#### Scenario: Current U-Mask RBMA branch 保留
- **WHEN** `u_mask_beam_jepa` 配置启用 current RBMA/prototype/teacher option
- **THEN** 系统 MUST 复用现有 U-Mask owner 与 training extension
- **AND** 普通 baseline MUST 不要求这些 diagnostics 或 metadata

#### Scenario: 独立 RBMA workflow 不恢复
- **WHEN** 用户请求旧 RBMA sweep config、runbook 或独立 model owner
- **THEN** current surface MUST 拒绝或标记 retired

### Requirement: RBMA workflow metadata
U-Mask 内嵌 RBMA/prototype/teacher option MUST 写出可审计 training metadata，覆盖 fusion type、mask sampler、prototype alignment、teacher stabilization、JEPA loss 状态、reliability metadata consumption 和 ablation id。旧独立 workflow metadata schema MUST 不作为额外 owner保留。

#### Scenario: Current metadata 最小字段
- **WHEN** U-Mask current RBMA/prototype/teacher option 被构建
- **THEN** metadata MUST 记录 `fusion_type`、`mask_sampler`、`use_jepa_loss`、`use_beam_prototype_alignment`、`use_full_to_partial_kd` 和 reliability consumption
- **AND** checkpoint teacher 未实现时 MUST 记录 pending/disabled，而不是成功启用

### Requirement: optimizer 参数组构建位于 optim 模块
训练引擎 MUST 将 current encoder/core/head 参数组解析、pattern 匹配、重复检测、未匹配处理和 summary 维护在 `kd_sensing.engine.optim` 或等价窄模块中。训练主循环 MUST 只消费构建好的 optimizer 和 summary；它 MUST 不保留 retired JEPA query pooler/adapter 专属组。

#### Scenario: Current 参数组不触碰 trainer
- **WHEN** 开发者调整 current JEPA mean-context、U-Mask、MMW/CSI encoder/core/head 参数组
- **THEN** 主要变更 MUST 限定在 optimizer owner及测试
- **AND** 不需要编辑 trainer epoch/batch loop

#### Scenario: 参数组 summary 保持
- **WHEN** current training 使用多个参数组
- **THEN** logs/TensorBoard MUST 能记录 learning rate 与参数数
- **AND** 未声明时保留单 main group

### Requirement: runtime metadata 收集位于 run metadata 模块
Current model/runtime structure metadata MUST 由 `engine.run_metadata`、artifact writer 或等价窄模块收集。JEPA mean-context reuse MAY 记录 checkpoint、freeze、pooling mean 和参数组；系统 MUST 不要求 query pooler、adapter、attention 或 predictive metadata。

#### Scenario: Current model metadata 被聚合
- **WHEN** current model/submodule 提供只读 training metadata
- **THEN** run metadata owner MUST 写入正式 artifact
- **AND** mean-context metadata MUST 不包含 retired query fields

#### Scenario: Config fallback 只处理 current fields
- **WHEN** 构建前从 config 生成 metadata
- **THEN** fallback MUST 只解析 current model/config fields
- **AND** 不恢复 retired pooler/adapter schema

## REMOVED Requirements

### Requirement: JEPA downstream 扩展实现边界
**Reason**: 通用 downstream pooler/adapter capability 整项退出，current 只保留 pretraining 与 mean-context reuse。
**Migration**: Future downstream extension 需要新 OpenSpec change；当前 owner 不预留 registry/framework。

#### Scenario: Downstream 扩展点退出
- **WHEN** current model architecture 被检查
- **THEN** 它 MUST 不要求新增/维护 query pooler 或 adapter extension surface

### Requirement: Geometry-prior route classification
**Reason**: Geometry-prior component 已无 current config、CLI、claim 或 source consumer，继续规定未来实现路径只会维持 speculative architecture surface。
**Migration**: 新 fusion component 仍遵守通用“模块化组件优先”要求；如未来重启 geometry prior，必须新建 OpenSpec change。

#### Scenario: Geometry route 不再预留
- **WHEN** post-C2 cleanup 完成
- **THEN** current extension contract MUST 不要求 geometry-prior component 或 whole-model exception 路径
- **AND** registry MUST 不保留 geometry 专属扩展点

### Requirement: BEV-Fusion reproduction boundary
**Reason**: BEV-Fusion 2604 已退役，当前仓库没有 reproduction owner；保留 BEV-lite 例外是未来假设。
**Migration**: 历史分类留在 archive/retired summary；任何新论文复现使用通用 workflow/paper reproduction 规则并单独提案。

#### Scenario: BEV 专属边界退出
- **WHEN** 开发者检查 current architecture contract
- **THEN** contract MUST 不预留 BEV-lite 或完整 BEV-Fusion 专属分支
- **AND** retired route MUST 不通过通用规则隐式恢复

### Requirement: Geometry-prior training metadata
**Reason**: 被描述的 baseline 整体删除，专属 metadata schema 没有 producer 或 consumer。
**Migration**: Current components 继续使用通用 model/component training metadata contract。

#### Scenario: Geometry metadata 不再要求
- **WHEN** current model metadata 被验证
- **THEN** validation MUST 不要求 geometry prior、geometry fusion 或 geometry teacher 字段
- **AND** current MMW direct geometry metadata MUST 继续由其 MMW owner 管理
