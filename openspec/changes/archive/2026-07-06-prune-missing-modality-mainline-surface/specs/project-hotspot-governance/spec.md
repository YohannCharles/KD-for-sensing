## ADDED Requirements

### Requirement: 缺失模态主线热点必须登记或右尺寸化
缺失模态主线清理中发现的未登记大 owner MUST 被登记为 accepted、monitor、split-next、merge-candidate 或 deleted。登记 MUST 说明职责边界、当前规模风险、保留或拆分理由、禁止回流路径和验证命令。属于当前主线证据链的热点 MUST 不因体积大而被直接删除。

#### Scenario: 未登记大 owner 被处置
- **WHEN** `project_surface_doctor` 或等价热点检查报告 `gps_query_evidence.py`、`run_metadata.py`、`u_mask_beam_jepa.py` 或后续同类大 owner 未登记
- **THEN** implementation MUST 在 hotspot inventory 或等价文档中登记处置状态，或完成职责明确的拆分/合并
- **AND** 最终检查 MUST 不再把这些 owner 报告为未分类热点

#### Scenario: 当前 U-Mask 主线不被误删
- **WHEN** 热点候选承载 U-MaskBeamJEPA 当前模型、loss、masking、missing-modality robustness 或 current evidence metadata
- **THEN** implementation MUST 保留其公共训练/评估语义
- **AND** 任何 branch 删除 MUST 先证明对应 config、OpenSpec specs、tests 和 claim provenance 不再消费该 branch

#### Scenario: 接受热点必须有 headroom
- **WHEN** implementation 选择暂不拆分某个大 owner
- **THEN** inventory MUST 记录接受原因、headroom 或后续 split-next 条件
- **AND** 后续新增 suite、analysis family 或 model branch MUST 先复核该热点预算

### Requirement: 清理 wave 不得扩大重复本地工作流热点
缺失模态主线清理 MUST 减少或稳定重复本地工作流代码，不得通过新增同构 generator、runner、paper helper 或 facade 来扩大热点。共享逻辑 MUST 收敛到一个明确 owner；仅服务单一历史场景的逻辑 MUST 删除或登记 local/manual lifecycle。

#### Scenario: 重复 ablation generator 不扩大
- **WHEN** 新增或修改 Scene31-34 encoder ablation 支持
- **THEN** 主要实现 MUST 位于统一 encoder ablation owner 或共享 helper 中
- **AND** 项目 MUST 不保留多个仅 encoder family 名称不同、主体逻辑同构的 generator

#### Scenario: 报告 helper 不形成新聚合层
- **WHEN** paper table、presentation artifact 或 final conclusion helper 被保留
- **THEN** implementation MUST 不新增跨报告的低价值聚合 facade
- **AND** 共享逻辑只有在两个以上 current 调用点需要时才可抽为窄 helper，否则 MUST 留在清晰 owner 内

### Requirement: 热点清理必须与行为重构隔离
缺失模态主线热点治理 MAY 登记、拆分或合并源码 owner，但 MUST 不把训练数学语义、数据 split 语义、label-space、指标口径、checkpoint schema 或默认输出目录作为隐式变更。若需要改变这些行为，MUST 另开专门 OpenSpec change。

#### Scenario: 拆分不改变公开语义
- **WHEN** implementation 拆分 `u_mask_beam_jepa.py`、`run_metadata.py`、`gps_query_evidence.py` 或相关 helper
- **THEN** 公开导入、CLI payload、配置字段、日志字段和输出路径语义 MUST 保持兼容
- **AND** focused tests MUST 覆盖正常路径、错误路径和至少一个当前主线配置加载或 dry-run 场景

#### Scenario: 删除只作用于非行为表面
- **WHEN** implementation 删除历史 helper、重复 runner 或可再生成配置
- **THEN** 删除 MUST 不改变 current model forward、loss 计算、metric comparability 或 evaluation aggregation
- **AND** 验证说明 MUST 区分表面收缩与任何未执行的行为验证风险
