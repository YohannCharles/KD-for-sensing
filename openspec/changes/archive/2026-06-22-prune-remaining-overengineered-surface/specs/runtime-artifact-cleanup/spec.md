## ADDED Requirements

### Requirement: Cleanup 历史规则必须有安全用途
runtime cleanup 和 organize MUST 保留 dry-run manifest、保护边界、显式确认、路径重验证和 execution report。只服务历史研究线考古、且不影响安全删除或当前输出整理的细粒度 legacy 规则 MUST 删除或降为文档说明。

#### Scenario: 删除低价值历史输出规则
- **WHEN** 某条 cleanup rule 只匹配已退役研究线的旧命名，且不参与当前 dry-run 安全保护
- **THEN** 本 change MAY 删除该规则
- **AND** manifest MUST 继续保护 tracked files、dataset、source/config/docs/OpenSpec、active run、cache 和 checkpoint 高风险路径

#### Scenario: 保留必要 legacy archive 分类
- **WHEN** organize dry-run 扫描根级 legacy run、legacy numeric scene、legacy registry 或 legacy evaluation
- **THEN** manifest MUST 继续给出 protect/review/archive/move action
- **AND** 执行阶段 MUST 继续要求显式确认和状态重验证

### Requirement: Cleanup 不能替代源码瘦身
源码表面瘦身 MUST 不调用 runtime cleanup 删除本地产物。runtime cleanup 只在用户明确要求清理本地产物时运行，并必须产生 manifest。

#### Scenario: 源码 change 不运行删除阶段
- **WHEN** 实施本 change 的源码、测试、配置或文档删减
- **THEN** 实现 MUST 不调用 cleanup execution 删除 `outputs/`、`logs/`、cache、checkpoint 或数据
- **AND** 如需整理本地产物，必须作为单独用户确认流程执行
