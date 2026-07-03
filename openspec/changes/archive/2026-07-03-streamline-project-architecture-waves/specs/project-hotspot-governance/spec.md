## ADDED Requirements

### Requirement: Architecture streamlining waves are mandatory implementation units
项目 MUST 将本 change 的全仓重构按可独立验证的 remediation waves 实施。每个 wave MUST 记录目标 owner、当前热点规模或问题证据、公开 surface 保持策略、focused validation commands、rollback 条件和不允许混入的无关变更。

#### Scenario: Wave 开始前声明目标
- **WHEN** 开发者开始 dataset、runtime、model forward、diagnostics、config/script/import surface 或 OpenSpec/docs guardrail wave
- **THEN** tasks 或实现说明 MUST 命名目标文件/owner、改动类型、public CLI/import/config 策略和最小 focused tests
- **AND** 若已有测试红点或工作树噪声，说明 MUST 区分既有状态和本 wave 引入的状态

#### Scenario: Wave 完成后独立验收
- **WHEN** 一个 wave 完成源码移动、拆分、删除或合并
- **THEN** 开发者 MUST 运行该 wave 对应 focused tests 和 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- **AND** 未运行的验证 MUST 在最终说明中记录原因、替代验证和剩余风险

### Requirement: Hotspot reductions must follow stable responsibility boundaries
热点模块重构 MUST 按稳定职责边界拆分或合并，不得只按行数机械切割。业务 owner 可保留 accepted-size 例外，但继续增长的 dataset、trainer、evaluation pass、model forward、diagnostic runner、manifest builder 和 local experiment surface MUST 拆到职责明确的窄模块或登记有验证命令支撑的暂缓理由。

#### Scenario: 拆分不产生新的私有聚合层
- **WHEN** 一个热点函数或类被拆分
- **THEN** 新 helper MUST 对应真实职责，例如 resource reader、target provider、run context、metric aggregation、artifact writer 或 forward stage
- **AND** 系统 MUST 不新增只搬运杂项 helper 的跨领域 `helpers.py`、私有聚合 facade 或兼容 re-export 层

#### Scenario: 合并低价值边界
- **WHEN** 一个 helper 文件只有单调用点、无独立 public contract、只服务 re-export 或只为降低行数存在
- **THEN** 本 change MAY 将其合并回清晰 owner
- **AND** 合并后 MUST 不恢复旧入口、旧 import alias 或跨领域工具聚合

### Requirement: Wave rollback must not restore retired routes
任一 wave 回滚 MUST 回到上一稳定 owner 布局或暂缓该 wave，不得通过恢复旧 CLI、旧 config、旧 registry name、退役研究线 facade 或 root-level wrapper 作为长期修复。

#### Scenario: 回滚保持退役边界
- **WHEN** 某个 wave 验证失败需要回滚
- **THEN** 回滚操作 MUST 不重新引入 HiST/Hist、KD、BGAM、viewer manifest、Raymobtime、AMR-Net_gps_image、JEPA-MSAC 或其它 retired route 的 current entry
- **AND** 若临时保留 compatibility shim，必须在 tasks 中记录删除条件和最短保留范围

