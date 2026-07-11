## REMOVED Requirements

### Requirement: LOSO supporting 语义不绑定 engine dataloader facade
**Reason**: `kd_sensing.data.loso` 与旧 executor 均无 current consumer，capability 整项退出 current specs。
**Migration**: 当前 MMW 跨场景行为由 `mmw-cross-scene-adaptation-protocol` 管理。
#### Scenario: Legacy LOSO owner 删除
- **WHEN** consolidation 完成
- **THEN** current package MUST 不要求 DeepSense legacy LOSO helper

### Requirement: DeepSense6G 31-34 LOSO fold 定义
**Reason**: 无 current runtime consumer。
**Migration**: Historical fold 说明留 archive/docs。
#### Scenario: Historical fold 可追溯
- **WHEN** 维护者查询旧 fold
- **THEN** MUST 使用 archive/docs 而非 runtime helper

### Requirement: Target adapt/test split 防泄漏
**Reason**: 该要求绑定退役 DeepSense LOSO workflow。
**Migration**: Current MMW 防泄漏由 MMW protocol spec 约束。
#### Scenario: Current leakage guard 有 owner
- **WHEN** MMW adaptation 运行
- **THEN** MUST 使用 MMW current contract

### Requirement: Source multi-scene 数据加载
**Reason**: 退役 LOSO executor 无 current consumer。
**Migration**: Future workflow 需新 change。
#### Scenario: Legacy loader unavailable
- **WHEN** 用户请求 legacy LOSO source loader
- **THEN** current package MUST not provide it

### Requirement: Few-shot target label budget 采样
**Reason**: 绑定退役 LOSO workflow。
**Migration**: Current MMW sampling 留 MMW spec。
#### Scenario: Legacy sampler unavailable
- **WHEN** 请求 DeepSense LOSO budget sampler
- **THEN** current package MUST not require it

### Requirement: LOSO execute preflight
**Reason**: Executor 已退役。
**Migration**: Future executor 必须新建 capability。
#### Scenario: Preflight 不再存在
- **WHEN** 请求 legacy LOSO execute
- **THEN** command MUST not exist

### Requirement: 数据集无关的 LOSO fold 规划
**Reason**: 通用 planner 没有 current consumer，保留会形成 speculative API。
**Migration**: 每个未来 dataset workflow 自己定义 split contract。
#### Scenario: Generic planner 删除
- **WHEN** current code imports legacy planner
- **THEN** import MUST fail or be removed

### Requirement: Single-scene smoke is not LOSO
**Reason**: Current MMW spec 已拥有该 claim guard。
**Migration**: 使用 MMW protocol requirement。
#### Scenario: MMW claim 仍保守
- **WHEN** 只有 single-scene data
- **THEN** MMW workflow MUST not claim LOSO

### Requirement: MMW target adapt/test no leakage
**Reason**: 与 current MMW adaptation spec 重复。
**Migration**: 使用 `mmw-cross-scene-adaptation-protocol`。
#### Scenario: MMW split 使用 current owner
- **WHEN** MMW split 生成
- **THEN** current MMW spec MUST govern it

### Requirement: MMW few-shot sampling strategy
**Reason**: 与 current MMW spec 重复。
**Migration**: 使用 MMW sampling owner。
#### Scenario: MMW sampler 保留
- **WHEN** MMW few-shot 运行
- **THEN** current MMW focused tests MUST cover it

### Requirement: MMW LOSO summary claim guard
**Reason**: 与 current MMW protocol 重复。
**Migration**: 使用 MMW claim guard。
#### Scenario: MMW summary 有 current contract
- **WHEN** MMW summary 生成
- **THEN** MUST follow MMW spec

### Requirement: Radio-semantic few-shot sampling
**Reason**: 已由 MMW radio-semantic current specs 管理。
**Migration**: 使用 current MMW owner。
#### Scenario: Radio semantic 不依赖 legacy LOSO
- **WHEN** MMW sampling uses radio semantics
- **THEN** MUST not import `data.loso`

### Requirement: Radio-semantic target 防泄漏
**Reason**: 已由 current MMW protocol 管理。
**Migration**: 使用 MMW leakage contract。
#### Scenario: Leakage guard 保留
- **WHEN** radio-semantic adaptation runs
- **THEN** current MMW tests MUST cover leakage

### Requirement: Radio-semantic quick validation conclusion
**Reason**: 已由 MMW evidence owner 管理。
**Migration**: 使用 MMW summary/claim docs。
#### Scenario: Conclusion 不依赖 legacy helper
- **WHEN** MMW evidence is summarized
- **THEN** MUST not require `data.loso`

### Requirement: LOSO 不再绑定 Hist 默认矩阵
**Reason**: Hist 与 LOSO capability 均已退役，集中 summary 足以防回流。
**Migration**: 使用 retired-route-summary。
#### Scenario: Hist 不回流
- **WHEN** current surface scanned
- **THEN** Hist LOSO commands MUST not exist

### Requirement: HiST-Beam LOSO executor 退役边界
**Reason**: 专属墓碑折叠到集中 retired summary。
**Migration**: 使用 retired-route-summary 和 ordinary unknown-name errors。
#### Scenario: Executor remains absent
- **WHEN** user requests old Hist executor
- **THEN** command MUST not exist

### Requirement: LOSO stage dataset 构建边界
**Reason**: 仅服务已删除 executor。
**Migration**: Future executor must define its own stage lifecycle.
#### Scenario: Stage helper removed
- **WHEN** current package loads data owners
- **THEN** MUST not require legacy LOSO stage builder

### Requirement: LOSO helper 退役必须先完成 current consumer 审计
**Reason**: 审计已经完成并确认零 current consumer，条件已满足。
**Migration**: 删除 `kd_sensing.data.loso`，不保留 future-only implementation。
#### Scenario: Audit conclusion applied
- **WHEN** implementation wave reaches LOSO
- **THEN** helper and dedicated tests MUST be deleted

### Requirement: LOSO 退役不得破坏 split 可追溯性
**Reason**: Runtime capability 退出 current specs后，历史可追溯性由 archive/docs而非 current requirement 保留。
**Migration**: 保留 dated OpenSpec archive、git history 和必要 historical notes。
#### Scenario: Historical provenance remains
- **WHEN** old claim references LOSO split
- **THEN** docs MUST point to archive or recorded split description
