## MODIFIED Requirements

### Requirement: MMW split leakage diagnostics
MMW Town10 split metadata MUST 包含可机器读取的泄漏诊断，用于判断当前 train/test CSV 是否可作为 strict validation 协议。诊断 MUST 至少覆盖 train/test frame overlap、test window 与 train window 的最大 frame overlap、相邻窗口跨 split 比例和未来标签序列复用比例。未来标签序列复用 MUST 作为标签分布诊断保留；当 `pred_len=1` 时，beam 类别重复本身 MUST NOT 被解释为 frame、window 或 trajectory 泄漏，也 MUST NOT 单独使 split strict-ineligible。

#### Scenario: group-safe split 诊断通过
- **WHEN** 系统使用默认 group-safe 协议生成 split
- **THEN** leakage diagnostics MUST 记录 train/test frame overlap count 为 0
- **AND** test window 与任一 train window 的最大 frame overlap MUST 小于完整窗口长度
- **AND** summary MUST 包含 guard band frames、window length、train/test window counts 和 diagnostics 生成时间或版本
- **AND** P1 future label class 在 train/test 重复时 MUST 继续报告 reuse ratio，但 strict eligibility MUST 由结构性 overlap diagnostics 决定

#### Scenario: 诊断发现高重叠
- **WHEN** leakage diagnostics 发现 test window 与 train window 共享完整或近完整历史+未来上下文
- **THEN** split metadata MUST 标记 `strict_validation_eligible=false`
- **AND** metadata MUST 包含超阈值统计和可执行修复提示
- **AND** 训练或评估产物消费该 metadata 时 MUST 能显示该 split 不适合作为 strict 主结论

## ADDED Requirements

### Requirement: H5/P1 metadata variants participate in readiness
MMW preparation availability writer MUST 识别同一 prepared scenario 下由显式 split tag 生成的 `metadata_<tag>.json` 与 `sanity_report_<tag>.json`，并 MUST 根据其中 manifest、window count、split eligibility 和 artifact path 判定 readiness，而不是只检查无后缀文件。

#### Scenario: rainy H5/P1 artifacts 已完整
- **WHEN** rainy scenario 具有 `metadata_h5p1.json`、`sanity_report_h5p1.json`、有效 manifest 和 strict split
- **THEN** condition availability MUST 将该 scenario 标记为可供对应 H5/P1 protocol 使用
- **AND** availability MUST 记录实际 metadata/sanity 路径和 split tag

