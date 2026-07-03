# dataset-reproducibility-audit Specification

## Purpose
TBD - created by archiving change add-paper-export-dataset-audit-literature. Update Purpose after archive.
## Requirements
### Requirement: Dataset reproducibility audit
系统 MUST 提供只读 dataset reproducibility audit，用于检查本地数据目录、CSV 字段、模态文件引用、label range、split metadata、official/local reproduction status 和 blocked reasons。Audit MUST 不移动、删除、复制或重写真实数据。

#### Scenario: 检查 DeepSense6G/BeamBench CSV
- **WHEN** 用户对 DeepSense6G 或 BeamBench CSV 运行 audit
- **THEN** 系统 MUST 报告 camera、radar、LiDAR、GPS、BS GPS、label、scene id、sample id、sequence id 和 timestamp 字段是否存在或可由别名解析
- **AND** 缺字段 MUST 标记 warning 或 blocked reason

#### Scenario: 检查模态文件引用
- **WHEN** CSV 行包含模态文件路径
- **THEN** audit MUST 按 data root 解析路径并统计存在数量、缺失数量、缺失比例和示例缺失路径
- **AND** audit MUST 不创建、删除或移动任何被引用文件

#### Scenario: 检查 label range 和 beam shift
- **WHEN** CSV 或 label 文件提供 beam target
- **THEN** audit MUST 检查 label 最小值、最大值、非法 label 数量、0-based/1-based 或 beam shift 配置
- **AND** 不同 label space MUST 在报告中清晰标记

#### Scenario: 检查 split leakage metadata
- **WHEN** 数据目录或 run artifact 包含 split metadata
- **THEN** audit SHOULD 检查 train/validation/test 样本 id、sequence id 或 group id 是否存在重叠
- **AND** 无法检查时 MUST 标记 split_leakage_check unavailable 而不是静默通过

### Requirement: Reproduction blocked status
Audit MUST 能输出 official reproduction blocked/local substitute 状态，供 README_REPRODUCE、claim registry 和 paper export 使用。

#### Scenario: official artifact 缺失
- **WHEN** official data、official weights、official source/config 或 official environment 任一缺失
- **THEN** audit MUST 标记 official reproduction 为 blocked 或 incomplete
- **AND** 报告 MUST 记录缺失项和建议下一步

#### Scenario: local substitute 可运行
- **WHEN** 本仓库 local substitute 所需数据、配置和 checkpoint provenance 满足要求
- **THEN** audit MAY 标记 local substitute ready
- **AND** 报告 MUST 明确该状态不等同 official reproduction

### Requirement: Dataset audit outputs
Dataset audit MUST 输出机器可读 JSON，并 MAY 输出 Markdown/CSV 摘要。默认输出 MUST 位于 ignored `outputs/analysis/dataset_audit/` 或用户显式指定目录。

#### Scenario: 写出 audit report
- **WHEN** audit 完成
- **THEN** 输出 MUST 包含 dataset family、data root、CSV path、scene scope、field summary、file reference summary、label summary、split summary、blocked status 和 warnings
- **AND** 输出 MUST 不包含真实数据内容的大段复制

