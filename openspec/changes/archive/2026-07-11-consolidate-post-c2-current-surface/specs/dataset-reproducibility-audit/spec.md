## REMOVED Requirements

### Requirement: Dataset reproducibility audit
**Reason**: 通用 dataset audit 扩展了独立产品面，但没有 current CLI 或 claim gate consumer。
**Migration**: 数据契约由各 dataset owner 的加载校验和 focused tests 直接维护。

#### Scenario: 不再提供通用 audit workflow
- **WHEN** 用户检查 current dataset tools
- **THEN** 系统 MUST NOT 暴露通用 dataset reproducibility audit 入口

### Requirement: Reproduction blocked status
**Reason**: blocked/local-substitute 聚合状态只服务已退役的通用 audit。
**Migration**: 仍在使用的 reproduction 状态由对应 claim 或 workflow owner 记录。

#### Scenario: owner 直接记录 blocked reason
- **WHEN** current workflow 缺少必需 artifact
- **THEN** 它 MUST 在自身 provenance 中记录 blocked reason，且 MUST NOT 依赖本 audit

### Requirement: Dataset audit outputs
**Reason**: 通用 audit JSON/Markdown/CSV 没有 current consumer。
**Migration**: 历史 audit 产物保留在 ignored outputs，不提升为 tracked contract。

#### Scenario: 不再生成通用 audit report
- **WHEN** current dataset workflow 运行
- **THEN** 系统 MUST NOT 要求 `outputs/analysis/dataset_audit/` 报告
