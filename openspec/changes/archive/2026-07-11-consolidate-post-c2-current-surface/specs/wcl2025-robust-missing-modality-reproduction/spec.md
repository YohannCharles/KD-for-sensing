## REMOVED Requirements

### Requirement: WCL 2025 source audit
**Reason**: WCL 2025 reproduction 没有 current official artifacts，也不应为了保留未完成应许而维护 source-audit 产品。
**Migration**: 历史 audit 与 blocked status 保留在 archive；不恢复 WCL current 路由。

#### Scenario: WCL audit 不再暴露
- **WHEN** 用户检查 current reproduction workflows
- **THEN** 系统 MUST NOT 提供 WCL 2025 source-audit 入口

### Requirement: Official 与 local-substitute 分支
**Reason**: official/local-substitute 分支只服务未落地的 WCL reproduction。
**Migration**: 已有 AMBER/AMR 与 current U-Mask/MMW/CSI 路径继续按自身 provenance 运行，不冒充 WCL substitute。

#### Scenario: current workflow 不声称 WCL reproduction
- **WHEN** current summary 生成
- **THEN** 它 MUST NOT 输出 WCL `official_reproduction` 或 `local_substitute` claim status

### Requirement: WCL 2025 missing-modality model
**Reason**: 专用 WCL model 扩展面无 current config、registry 或 claim consumer。
**Migration**: current missing-modality behavior 由 U-MaskBeamJEPA 及受保护 owner 继续管理。

#### Scenario: WCL model name 不再构建
- **WHEN** config 或 registry 请求 WCL 2025 local-substitute model
- **THEN** component construction MUST 拒绝该退役名称

### Requirement: WCL 2025 condition-level evaluation
**Reason**: WCL-specific condition matrix 和 strict-ranking adapter 随 reproduction 退役。
**Migration**: current missing-modality statistics/stress 依据现有 U-Mask/evaluation owner 契约保留。

#### Scenario: WCL rows 不进入 current ranking
- **WHEN** current condition-level summary 生成
- **THEN** 它 MUST NOT 依赖 WCL-specific row schema 或 comparability adapter

### Requirement: WCL 2025 产物边界
**Reason**: WCL 专用 outputs 契约随 workflow 一并退役。
**Migration**: 已有 `outputs/analysis/wcl2025_missing_modality_reproduction/` 内容保持 ignored，不移入 tracked source。

#### Scenario: 历史 WCL 产物不作为 current source
- **WHEN** consolidation 删除 WCL workflow
- **THEN** 实现 MUST NOT 提交历史 external source、checkpoint、logs 或 metrics
