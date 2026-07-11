## REMOVED Requirements

### Requirement: RBMA/KD workflow remains retired
**Reason**: 独立 RBMA/prototype-KD missing-modality workflow 的配置、runbook、sweep 和 claim surface 已由 post-C2 主线取代；其防回流语义改由集中 retired-route guard 承接，不再保留专用 tombstone capability。
**Migration**: 只退役独立 workflow。U-MaskBeamJEPA 内嵌且仍受 current owner 保护的 `reliability_biased_missing_attention`、prototype 分支和 full-to-partial teacher 机制 MUST 保留，并继续按 U-Mask/final C2 specs 与 focused tests 维护；旧 RBMA/prototype-KD configs、scripts、runbooks 和 claim provenance 从 archive/git 查询。

#### Scenario: 独立 RBMA workflow 退出但 U-Mask 内嵌机制保留
- **WHEN** current configs、scripts、registry 和 model branches 被审计
- **THEN** 系统 MUST 不恢复独立 RBMA/prototype-KD/BTAPA/weakKD workflow、旧 sweep YAML 或同名 wrapper
- **AND** U-MaskBeamJEPA 内嵌的 `reliability_biased_missing_attention`、prototype 和 full-to-partial teacher MUST 不因 tombstone 折叠被删除、重命名或降级
