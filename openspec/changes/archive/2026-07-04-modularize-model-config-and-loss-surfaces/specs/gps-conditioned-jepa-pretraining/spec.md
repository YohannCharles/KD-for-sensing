## ADDED Requirements

### Requirement: JEPA pretraining helper 拆分必须保持 checkpoint 与 token 语义
GPS-conditioned JEPA pretraining 重构 MUST 在移动 helper 到窄模块时保持 tokenizer、context/target encoder、GPS conditioning、checkpoint reuse 和 training metadata 语义。

#### Scenario: checkpoint reuse 兼容
- **WHEN** JEPA checkpoint loading or encoder extraction helper is moved
- **THEN** missing/unexpected key reporting, strictness behavior and runtime metadata MUST remain compatible
