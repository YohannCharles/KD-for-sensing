## ADDED Requirements

### Requirement: 模型表面重构不得改变 registry 扩展契约
Model/config/loss 重构 MUST 保持 registry build 行为、component baseline 路径、whole-model exception policy 和 architecture summary 只读语义。

#### Scenario: registry current 名称保持稳定
- **WHEN** model owner modules are split or helper files are moved
- **THEN** current registry names and removed-name guards MUST behave as before
- **AND** new whole-model exceptions MUST NOT be introduced without a separate current spec or active change
