## ADDED Requirements

### Requirement: U-MaskBeamJEPA 可选挂载 BCACL companion
UMaskBeamJEPA MUST 仅在显式启用 BCACL 时挂载独立 companion module，并输出 Phase 1 所需的投影特征、私有 logits 和共享 logits。现有 `prototype_bank` MUST 继续独占融合恢复 Beam 原型职责，不得被模态原型替换或共用。

#### Scenario: 默认 T2 构建
- **WHEN** canonical T2 配置未启用 BCACL
- **THEN** 模型 MUST 不包含 BCACL 参数或 buffer
- **AND** 现有融合、prototype、Router 与输出 schema MUST 保持不变

#### Scenario: Phase 1 构建
- **WHEN** `bcacl.enabled=true` 且 stage 为 phase1
- **THEN** 模型 MUST 训练四个编码器、独立投影和启用的单模态头
- **AND** 最终融合与融合恢复原型 MUST 不从 Phase 1 总损失获得梯度

### Requirement: Phase 2 在 optimizer 构建前冻结 BCACL encoder 路径
Phase 2 模型 MUST 在 optimizer 构建前冻结四个编码器和按配置冻结的 BCACL 投影/头，并保持这些模块的 BatchNorm/Dropout 为 eval；现有融合和融合恢复原型 MUST 继续按原有 loss 训练。

#### Scenario: Phase 2 optimizer step
- **WHEN** Phase 2 从 Phase 1 checkpoint 初始化并完成一个 optimizer step
- **THEN** 冻结的编码器和 BCACL 参数以及 running statistics MUST 不变
- **AND** 至少一个现有融合或融合恢复原型参数 MUST 可训练
