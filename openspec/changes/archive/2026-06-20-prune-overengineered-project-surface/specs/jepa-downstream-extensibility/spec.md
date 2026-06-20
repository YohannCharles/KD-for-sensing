## MODIFIED Requirements

### Requirement: JEPA downstream pooler 和 adapter 可插拔
系统 MUST 为 JEPA context image encoder 的下游 supervised reuse 提供可插拔 pooler 边界，并 MAY 在存在非 identity 实现时提供 adapter 边界。pooler MUST 消费 JEPA context encoder 输出的 patch tokens `[B,T,N,D]`，并默认输出现有 fusion projector 可消费的 `[B,T,D]` image feature。identity adapter MAY 被实现为内联 no-op，而不是独立注册组件；任何非 identity adapter MUST 不修改 Stage 1 JEPA checkpoint schema、target encoder EMA、mask sampler 或 latent prediction loss。

#### Scenario: 默认 mean pooler 兼容
- **WHEN** 用户配置 `jepa_context_image` 且未显式声明 pooler 或继续使用 `pooling: mean`
- **THEN** 系统 MUST 使用 mean pooling 生成 `[B,T,D]` image feature
- **AND** 现有 `fair_gps_biased` mean-pooling 配置 MUST 无需修改即可构建和 forward

#### Scenario: GPS-query pooler 通过配置构建
- **WHEN** 用户配置 `jepa_context_image` 的 pooler 为 GPS-query attention
- **THEN** 系统 MUST 构建对应 pooler 并将 JEPA patch tokens 与同 batch/time 的 GPS 条件特征传入 pooler
- **AND** pooler 输出 MUST 默认保持 `[B,T,D]`
- **AND** 系统 MUST 不要求 JEPA target encoder、EMA 更新或 JEPA latent loss 参与 supervised downstream 训练

#### Scenario: identity adapter 为无操作路径
- **WHEN** 用户未配置 JEPA downstream adapter 或配置 adapter 为 `identity`
- **THEN** 系统 MUST 保持现有 image feature shape 和 downstream 输出契约
- **AND** 系统 MAY 通过内联 no-op 而不是 adapter registry 完成该行为

#### Scenario: 非 identity adapter 不改变训练主输出契约
- **WHEN** 用户为 JEPA downstream image encoder 配置非 identity adapter
- **THEN** adapter 输出 MUST 继续被转换为现有 model output 可消费的 image feature
- **AND** `ModelOutput` 适配、beam loss、beam metrics 和 checkpoint workflow MUST 无需新增 JEPA 专用分支
