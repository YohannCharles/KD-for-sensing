## MODIFIED Requirements

### Requirement: JEPA context encoder 下游复用
系统 MUST 提供 supervised fusion 可配置使用的 JEPA context image encoder 初始化入口。该入口 MUST 从 JEPA checkpoint 中抽取 `context_encoder` 权重，MUST 兼容 objective checkpoint 和恢复 checkpoint。Current downstream 只承诺 patch-token mean pooling 输出帧级 `[B,T,D]` 特征；系统 MUST 不要求 GPS-query、content-query、hybrid、predictive、K-token 或 attention-diagnostics pooler 存在。

#### Scenario: 从 JEPA best checkpoint 初始化 mean image encoder
- **WHEN** supervised fusion 配置将 image encoder 设置为 `jepa_context_image` 且 checkpoint 指向 JEPA `best.pth`
- **THEN** 系统 MUST 加载 `context_encoder.*` 权重
- **AND** forward MUST 将 `[B,T,3,H,W]` 转为 `[B,T,D]` mean-pooled 特征

#### Scenario: 从 JEPA last checkpoint 初始化 mean image encoder
- **WHEN** checkpoint payload 通过 `state_dict` 保存恢复状态
- **THEN** 系统 MUST 抽取 `context_encoder.*` 权重
- **AND** 输出维度 MUST 与配置 `output_dim` 一致

#### Scenario: 退役 pooler 被拒绝
- **WHEN** 配置请求 GPS-query、predictive、hybrid、query-weighted 或 K-token downstream pooler
- **THEN** 配置/组件构建 MUST 失败并列出 current mean 路径
- **AND** 系统 MUST 不静默映射到 mean pooling

#### Scenario: MMW mean reuse 保持可用
- **WHEN** current MMW/JEPA config 使用 `jepa_context_image` 和 `pooling: mean`
- **THEN** 配置加载和 model construction MUST 成功
- **AND** 不得要求 retired GPS-query helper、evidence 或 diagnostics
