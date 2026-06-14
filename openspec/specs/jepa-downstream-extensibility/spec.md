# jepa-downstream-extensibility Specification

## Purpose
定义 JEPA context image encoder 在 supervised downstream 中的可插拔 pooler、adapter、optimizer 参数组和 runtime metadata 契约，保证派生实验能与 GPS-biased mean-pooling baseline 可比，同时不改变 Stage 1 JEPA checkpoint 或主训练评估流程。
## Requirements
### Requirement: JEPA downstream pooler 和 adapter 可插拔
系统 MUST 为 JEPA context image encoder 的下游 supervised reuse 提供可插拔 pooler/adapter 边界。pooler MUST 消费 JEPA context encoder 输出的 patch tokens `[B,T,N,D]`，并默认输出现有 fusion projector 可消费的 `[B,T,D]` image feature。adapter MAY 在 pooling 前后执行轻量可训练变换，但 MUST 不修改 Stage 1 JEPA checkpoint schema、target encoder EMA、mask sampler 或 latent prediction loss。

#### Scenario: 默认 mean pooler 兼容
- **WHEN** 用户配置 `jepa_context_image` 且未显式声明 pooler 或继续使用 `pooling: mean`
- **THEN** 系统 MUST 使用 mean pooling 生成 `[B,T,D]` image feature
- **AND** 现有 `fair_gps_biased` mean-pooling 配置 MUST 无需修改即可构建和 forward

#### Scenario: GPS-query pooler 通过配置构建
- **WHEN** 用户配置 `jepa_context_image` 的 pooler 为 GPS-query attention
- **THEN** 系统 MUST 构建对应 pooler 并将 JEPA patch tokens 与同 batch/time 的 GPS 条件特征传入 pooler
- **AND** pooler 输出 MUST 默认保持 `[B,T,D]`
- **AND** 系统 MUST 不要求 JEPA target encoder、EMA 更新或 JEPA latent loss 参与 supervised downstream 训练

#### Scenario: adapter 不改变训练主输出契约
- **WHEN** 用户为 JEPA downstream image encoder 配置 adapter
- **THEN** adapter 输出 MUST 继续被转换为现有 model output 可消费的 image feature
- **AND** `ModelOutput` 适配、beam loss、beam metrics 和 checkpoint workflow MUST 无需新增 JEPA 专用分支

### Requirement: JEPA downstream 派生实验保持 baseline 可比
系统 MUST 保留现有 `fair_gps_biased` mean-pooling 配置作为主 baseline。新增 JEPA downstream pooler、adapter、冻结策略或参数组实验 MUST 通过派生配置表达，并 MUST 记录与 baseline 的差异变量。

#### Scenario: 派生配置只覆盖实验变量
- **WHEN** 用户加载基于 `fair_gps_biased` 的 JEPA downstream 派生配置
- **THEN** 配置 MUST 复用匹配口径的 GPS-biased 多场景 JEPA checkpoint、Image+GPS 模态、beam objective、label space 和 split protocol
- **AND** 配置 MUST 只显式覆盖 pooler/adapter、冻结策略、optimizer 参数组、run name 或 ablation metadata

#### Scenario: 不替换 baseline
- **WHEN** 新增 JEPA downstream 派生配置
- **THEN** 系统 MUST 不删除、重命名或语义替换现有 `fair_gps_biased` mean-pooling baseline 配置
- **AND** README 或实验说明 MUST 指出派生配置应与匹配 baseline 成对比较

### Requirement: JEPA downstream 参数组优化
系统 MUST 支持通过配置为 JEPA downstream supervised fusion 指定 optimizer 参数组。参数组 MUST 能区分 JEPA context encoder、JEPA pooler/adapter、GPS encoder/projector、representation core 和 task head 等模块，并 MUST 记录每组学习率和参数数量。

#### Scenario: 构建命名参数组
- **WHEN** 配置声明 optimizer parameter groups
- **THEN** optimizer 构建 MUST 按配置生成命名参数组
- **AND** 每个参数组 MUST 记录 `name`、`lr`、`weight_decay` 和 `param_count` 或等价 summary
- **AND** 训练日志和 runtime metadata MUST 能区分这些参数组

#### Scenario: 未匹配参数可诊断
- **WHEN** 参数组配置中的 module pattern 没有匹配任何 trainable 参数
- **THEN** 系统 MUST 抛出清晰错误或在显式允许时记录 warning
- **AND** 错误或 warning MUST 包含未匹配 pattern 和可用参数名前缀提示

#### Scenario: 默认 optimizer 兼容
- **WHEN** 配置未声明 parameter groups
- **THEN** 系统 MUST 保持现有单 `main` 参数组行为
- **AND** 既有训练配置 MUST 无需新增 optimizer 字段即可运行

### Requirement: JEPA downstream metadata 可追踪
系统 MUST 在 `final_config.yaml` 或 runtime metadata 中记录 JEPA downstream 结构。metadata MUST 至少包含 JEPA checkpoint 路径、context state dict prefix、pooler type、adapter type、条件特征来源、是否 freeze context encoder、参数组摘要和 ablation 标识。

#### Scenario: 写出 pooler 和 adapter metadata
- **WHEN** JEPA downstream supervised run 写出 runtime metadata
- **THEN** metadata MUST 记录 image encoder 使用的 pooler type 和 adapter type
- **AND** GPS-query 类 pooler MUST 记录 `k_queries`、`num_heads`、`condition_source` 和是否启用 attention diagnostics

#### Scenario: metadata 区分 baseline 和派生实验
- **WHEN** 分别运行 mean-pooling `fair_gps_biased` baseline 与某个 JEPA downstream 派生配置
- **THEN** runtime metadata MUST 能通过 ablation、pooler 或 adapter 字段区分两者
- **AND** metadata MUST 记录二者使用的 JEPA checkpoint 路径以便审计实验口径

### Requirement: JEPA temporal context fallback
JEPA downstream image encoder MUST 支持可配置 temporal context fallback，用历史 image latent 预测当前 degraded/missing image latent。Fallback MUST 只使用当前时间步之前的 image history，例如 `image_history[t-4:t-1]`，MUST 不读取未来帧或移动 target。

#### Scenario: 历史 latent 预测当前 latent
- **WHEN** 配置启用 JEPA temporal context fallback 且 batch 提供足够历史帧
- **THEN** encoder MUST 使用历史 image latent 生成 predicted `z_img[t]`
- **AND** metadata MUST 记录 source history range 和是否命中 fallback

#### Scenario: 不足历史可审计降级
- **WHEN** 历史长度不足以构造 `t-4:t-1`
- **THEN** 系统 MUST 使用配置声明的 clamp、zero、skip 或 raw latent fallback
- **AND** warnings MUST 记录受影响样本数和 fallback 策略

### Requirement: JEPA downstream 消费 observability metadata
JEPA downstream 模型 MUST 能消费 `image_valid_mask`、`image_observability_score` 和 Scenario D condition metadata，用于决定是否使用 raw current latent、temporal predicted latent 或二者的 gated mixture。

#### Scenario: 低 image observability 启用 predicted latent
- **WHEN** `image_observability_score` 低于配置阈值或 `image_valid_mask=false`
- **THEN** JEPA downstream MUST 能输出 temporal predicted latent 或 predicted/raw mixture
- **AND** downstream metadata MUST 记录 gating decision

#### Scenario: clean image 使用 current latent
- **WHEN** image condition 为 `D0_full_image` 且 `image_valid_mask=true`
- **THEN** JEPA downstream MUST 保持 current latent 作为默认输入
- **AND** mean-pooling 和 GPS-query baseline MUST 不因未启用 fallback 而改变行为

### Requirement: JEPA fallback 与 benchmark condition 对齐
JEPA downstream MUST 能接收 benchmark condition metadata，用于标记 `C3/C4 + D3/D4/D6/D7` advantage condition。该 metadata MUST 只影响 gating/fallback 选择和 diagnostics，不得改变 target、loss label 或 sample order。

#### Scenario: advantage condition metadata 传递
- **WHEN** Scenario D benchmark 评估 Image-JEPA+GPS 且 condition 为 `C4_severe_async + D6_burst_missing`
- **THEN** downstream MUST 能识别该 condition 为 JEPA advantage condition
- **AND** runtime metadata MUST 记录该 condition 与 fallback decision

