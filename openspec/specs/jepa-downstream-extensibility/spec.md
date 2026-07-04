# jepa-downstream-extensibility Specification

## Purpose
定义 JEPA context image encoder 在 supervised downstream 中的可插拔 pooler、adapter、optimizer 参数组和 runtime metadata 契约，保证派生实验能与 GPS-biased mean-pooling baseline 可比，同时不改变 Stage 1 JEPA checkpoint 或主训练评估流程。
## Requirements
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

### Requirement: Hybrid residual query pooler
JEPA downstream MUST 支持一个可注册 hybrid residual query pooler，用于结合 patch-token mean pooling、learned content query pooling 和 GPS-conditioned residual-bias query pooling。该 pooler MUST 默认输出 `[B,T,D]`，并 MUST 保持现有 mean 和 GPS-query pooler 行为兼容。

#### Scenario: Hybrid pooler 构建和输出
- **WHEN** `jepa_context_image` 配置 `pooler.type: hybrid_residual_query` 或等价类型
- **THEN** 系统 MUST 构建 mean/content/GPS-conditioned residual-bias query 路径
- **AND** forward MUST 接收 JEPA patch tokens `[B,T,N,D]` 与可选 GPS condition features `[B,T,C]`
- **AND** 输出 MUST 为现有 modular sequence projector 可消费的 `[B,T,D]`

#### Scenario: GPS 只作为 residual bias
- **WHEN** hybrid pooler 同时计算 mean/content query latent 和 GPS-query latent
- **THEN** GPS-query latent MUST 作为相对 mean/content anchor 的 residual 修正参与输出
- **AND** pooler MUST 提供配置项或初始化策略，避免 GPS-query path 在训练初期完全覆盖 mean/content anchor

### Requirement: Temporal predicted latent auxiliary branch
JEPA downstream image encoder MUST 在 opt-in 配置下暴露 current latent 与 temporal predicted latent auxiliary branch。该行为 MUST 不改变默认 forward 输出、checkpoint schema 或现有 mean/GPS-query baseline 语义。

#### Scenario: 暴露 current 和 predicted latent
- **WHEN** 配置启用 predictive auxiliary branch 且输入序列提供历史帧
- **THEN** encoder MUST 记录或返回 `current_latent`、`temporal_predicted_latent`、branch availability、source history range 和 fallback metadata
- **AND** temporal prediction MUST 只使用当前时间步之前的历史 latent，不得读取未来帧

#### Scenario: 历史不足可诊断降级
- **WHEN** 历史长度不足以生成 temporal predicted latent
- **THEN** encoder MUST 按配置使用 raw、skip、zero 或 clamp fallback
- **AND** metadata MUST 记录 insufficient history count 和 fallback strategy

### Requirement: Feature-consistency fusion diagnostics
JEPA downstream predictive fusion MUST 支持基于 latent 一致性的 gate 或 helper，用于融合 current image latent、temporal predicted latent 和 GPS-derived reliability/bias 信息。该 gate MUST 不直接读取 benchmark condition id。

#### Scenario: Gate 输入不包含 condition id
- **WHEN** feature-consistency gate forward
- **THEN** gate MUST 只消费 latent tensors、valid masks、observability score、GPS delay/reliability 或等价连续特征
- **AND** gate MUST NOT 直接消费 `c_idx`、`d_idx`、`predictive_condition_id`、`gps_condition` 或 `image_condition`

#### Scenario: 写出 consistency diagnostics
- **WHEN** predictive JEPA model forward 完成
- **THEN** output 或 runtime metadata MUST 包含 current/predicted/GPS-derived branch availability、gate weights 或 equivalent scores、latent consistency summary 和 warnings
- **AND** 普通 JEPA mean/GPS-query baseline 在未启用该功能时 MUST 不要求这些 diagnostics

### Requirement: Predictive GPS-query++ downstream pooler
JEPA downstream MUST support an opt-in Predictive GPS-query++ path that combines current content latent, GPS-conditioned residual latent, and causal temporal predicted latent. This path MUST preserve existing mean-pooling and `gps_query_attention` behavior unless explicitly selected by configuration.

#### Scenario: 构建 Predictive GPS-query++ pooler
- **WHEN** `jepa_context_image` 配置声明 `pooler.type: predictive_gps_query` 或等价 opt-in 类型
- **THEN** 系统 MUST 构建 content-query anchor、GPS-query residual path、temporal latent predictor 和 reliability-aware gate
- **AND** pooler 输出 MUST 保持 `[B,T,D]`，可被现有 projector、representation core 和 beam head 消费
- **AND** 现有 `pooling: mean`、`pooling: gps_query_attention` 和 `pooler.type: hybrid_residual_query` 配置 MUST 不改变语义

#### Scenario: GPS path 作为 residual 条件
- **WHEN** Predictive GPS-query++ 同时计算 content latent 和 GPS-query latent
- **THEN** GPS-query latent MUST 作为相对 content 或 mean anchor 的 residual/bias 参与输出
- **AND** 配置 MUST 提供 residual scale、initialization 或 gating 机制，避免 GPS path 在训练初期完全覆盖 content anchor

#### Scenario: 输出 GPS-query++ diagnostics
- **WHEN** Predictive GPS-query++ forward 完成
- **THEN** runtime diagnostics MUST 包含 content branch、GPS residual branch、temporal predicted branch 的 availability、gate weights 或 equivalent scores
- **AND** diagnostics MUST 记录 residual scale、GPS-query attention summary、temporal source history range 和 fallback/warning 状态

### Requirement: Causal temporal latent predictor
JEPA downstream predictive path MUST support a causal temporal latent predictor that predicts current or future image latent from prior image latents only. The predictor MUST be opt-in and MUST NOT read future frames, target labels, beam powers, or sample order beyond the current batch/time sequence.

#### Scenario: 使用历史 latent 预测当前 latent
- **WHEN** Predictive GPS-query++ 启用 temporal predictor 且输入序列提供当前步之前的历史 image latent
- **THEN** predictor MUST produce `temporal_predicted_latent` aligned with the current prediction step
- **AND** metadata MUST record history window、source history range、predictor type、availability mask 和 insufficient-history fallback

#### Scenario: 拒绝 future leak
- **WHEN** temporal predictor 生成 step `t` 的 predicted latent
- **THEN** predictor MUST only consume image latent from steps `< t`
- **AND** tests MUST cover that source history range never includes `t` or future steps

#### Scenario: 历史不足可审计降级
- **WHEN** 当前样本没有足够历史 latent
- **THEN** predictor MUST use configured `raw`、`skip`、`zero` 或 `clamp` fallback
- **AND** diagnostics MUST record affected count and fallback strategy

### Requirement: Predictive JEPA auxiliary latent objectives
Predictive GPS-query++ training MAY enable auxiliary latent objectives that encourage temporal predicted latent and corrupt-view latent to align with clean target latent. These objectives MUST be opt-in and MUST NOT change default beam-only training unless configured.

#### Scenario: 启用 latent prediction loss
- **WHEN** training config declares predictive latent auxiliary loss
- **THEN** training MUST compute a loss between predicted/corrupt latent and clean detached target latent or configured target representation
- **AND** loss logging MUST include objective name、weight、sample count 和 whether target latent is detached

#### Scenario: 默认训练保持兼容
- **WHEN** config does not declare predictive latent auxiliary losses
- **THEN** supervised beam loss、metrics、checkpoint workflow 和 model output adaptation MUST remain unchanged

### Requirement: GPS-query++ metadata and compatibility
Predictive GPS-query++ runs MUST be distinguishable from existing JEPA GPS-query baseline runs in final config and runtime metadata.

#### Scenario: 写出架构 metadata
- **WHEN** Predictive GPS-query++ model writes final config or runtime metadata
- **THEN** metadata MUST include pooler type、content query count、GPS query count、temporal predictor type、reliability gate type、residual scale and enabled auxiliary losses
- **AND** metadata MUST include source JEPA checkpoint path and whether context encoder is frozen

#### Scenario: 旧 GPS-query checkpoint 不被误加载为 GPS-query++
- **WHEN** loader receives a checkpoint whose metadata indicates `gps_query_attention`
- **THEN** system MUST NOT silently treat it as Predictive GPS-query++
- **AND** incompatible missing/unexpected keys MUST produce clear diagnostics unless user explicitly requests non-strict transfer

### Requirement: Downstream visual token source variants
JEPA downstream image encoder MUST 能在 opt-in 配置下消费不同 visual token sources，包括 JEPA patch tokens、overlap/conv/local visual encoder tokens、CNN feature-map tokens 和多尺度 tokens。默认输出 MUST 继续保持现有 `[B,T,D]` image feature 契约，除非配置显式启用 token-aware fusion。

#### Scenario: 新 token source 通过 pooler 消费
- **WHEN** `jepa_context_image` 或等价 downstream image encoder 使用新的 visual token source
- **THEN** 系统 MUST 将 tokens `[B,T,N,D]` 和 token metadata 传给配置的 pooler
- **AND** pooler 默认输出 MUST 为现有 projector/core 可消费的 `[B,T,D]`

#### Scenario: CNN feature-map tokens 保留来源 metadata
- **WHEN** downstream 使用 CNN layer feature map tokens
- **THEN** metadata MUST 记录 backbone type、selected stages、feature grid、token count、pretrained/freeze policy 和 projection dimension
- **AND** 系统 MUST 区分该候选是 JEPA reuse、JEPA-style retrain 还是 supervised-only anchor

### Requirement: K-token downstream fusion opt-in
JEPA downstream MUST 支持显式 opt-in 的 K-token output mode，用于保留 GPS-query、content-query 或多尺度 query tokens 给 token-aware representation core 或显式 token readout。未启用时，mean/GPS-query/hybrid/Predictive GPS-query++ pooler MUST 继续输出 `[B,T,D]`。启用 K-token output mode 时，系统 MUST 记录 token source、readout type、是否 trainable、`k_tokens` 和 core/input compatibility metadata。

#### Scenario: 默认 pooler 输出不变
- **WHEN** 配置未声明 K-token output mode
- **THEN** JEPA downstream pooler MUST 输出 `[B,T,D]`
- **AND** 现有 beam head、loss、metrics 和 ModelOutput adaptation MUST 无需新增分支

#### Scenario: 启用 K-token output mode
- **WHEN** 配置声明 pooler 输出 `[B,T,K,D]` 或等价 token-aware output
- **THEN** 配置 MUST 同时声明能消费该输出的 representation core、adapter 或 token readout
- **AND** runtime metadata MUST 记录 `output_mode`、`k_tokens`、token source、core type 和 token readout type

#### Scenario: 不兼容 core 被拒绝
- **WHEN** pooler 输出 K-token representation 但 representation core、adapter 或 readout 不能消费该 token shape
- **THEN** 系统 MUST 在配置加载或模型构建时抛出清晰错误
- **AND** 错误信息 MUST 指出 pooler output mode、实际 output shape 和 core/readout input contract 不兼容

#### Scenario: legacy token-aware transformer 可审计
- **WHEN** 配置使用现有 `token_aware_transformer` 消费 K-token features 且未声明显式 readout
- **THEN** metadata MUST 将 readout 标记为 `legacy_uniform_mean` 或等价值
- **AND** metadata MUST 记录该路径最终会对 token/channel 维做均值聚合
- **AND** 旧 checkpoint 和旧 final config MUST 不被误标记为 learned readout

#### Scenario: learned token readout 显式 opt-in
- **WHEN** 配置声明 learned、weighted 或 attention-based token readout
- **THEN** 系统 MUST 构建对应 readout 并输出现有 beam head 可消费的 `[B,T,D]` feature
- **AND** readout MUST 记录 trainable parameter count、readout weight summary 或等价 diagnostics
- **AND** 默认 mean、GPS-query frame、hybrid residual query 和 Predictive GPS-query++ 配置 MUST 不改变语义

#### Scenario: token readout 不读取 oracle 信息
- **WHEN** token readout 或 token-aware core forward
- **THEN** readout MUST NOT 读取 target beam、beam power oracle、sample label、P0-P5 condition id 或 evaluation metric
- **AND** condition metadata MAY 只用于 diagnostics、masking 可用性或离线分组统计

### Requirement: Visual token diagnostics for downstream sweep
JEPA downstream architecture variants MUST 写出统一 visual token diagnostics。diagnostics MUST 能区分 token count、attention map shape、branch/gate weights、pooler output mode、checkpoint policy 和 condition feature source。

#### Scenario: GPS-query attention diagnostics 记录 token grid
- **WHEN** GPS-query 类 pooler 启用 attention diagnostics
- **THEN** diagnostics MUST 记录 attention map shape、token grid、token count、query count、attention entropy 或 peakiness summary
- **AND** attention map MUST detach 后用于日志或诊断，训练主损失 MUST 不依赖诊断张量

#### Scenario: predictive pooler 记录 branch 来源
- **WHEN** Predictive GPS-query++ 或 hybrid residual query pooler 完成 forward
- **THEN** diagnostics MUST 记录 content branch、GPS residual branch、temporal branch 或 equivalent branch availability
- **AND** diagnostics MUST 不直接消费 target label、beam power oracle 或 benchmark condition id 作为模型输入

### Requirement: GPS-query attention aggregation metadata
JEPA downstream GPS-query 类 pooler MUST 在启用 attention diagnostics 时记录 attention 聚合 metadata。Metadata MUST 说明 attention 是否跨 head 平均、是否跨 query/time 聚合、原始 attention shape、诊断输出 shape、condition feature source 和 token grid 或 token count。

#### Scenario: 记录平均 attention metadata
- **WHEN** `GPSQueryPool` 或等价 GPS-query pooler 使用默认 averaged attention diagnostics
- **THEN** pooler diagnostics MUST 记录 `attention_head_aggregation=averaged`
- **AND** diagnostics MUST 记录原始可见 attention shape、输出 attention map shape、query count、token count 和 condition feature source

#### Scenario: 记录分支 attention metadata
- **WHEN** predictive 或 hybrid GPS-query pooler 同时产生 content attention 和 GPS attention
- **THEN** diagnostics MUST 分别记录 content branch 和 GPS branch 的 attention summary 或 unavailable reason
- **AND** GPS branch attention MUST 标明是否作为 `last_attention_map` 暴露给 visual analysis

### Requirement: Opt-in per-head attention diagnostics
JEPA downstream GPS-query 类 pooler MUST 支持 opt-in per-head attention diagnostics，且该模式 MUST 不改变训练主输出、loss 输入、checkpoint 加载语义或默认配置行为。未显式开启时，系统 MUST 保持现有 averaged attention 行为。

#### Scenario: 默认保持 averaged attention
- **WHEN** 用户未显式启用 per-head attention diagnostics
- **THEN** GPS-query 类 pooler MUST 保持现有 averaged attention map 输出语义
- **AND** 现有 `return_attention=True` shape 兼容测试 MUST 继续通过

#### Scenario: 开启 per-head diagnostics
- **WHEN** 分析或诊断配置显式启用 per-head attention diagnostics
- **THEN** pooler MUST 返回或缓存包含 head 维度的 attention diagnostics
- **AND** diagnostics MUST 记录 per-head shape、head count 和用于下游 summary 的 head aggregation method
- **AND** 训练 forward 的主 pooled feature shape MUST 不变

#### Scenario: per-head diagnostics 受采样限制
- **WHEN** per-head attention diagnostics 开启且样本数超过 attention case 限制
- **THEN** 系统 MUST 只为受 `max_attention_cases` 或等价配置限制的样本保留 per-head 明细
- **AND** manifest MUST 记录被截断的样本数或 skipped reason

### Requirement: 现有 supervised/adaptation workflow 不变
新增 JEPA 预训练 workflow MUST 不改变现有 beam、occlusion、position、multitask、GPS v2、CSI hardening 或 supervised fusion workflow 的默认配置和指标。Raymobtime s008、legacy KD、standalone Top8 selector、residual、BGAM 和 viewer 路线仍只作为退役或 supporting guard 语义保留，不属于当前默认 workflow。

#### Scenario: 默认 beam 配置行为不变
- **WHEN** 用户加载未设置 `experiment.objective` 的现有 supervised beam 配置
- **THEN** 系统 MUST 继续默认使用 `beam` objective
- **AND** 系统 MUST 继续计算 beam loss、Top-K、DBA 和 `val_adba`

#### Scenario: 旧 KD 入口仍被拒绝
- **WHEN** 用户请求旧 `logits_kd`、`rkd`、`teacher_no_kd` 或 retired fusion KD 配置
- **THEN** 系统 MUST 继续拒绝该配置
- **AND** 错误信息 MUST 继续指向当前 supervised/adaptation 或 JEPA 预训练入口，而不是恢复旧 KD workflow

### Requirement: JEPA downstream pooler 和 adapter 注册
项目 MUST 通过轻量组件构建边界暴露 JEPA downstream pooler。内置 mean pooler 和 GPS-query attention pooler MUST 能通过配置名称构建；identity adapter MAY 作为默认 no-op 路径内联，而不是必须注册为独立 adapter。未知 pooler 名称 MUST 使用现有 registry 错误风格报告；未知 adapter 名称只有在非 identity adapter 配置面被保留时才需要注册表式错误。

#### Scenario: 按名称构建 mean pooler
- **WHEN** `jepa_context_image` 配置声明 downstream pooler 为 `mean`
- **THEN** 系统 MUST 构建 mean pooler
- **AND** 该 pooler MUST 接收 patch tokens `[B,T,N,D]` 并输出 `[B,T,D]`

#### Scenario: 按名称构建 GPS-query pooler
- **WHEN** `jepa_context_image` 配置声明 downstream pooler 为 `gps_query_attention`
- **THEN** 系统 MUST 构建 GPS-query attention pooler
- **AND** 构建参数 MUST 支持 `k_queries`、`num_heads`、`condition_dim`、`latent_dim`、dropout 和 condition source

#### Scenario: identity adapter 内联为 no-op
- **WHEN** `jepa_context_image` 配置未声明 adapter 或声明 adapter 为 `identity`
- **THEN** 系统 MUST 使用不改变输入 shape 的无操作路径
- **AND** 现有配置 MUST 无需新增 adapter 字段即可运行

#### Scenario: 未知 JEPA downstream 组件可诊断
- **WHEN** 用户配置不存在的 JEPA downstream pooler 名称
- **THEN** 系统 MUST 拒绝构建
- **AND** 错误信息 MUST 包含请求名称、组件类别和可用 pooler 名称

### Requirement: JEPA downstream 注册保持轻量导入
JEPA downstream pooler 的注册 MUST 不破坏 registry 轻量导入边界。导入 `kd_sensing.registries` MUST 不 eager import torch model implementation、dataset、diagnostics、训练器或 checkpoint 文件；默认组件导入流程 MUST 显式注册内置 JEPA downstream pooler。identity adapter 若内联为 no-op，则不需要默认注册流程。

#### Scenario: 轻量导入 registry 不触发 JEPA model
- **WHEN** 开发者仅执行 `import kd_sensing.registries`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不 eager import `kd_sensing.models.jepa` 或 JEPA downstream 实现模块

#### Scenario: 默认组件导入后可构建 JEPA downstream pooler
- **WHEN** 构建流程调用默认组件导入函数
- **THEN** 内置 JEPA downstream pooler MUST 完成注册
- **AND** 用户配置中的内置 pooler 名称 MUST 可解析

### Requirement: JEPA downstream query/pooling helper 必须可独立演进
JEPA downstream model 重构 MUST 在不改变 current config 行为的前提下拆分 query construction、token pooling、GPS-query compatibility、head construction 和 diagnostics metadata。

#### Scenario: downstream config 兼容
- **WHEN** downstream helper modules are introduced
- **THEN** 既有 JEPA downstream configs MUST 加载并构建相同 model surface
- **AND** GPS-query/pooler diagnostics fields MUST remain available to visual analysis and benchmark workflows

