## Why

当前默认多模态融合仍偏向 early-concat 或 legacy fusion 路线，跨模态交互主要发生在拼接后的时序层，无法显式表达“同一时间步不同传感器 token 之间”的自注意力关系。新增 CLS-token Transformer 融合可以把五个模态的帧级特征作为 token 序列建模，并将其设为默认融合方式，给后续多模态实验一个更清晰、可扩展、可诊断的基线。

## What Changes

- 新增一种默认 fusion student 架构：将每个时间步的五个模态特征序列化为 token，前置可学习 CLS token，并添加 token-type embedding 与 time embedding。
- 新增 Transformer Encoder 融合路径，通过多头自注意力和前馈层对模态 token 进行深度交互。
- 新架构 MUST 支持 `image`、`radar`、`gps`、`lidar`、`mmwave` 的任意合法非空组合；五模态场景下每个时间步的 token 序列为 `CLS + 5` 个模态 token。
- 将推荐/default 多模态 fusion 配置切换到该 CLS-token Transformer 融合方式，同时保留 early-concat、legacy fusion、CRAF 和 MARF 的显式配置入口。
- 新模型输出 MUST 兼容现有训练、验证、KD、G2D feature diagnostics 和 `adapt_model_output()` 契约。
- 不移除现有 `fusion_teacher`、`fusion_student`、`craf_fusion`、`marf_fusion` 或模块化 `early_concat_gru` 配置。

## Capabilities

### New Capabilities
- `cls-token-transformer-fusion`: 定义 CLS-token Transformer fusion 模型的 token 序列化、嵌入、Transformer Encoder、输出和诊断契约。

### Modified Capabilities
- `configurable-multimodal-fusion`: 修改推荐/default fusion 配置语义，使默认混合方式使用 CLS-token Transformer fusion，并要求 legacy/early-concat 路线保留为显式 baseline。
- `component-registry`: 要求默认组件导入流程注册新的 CLS-token Transformer fusion 模型入口，并保持 registry 轻量导入边界。

## Impact

- 影响模型层：新增 fusion 模型或模块化 representation core，复用现有模态 encoder/projector，新增 CLS token、token-type embedding、time embedding 和 Transformer Encoder。
- 影响配置层：更新默认 fusion 配置、五模态 fusion 推荐配置、可选双/三/四模态配置生成语义，以及 run name/模型 type 命名。
- 影响训练与评估：继续复用 `experiment.task: fusion` 的 batch 准备、forward helper、future slot 选择、loss、metric 和 checkpoint 逻辑。
- 影响测试：新增 forward shape、mask/模态子集、配置加载、registry 注册、G2D diagnostics 和默认配置选择测试。
