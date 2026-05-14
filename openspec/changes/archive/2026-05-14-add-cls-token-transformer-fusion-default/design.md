## Context

当前 fusion 路线同时存在 legacy early-concat、模块化 `early_concat_gru`、token transformer baseline、CRAF 和 MARF。early-concat 的优势是简单稳定，但跨模态交互发生在拼接后的向量空间中；CRAF/MARF 更强，但带有 reliability、counterfactual、routing 或 subset training 等额外机制，不适合作为最小默认融合基线。

本变更新增一个专门的 CLS-token Transformer fusion 默认方法。它只负责把多模态帧级特征序列化为 token，通过 token-type embedding、time embedding 和 Transformer Encoder 建模跨模态与跨时间交互，然后用 CLS 表示完成 beam prediction。该方法不引入新外部依赖，继续使用 PyTorch 和现有 fusion 训练入口。

## Goals / Non-Goals

**Goals:**
- 新增可通过 registry 构建的 `cls_token_transformer_fusion` 模型。
- 支持 `image`、`radar`、`gps`、`lidar`、`mmwave` 的任意合法非空组合，并默认面向五模态融合。
- 使用固定模态顺序，将 encoder/projector 后的特征序列化为 token；五模态每个时间步产生 `5 x d_model` 模态 token。
- 在 token 序列最前方添加可学习 CLS token，并为模态 token 添加 token-type embedding 与 time embedding。
- 使用 Transformer Encoder 实现模态与时间交互，输出兼容现有 loss、metric、KD、G2D diagnostics 和 subset evaluation。
- 将推荐/default fusion student 配置切换到新模型，同时保留 legacy/early-concat/CRAF/MARF 显式配置入口。

**Non-Goals:**
- 不移除 `fusion_teacher`、`fusion_student`、`token_transformer_fusion`、`craf_fusion` 或 `marf_fusion`。
- 不引入新的数据字段、数据集 split、标签定义或训练入口。
- 不实现 CRAF/MARF 的 reliability gate、anchor routing、counterfactual loss 或 subset-aware training 语义。
- 不改变单模态 teacher/student 配置。

## Decisions

### 1. 新增独立注册模型，而不是改写现有 token transformer baseline

新增 `cls_token_transformer_fusion`，保持现有 `token_transformer_fusion` 作为不带 CLS 的 baseline。这样默认配置可以明确切到新架构，历史实验仍能通过旧注册名复现。

备选方案是直接修改 `token_transformer_fusion`。该方案会让已有 baseline 的语义漂移，影响旧 checkpoint 和对比实验解释，因此不采用。

### 2. 复用现有模态 encoder，新增统一 token 序列化层

模型继续使用现有 image/radar/gps/lidar/mmWave encoder，将每个模态映射到 `[B, T, d_model]`。多模态特征按项目固定模态顺序堆叠为 `[B, K, T, d_model]`，再按时间优先序列化为 `[B, T*K, d_model]`。五模态时，每个时间步贡献 `5 x d_model` 的模态 token。

备选方案是先在每个时间步做模态拼接再 Transformer。该方案会丢失传感器 token 的独立身份，不符合 token-type embedding 的设计目标。

### 3. 使用单个全局 CLS token 加 horizon head 输出未来槽位

序列最前方添加一个可学习 CLS token。Transformer 后取 CLS hidden state，通过 horizon head 生成 `[B, num_pred, num_classes]` logits。horizon head 可以用 `Linear(d_model, num_pred * num_classes)`，也可以使用 learnable horizon embedding 加共享分类头；实现应优先选择可测试、形状清晰的方案。

备选方案是为每个时间步或每个 horizon 添加多个 CLS token。该方案更复杂，也偏离用户描述的单个 CLS token，暂不作为默认方案。

### 4. token-type embedding 与 time embedding 分离

每个模态 token 添加模态类型编码，CLS token 使用独立的 CLS type id。每个模态 token 还添加 time embedding，用于区分历史时间位置。CLS token 不绑定某个历史时间，可使用独立 CLS embedding 或零 time embedding。

该设计让模型同时知道“这个 token 来自哪个传感器”和“它属于哪个历史时间步”，避免把不同时间的同一模态 token 混淆。

### 5. force modality mask 通过 token padding mask 实现

模型支持 `force_modality_mask`，用于评估模态子集和后续 diagnostics。被屏蔽模态的 token MUST 在 Transformer attention 中通过 padding mask 排除，并且 diagnostics 中对应模态 token 不应被 G2D 或 subset 评估误用。至少保留一个模态；否则抛出清晰错误。

备选方案是将屏蔽模态 token 置零但仍参与 attention。该方案仍会通过 type/time embedding 泄露被屏蔽模态存在，不采用。

### 6. 默认配置切换只影响推荐 fusion student

默认/recommended fusion student no-KD、logits KD 和 RKD 配置使用 `cls_token_transformer_fusion`。需要训练 legacy fusion teacher 时，`teacher_no_kd` 可继续使用 `fusion_teacher`，作为蒸馏 teacher 或对照基线。显式 early-concat、CRAF 和 MARF 配置不变。

这样既满足“默认混合方式”切换，也保留 teacher/student KD 工作流和历史可比性。

## Risks / Trade-offs

- [Risk] Transformer token 数为 `1 + T*K`，五模态和较长序列会增加显存与训练时间。→ Mitigation：默认 `d_model`、层数和 heads 保持保守；测试覆盖 batch smoke，并在配置中暴露 `num_layers`、`num_heads`、`dropout`。
- [Risk] 直接输出 `[B, num_pred, C]` 与 early-concat 的 `[B, T, C]` 形状不同。→ Mitigation：现有 `select_prediction_slots()` 已接受时间维等于 `num_pred` 的 logits；测试必须覆盖 loss/metric 对齐。
- [Risk] 默认配置切换会改变历史实验结果。→ Mitigation：保留 legacy/early-concat 显式配置和 run name，默认配置的 `final_config.yaml` 记录新模型 type。
- [Risk] 复用 CRAF 内部 tokenizer/helper 可能扩大模块耦合。→ Mitigation：优先抽取或新增共享 token helper，避免新模型依赖 CRAF 的私有实现细节。
- [Risk] G2D diagnostics 需要按模态拆分 token。→ Mitigation：输出 `token_features: [B,K,T,D]`、`modalities` 和 `effective_modality_mask`，不只输出扁平 Transformer memory。

## Migration Plan

1. 新增模型注册名和单元测试，不修改默认配置。
2. 增加新模型 smoke 配置并验证训练/评估 forward、loss、metric。
3. 更新推荐/default fusion 配置，让 student 路线使用 `cls_token_transformer_fusion`。
4. 保留 legacy/early-concat 配置路径，并在文档或配置命名中明确 baseline 语义。
5. 如默认配置出现回归，可将默认配置恢复为 early-concat，同时保留新模型显式配置入口。

## Open Questions

- horizon head 采用一次性 `Linear(d_model, H*C)`，还是 `CLS + horizon_embedding` 后共享分类头？两者均满足契约，实施时可根据测试清晰度和参数可解释性选择。
- 默认五模态配置是否应加载单模态 teacher registry 初始化 encoder，还是先保持纯端到端训练？当前提案不强制 teacher-prior 初始化。
