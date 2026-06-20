## Context

当前 GPS-conditioned JEPA 视觉端使用 `VisualPatchTokenEncoder`，默认 224x224、patch16、latent_dim=64、depth=1，输出 `[B,T,N,D]` patch tokens。supervised downstream 的 `jepa_context_image` 再通过 mean、GPS-query、hybrid residual 或 Predictive GPS-query++ pooler 输出 `[B,T,D]` image feature，交给现有 `modular_sequence` fusion/runtime 消费。

这个结构的优点是轻量、checkpoint schema 简单、下游复用成本低；风险是非重叠 patchify 和浅层 Transformer 可能丢失相邻 64-beam 所需的小尺度视觉线索。已有结果也显示 JEPA GPS-query 没有稳定超过 `Image ResNet+GPS` 强 baseline，因此需要一个可比较的 architecture sweep，而不是只凭“CNN local + Transformer global”口号替换架构。

本设计把候选分成三类，避免混淆不同收益来源：

- **checkpoint-compatible downstream variants**：复用现有 GPS-biased JEPA checkpoint，只改变 pooler、adapter、freeze/parameter group 或 fusion core。
- **Stage 1 retrain variants**：改变 visual tokenizer/backbone，需要重新训练 GPS-conditioned JEPA checkpoint，再做同口径 downstream。
- **non-JEPA anchors and controls**：不声称 JEPA reuse，例如 Image ResNet+GPS、CNN feature-map tokens、ConvNeXt-like frame/token encoder，用来判断 JEPA tokenizer 是否真正有收益。

## Goals / Non-Goals

**Goals:**

- 提供一组项目内可实现、可训练、可评估的视觉架构候选，而不是只实现单个 CNN+Transformer。
- 所有候选都走现有 `modular_sequence`、encoder/pooler/core/head 组件或 config derivatives，默认不新增 whole-model exception。
- 每个候选写出可审计 metadata：架构类别、token source、grid/token count、pooler、checkpoint reuse policy、freeze policy、参数量、比较协议和输出目录。
- 用同一 strict protocol 比较 Top-1/3/5、DBA、相邻 beam error、P0-P5 condition metrics、attention/branch diagnostics 和 compute proxy。
- 明确哪些候选可直接跑 downstream，哪些必须先重训 JEPA Stage 1，哪些只是非 JEPA anchor/control。

**Non-Goals:**

- 不把某个候选直接提升为当前推荐主线；保留/淘汰由 sweep 结果和 claim gate 决定。
- 不引入 SAM、DETR、大型 foundation vision backbone、Mamba 或新的重依赖作为第一版实现。
- 不复制训练循环、dataset 解析、batch forward 或 root-level 脚本。
- 不恢复旧 KD、HiST/Hist、Top8 selector、camera residual、GPS residual 旧路线。
- 不把 sweep 训练产物、checkpoint、attention 图或结果 CSV 纳入源码。

## Decisions

### Decision 1: Sweep 作为配置化 component baseline，而不是 whole-model exception

新增候选优先落在 `ENCODERS`、JEPA visual token encoder、downstream pooler/adapter 或 representation core 中。训练和评估继续复用现有配置加载、batch runtime、loss、metrics 和 output adaptation。

替代方案是新增一个统一 `MODELS.register("jepa_visual_sweep")` 整模型入口。该方案会复制 `modular_sequence` 已解决的 encoder/projector/core/head 组合逻辑，也会让后续 baseline 与普通 Image+GPS、JEPA GPS-query baseline 难以同口径比较，因此不采用。

### Decision 2: 定义统一 visual token encoder 输出契约

所有 JEPA Stage 1 visual encoder variant MUST 输出：

- `tokens`: `[B,T,N,D]`
- `grid_size`: `(H_tokens, W_tokens)` 或多尺度时的主 grid
- `metadata`: token source、image size、effective stride、token count、positional encoding、variant type

默认 `VisualPatchTokenEncoder` 行为保持兼容；新 tokenizer 通过 `visual_encoder.type` 选择。mask sampler、GPS angle biased sampling 和 predictor 继续基于 token/grid metadata 工作，不能假设永远是 patch16 14x14。

替代方案是在每个 variant 内部各自处理 mask 和 predictor。这样会让 JEPA Stage 1 目标不再可比，也会增加 future leak 和 checkpoint schema 风险，因此不采用。

### Decision 3: 候选矩阵分层组织

第一版 sweep 应覆盖以下候选族：

| family | variant examples | pretrain policy | purpose |
| --- | --- | --- | --- |
| baseline | patch16 mean, patch16 GPS-query K2/K4, Predictive GPS-query++ | reuse existing | 锚定当前主线 |
| patch granularity | patch14@224, patch8@224, patch16@320/384 | retrain or partial init | 判断 token 粒度/分辨率瓶颈 |
| overlap tokenizer | kernel16 stride8, kernel12 stride8, conv projection + pos embed | retrain | 测试重叠 patch 和局部平滑 |
| conv stem tokenizer | stacked 3x3 stride2 conv stem -> tokens | retrain | 测试 Early Convolution 的优化稳定性和局部偏置 |
| local transformer | depthwise FFN, local-window preblock, relative/conditional position | retrain | 测试局部性是否比纯 patchify 更重要 |
| CvT-like | convolutional Q/K/V projection 或 depthwise token mixing | retrain | 测试卷积投影和 token mixing |
| CNN tokens | ResNet18/34 layer3/layer4 feature map tokens -> GPS-query pooler | supervised anchor or JEPA-style optional | 判断 CNN feature map tokens 是否优于 JEPA patch tokens |
| multi-scale tokens | layer3+layer4/FPN-like token concat + scale embedding | supervised anchor or optional JEPA | 判断多尺度视觉线索是否关键 |
| frame embedding | existing ResNet18 frame embedding, ConvNeXt-like lightweight CNN | supervised anchor | 强 anchor 和非 Transformer 对照 |
| pooling/core | mean, GPS-query K2/K4/K8, content+GPS residual, Predictive GPS-query++, K-token fusion | reuse or paired | 判断瓶颈是否在 pooling/fusion |

“所有可能”在本变更中定义为：不新增重依赖、不复制 runtime、能在当前数据/metric/输出边界内同口径训练和比较的实用候选。超大外部 backbone 或完整 detection/segmentation model 不纳入第一版。

### Decision 4: Checkpoint compatibility 显式建模

每个 variant MUST 声明 `checkpoint_policy`：

- `exact_reuse`: 参数形状与现有 `context_encoder` checkpoint 完全匹配。
- `partial_reuse`: 只加载可匹配子模块，必须记录 missing/unexpected keys。
- `pos_interpolate`: 允许位置编码插值，必须记录原 token grid 和目标 token grid。
- `fresh_stage1_required`: tokenizer/backbone 改变，不允许伪装成复用现有 JEPA checkpoint。
- `supervised_only_anchor`: 非 JEPA anchor，不加载 JEPA checkpoint。

替代方案是让 `strict=false` 静默吞掉不匹配权重。该方案会让结果不可解释，尤其容易把随机初始化 tokenizer 当作 JEPA 复用，因此不采用。

### Decision 5: Pooling 输出默认保持 `[B,T,D]`，K-token fusion 单独 opt-in

现有 downstream 契约默认输出 `[B,T,D]`，这能复用 projector、representation core、beam head 和 metrics。K-token 保留给 fusion core 是重要候选，但必须显式配置，例如 `pooler.output_mode: tokens` 或专用 token-aware core，并记录 token count。

替代方案是所有 pooler 都输出 `[B,T,K,D]`。该方案表达力更强，但会影响当前 runtime、head 和已有 baseline，第一版不作为默认。

### Decision 6: 统一 architecture sweep manifest 和 claim gate

新增 sweep manifest 记录候选、命令、协议、metadata、metric、diagnostics 和结果路径。claim gate 只允许 strict comparable rows 参与“保留/淘汰”判断；smoke 或 partial runs 只能作为可运行性和机制诊断。

至少记录：

- `variant_id`
- `family`
- `visual_encoder.type`
- `image_size`
- `token_grid`
- `token_count`
- `effective_stride`
- `pooler.type`
- `pooler.output_mode`
- `checkpoint_policy`
- `freeze_policy`
- `parameter_groups`
- `params_trainable`
- `compute_proxy`
- `strict_comparison`
- `metrics`
- `diagnostics`
- `provenance`

## Risks / Trade-offs

- [Risk] 矩阵太大，训练成本失控。→ Mitigation：生成 `smoke`, `lowmem`, `strict` 三层配置；strict 只跑通过 smoke 的候选，manifest 明确 run tier。
- [Risk] Stage 1 retrain variants 与 downstream-only variants 混比。→ Mitigation：`checkpoint_policy` 和 `pretrain_policy` 必须进入 metadata，claim gate 按 policy 分组比较。
- [Risk] 高分辨率或 patch8 token 数导致显存爆炸。→ Mitigation：配置必须声明 `max_tokens`、batch size、AMP、gradient accumulation 或 lowmem fallback；超过预算时 fail fast。
- [Risk] GPS-query attention 学成 GPS shortcut。→ Mitigation：保留 GPS-only、Image ResNet+GPS、wrong-GPS/P3、counterfactual GPS 和 attention entropy diagnostics。
- [Risk] CNN token anchor 提升来自 ImageNet 预训练而非 token/pooler 设计。→ Mitigation：单独标记 pretrained/frozen policy，并与 supervised ResNet+GPS anchor 成对比较。
- [Risk] K-token fusion 改变 downstream 契约。→ Mitigation：默认仍 `[B,T,D]`；K-token 作为 opt-in core，并有 focused forward/config tests。
- [Risk] 结果目录和 checkpoint 太多。→ Mitigation：统一 ignored output root，并要求 manifest 只记录相对路径和 provenance，不提交产物。

## Migration Plan

1. 新增 visual token encoder registry 或等价窄构建 helper，保持默认 `patch_vit` 与现有 `VisualPatchTokenEncoder` 行为一致。
2. 实现 tokenizer variants：overlap patch、conv stem、local/depthwise token mixing、CvT-like projection；CNN token variants 复用已有 ResNet/torchvision 依赖或项目内轻量 CNN。
3. 扩展 JEPA mask sampler/predictor metadata，使其从 token/grid metadata 读取 token count，不硬编码 196 或 patch16。
4. 扩展 `jepa_context_image` downstream，支持新 token metadata、CNN feature-map tokens、多尺度 tokens 和可选 K-token output mode。
5. 新增 sweep config generator 或实体 YAML 矩阵；所有配置派生自匹配 baseline，只覆盖 architecture variables。
6. 新增 manifest/diagnostic writer，统一记录 architecture metadata、metrics、attention/branch diagnostics 和 run commands。
7. 新增 focused tests 与配置加载 tests。
8. 运行 OpenSpec 校验、配置加载、模型 forward smoke 和必要 focused tests。

Rollback 方式：删除新增 variant registry/components、sweep configs、diagnostics helper、tests 和本 change artifacts；现有 patch16 JEPA checkpoint、mean/GPS-query/Predictive GPS-query++ 配置不需要迁移。

## Open Questions

- strict sweep 是否默认只跑 2604 S32/S33/S34，还是同时生成 BeamBench-fair 版本。
- `ConvNeXt-like` 是否使用 torchvision 现成 backbone，还是先实现项目内轻量 ConvNeXt block 以减少依赖/权重下载风险。
- K-token fusion 的第一版是否只做 supervised downstream，不进入 JEPA Stage 1 pretraining。
- 高分辨率候选的最大 token budget 和 batch size 是否按当前单卡显存写死为 lowmem overlay，还是由用户运行时覆盖。
