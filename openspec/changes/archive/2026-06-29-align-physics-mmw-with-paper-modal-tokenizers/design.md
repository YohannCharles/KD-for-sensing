## Context

`pinn_multimodal_beam` 当前已经是登记过的 whole-model exception：它在 forward 内耦合了路径参数预测、可微 ULA 信道合成、physics logits、direct logits 和 diagnostics，因此不能简单降级成普通 `modular_sequence`。但现有前端 `_SmallSequenceEncoder` 只对每个模态取 mean/std/max/min，再做线性映射；这对于物理一致链路过弱，容易退化成“粗融合 + 物理辅助 loss”。

arXiv:2603.29796 的可借鉴点是：不同模态先由专用 encoder/tokenizer 映射成统一 token，再加入时间、模态和局部位置嵌入，并通过共享 Transformer 融合。该路线比直接特征拼接更适合本项目的多模态感知层。用户额外要求图像编码器必须使用项目已有 `jepa_context_image`，即卷积/patch embedding + 单层 Transformer 的轻量 ViT 风格 JEPA context encoder，并且不能使用 GPS context。

数据边界也必须收敛：真实 MMW inspection 显示当前 CSI 监督是 `[T, Nsc, Nant, 2]`，实际样例可为 `[1, 1, 64, 2]`。因此本设计只声明窄带阵列信道重构，不声明完整宽带 OFDM CSI 重构。

## Goals / Non-Goals

**Goals:**

- 将 `pinn_multimodal_beam` 前端升级为论文风格的模态 tokenizer + 共享 Transformer。
- 图像 tokenizer 复用 `jepa_context_image`，使用不依赖 GPS 的 pooling，并要求正式实验使用预训练 checkpoint。
- 将 sparse pilot / sparse antenna / 受限 RF 观测作为无线输入主线，完整窄带 CSI 只作为监督 target 或 oracle upper-bound。
- 保留现有物理链路：路径参数头、可微信道合成器、beam scoring、hybrid logits 和 physics loss。
- 让实验 metadata 能明确区分正式 paper-style baseline、debug/smoke、oracle upper-bound 和 checkpoint/pretraining 状态。

**Non-Goals:**

- 不复现 arXiv:2603.29796 的完整训练协议、数据集、表格或所有实现细节。
- 不引入 diffusion、flow matching 或生成式 CSI 重构。
- 不把 wide-beam RSS 作为当前物理一致框架的主无线输入；它可以作为后续低开销 beam probing 观测，但不足以直接支撑当前的窄带阵列信道重构监督链。
- 不声明完整 CSI 或宽带多子载波重构能力。
- 不新增根目录训练脚本，不复制训练循环。

## Decisions

### Decision 1: 继续使用 `pinn_multimodal_beam`，不新建整模型注册名

实现应扩展现有 whole-model exception，而不是新增 `pinn_paper_transformer_beam` 之类的新注册名。新行为通过配置字段选择，例如 `model.primary.frontend.type: paper_modal_tokenizers` 或等价字段。

理由：物理头、loss、output contract 和 focused tests 已围绕 `pinn_multimodal_beam` 建立。新增整模型名会扩大 registry surface，且没有必要。

替代方案：新增完整模型注册名。该方案会让配置更显式，但会重复物理头和输出适配逻辑，维护成本更高。

### Decision 2: 模态 tokenizer 按“复用现有 encoder 优先”实现

各模态 tokenizer 应优先复用已有 registry 组件：

- Image：`jepa_context_image`，`pooling: mean` 或其它不依赖 GPS context 的 pooler，`gps_query_pool` 不启用，`gps_condition_features` 不传入。
- CSI/RF：优先复用 `pilot_dual_view_csi` 处理 sparse pilot / masked CSI；低维 RF scan 或 compressed latent 可用 `Linear + LayerNorm`。
- Radar：复用 `radar_cnn` 或等价 2D CNN，把 range-angle / DFT map token 化。
- LiDAR：复用 `lidar_cnn` 处理已有 BEV / depth / projection tensor；不新增点云大模型。
- GPS：使用 `Linear + LayerNorm` 或现有 `gps_mlp` 的轻量等价形式。

理由：这与论文 tokenizer 思路一致，也符合项目“组件优先、少造新边界”的规则。

替代方案：为所有模态手写一套新 tokenizer。该方案可控性强，但会绕开已有 registry 和 focused tests，且容易重复现有 encoder。

### Decision 3: 图像分支要求预训练，但 smoke 可显式放宽

正式实验配置必须提供 `jepa_context_image.checkpoint_path`，并在 metadata 中记录 checkpoint、freeze policy、pooling 和是否使用 GPS context。若没有 checkpoint，只允许 debug/smoke 配置运行，并必须标记 `formal_experiment_eligible=false`。

理由：用户要求该图像编码器需要预训练；没有预训练的随机轻量 ViT 不应进入论文主结论。

替代方案：允许随机初始化参与所有实验。该方案启动方便，但会破坏“预训练 image tokenizer”这一设计前提。

### Decision 4: 共享 Transformer 输出路径 token，再进入物理头

每个启用模态输出 `[B, T, K_m, D]` 或 `[B, T, D]` token。系统为 token 加上 modality embedding、time/position embedding 和可选局部位置 embedding，拼接后进入共享 Transformer。Transformer 输出经过 pooling 或预测 horizon adapter 得到 `[B, num_pred, D]` latent，再送入 direct head 和 path head。

理由：这保留论文的 token 化多模态融合思想，同时让后续 path head/channel synthesizer 继续工作。

替代方案：只把每个 encoder 输出平均后拼接。该方案实现最小，但不足以体现本次变更的核心创新。

### Decision 5: 无线输入主线选择 sparse pilot / sparse antenna CSI

当前项目最适合的受限无线输入是 sparse pilot / sparse antenna observation：它能直接与 `csi_target` 的 `[T, Nsc, Nant, 2]` 监督对齐，并可用 observation mask 表达观测位置。wide-beam RSS 适合做低开销 beam probing 或候选排序输入，但它与阵列信道 target 的反演关系过粗，不作为当前“路径推断 - 信道重构”主线。

理由：用户的两个创新点依赖 CSI/path 物理监督，sparse CSI 观测能最小代价保持这个闭环；wide-beam RSS 更像 beamspace 功率观测，若强行做信道重构，需要更复杂的生成式先验，不适合作为当前下一步。

替代方案：完全不输入任何无线观测，只用外源传感器重构 CSI。该方案更纯粹，但难度和不确定性更高，容易把问题变成生成式信道估计。

## Risks / Trade-offs

- [Risk] `jepa_context_image` checkpoint 在本地不可用，正式配置无法直接运行。→ Mitigation：配置保留 checkpoint 路径字段，测试使用 synthetic/mock 或 debug 标记；正式实验前先确认 checkpoint 存在。
- [Risk] 模态 token 数过多导致训练显存上升。→ Mitigation：默认使用小 `hidden_dim`、少层 Transformer、token pooling/max_tokens 和 lowmem 配置。
- [Risk] sparse pilot 从 clean CSI 派生，仍是离线 proxy，不等于真实硬件采集。→ Mitigation：metadata 写明 input source、mask pattern 和 observed fraction，不把其表述成真实测量。
- [Risk] 当前 `[1, 1, 64, 2]` CSI target 信息有限，path 参数监督可能缺失或噪声较大。→ Mitigation：loss bundle 必须按 valid mask 跳过缺失字段，并把主 claim 限定为窄带阵列信道。
- [Risk] 物理分支可能不提升 Top-K，但改善 NMSE/path/beamspace 指标。→ Mitigation：实验矩阵同时报告 Top-K、NBG/beamspace、CSI NMSE 和 path metrics，不只看分类准确率。

## Migration Plan

1. 在 `pinn_multimodal_beam` 中加入可配置 paper-style tokenizer frontend，默认保留旧统计编码器用于兼容和 ablation。
2. 添加或复用 tokenizer wrapper，让 `jepa_context_image`、`pilot_dual_view_csi`、`radar_cnn`、`lidar_cnn`、`gps_mlp` 输出统一 token。
3. 增加共享 Transformer fusion、embedding 和 token pooling，输出保持 `[B, num_pred, hidden_dim]`。
4. 更新 configs：新增 paper-style sparse-pilot multimodal 配置、debug 配置和 ablation 配置；保留 oracle upper-bound 显式授权。
5. 更新 metadata、architecture summary 和 focused tests。
6. 回滚时将配置切回旧 `frontend.type: stats` 或旧物理 baseline 配置即可。

## Open Questions

- 正式实验使用哪个 JEPA checkpoint 作为默认路径，需要在本地产物和实验矩阵中确认。
- Radar/LiDAR 在 MMW 当前样本中的真实 tensor profile 是否完全匹配现有 `radar_cnn` / `lidar_cnn` 输入；若不匹配，优先添加薄 adapter，而不是新建大模型。
- 是否把 wide-beam RSS 作为后续单独 change，用于 deployable low-overhead probing baseline。
