## Context

现有项目已经有 `MMWDataset`、CSI loader/encoder、beamspace physical label、MMW preparation、`modular_sequence`、registry、统一训练/评估 CLI 和 `ModelOutput` 适配。用户请求的 PINN 框架跨越数据监督、模型、loss、metric、配置和文档；如果按附件中新建根目录 `data/ models/ train.py`，会违反当前项目的包结构、入口和退役 thin alias 约束。

本设计把该需求收敛为一个可审计的 MMW physics-informed baseline：它不是替代现有 sensor-assisted 主线，而是一个显式使用 CSI/path/beam-power 物理监督的 research baseline。当前完整 CSI 被命名为 `csi_target`，默认只用于监督；模型输入只能接收 `csi_input`，且必须是历史、部分、噪声、压缩等受限观测。任何使用当前完整 CSI 作为输入的 run 只能作为 oracle upper-bound，并必须在 metadata 中标记为不可进入 `mmw-sensor-assisted-beam-prediction` 主结论。

## Goals / Non-Goals

**Goals:**

- 在 `src/kd_sensing` 内实现可训练的 physics-informed MMW beam baseline，复用现有 dataset、batch、registry、训练、评估和输出产物边界。
- 支持“多模态编码 -> 融合 latent -> 主导路径参数预测 -> 可微信道合成 -> direct/physics hybrid beam logits”的闭环。
- 对 CSI、path 参数、beam power/beamspace label 缺失保持可选监督：缺字段时 mask 对应 loss，不让训练崩溃。
- 记录关键 shape、物理监督来源、sensitive usage flags、main-conclusion eligibility、模型架构摘要和可比较实验口径。
- 用 synthetic/focused tests 验证 forward、loss、backward、complex autograd 和 missing-field 行为，不依赖真实 `dataset/`。

**Non-Goals:**

- 不新增仓库根 `train.py`、`evaluate.py`、`scripts/inspect_dataset.py` 或新的旧式入口。
- 不复制通用训练循环、dataset 全量解析逻辑、checkpoint 管理或 metric 聚合框架。
- 不把当前完整 CSI/path/beam-power 当作 MMW sensor-assisted 主结论的 sensing input。
- 不在第一版实现 UPA、真实厂商 codebook 全覆盖、Hungarian path matching、多阶段预训练或完整论文 Table 复现。
- 不提交真实数据 inspection 输出、cache、logs、checkpoint 或 `outputs/` 产物。

## Decisions

1. **路径归类：使用 whole-model exception，而不是强塞进 `modular_sequence`。**

   `modular_sequence` 适合 encoder/projector/core/head 组合，但该模型需要 forward 内部同时输出 path 参数、合成 `H_hat`、计算 physics logits，并把多个中间物理张量交给 loss/diagnostics。把这些塞成 head 会让 core/head 边界承担信道合成职责。新增 `@MODELS.register("pinn_multimodal_beam")` 更小、更可测，前提是保留 `adapt_model_output` 兼容和 focused tests。

   备选：只新增 representation core + head。放弃，因为 `H_hat`、path mask、physics logits 和 direct logits 之间耦合较强，组件化后反而需要更多胶水配置和 runtime 分支。

2. **数据层只加窄 adapter，不新增 `MultimodalWirelessDataset` 平行实现。**

   现有 `MMWDataset` 已处理 scene、condition、split、modality availability、beam labels、CSI 和 beamspace physical label。新增 `kd_sensing.data.datasets.mmw_physics_adapter` 或等价窄 helper，把已有字段规范化为顶层 `csi_input`/`csi_target`、`beam_label`、`beam_power`、结构化 `path_params` 以及 `physics_targets` 中的 supervision 字段、shape summary 和 unavailable reasons。

   备选：新建 `MultimodalWirelessDataset`。放弃，因为会重复 CSV/sequence/cache/calibration 逻辑，且容易绕过现有 leakage guard。

3. **CSI 格式采用现有末维 real/imag `[B, T, Nsc, Nant, 2]`，但语义上拆成 `csi_input` 和 `csi_target`。**

   `csi-channel-data` 已规定 dataset 返回 `[T, Nsc, Nant, 2]`。`csi_target` 是当前完整 CSI，仅用于 reconstruction loss、beam label/beam gain 监督和评估；`csi_input` 是可选受限观测，可来自 history、partial、noisy、compressed 或显式 oracle full。模型/loss 在边界处使用 `ri_to_complex`，内部物理模块使用 `torch.complex`、`torch.exp` 和 broadcast，保持 autograd。

   备选：改成 `[B, 2, Nr, Nt, K]`。放弃，因为会破坏现有 CSI loader/encoder 契约，需要大量 reshape 分支。

4. **第一版物理模块只实现 ULA 和窄带/多子载波 path synthesis 的最小闭环。**

   新增 `kd_sensing.models.physics.array_response`、`channel_synthesizer`、`beam_scoring` 和 `complex_utils`。ULA 默认使用 `antenna_spacing_ratio * wavelength`，路径参数使用弧度标准化；`H_hat` 输出采用 `[B, T_or_H, Nsc, Nr, Nt]` 或在缺 Rx 维时 `[B, T_or_H, Nsc, Nant]` 的 complex tensor，并由 loss adapter 对齐到 dataset CSI shape。

   备选：同时支持 UPA、2D AoA/AoD 和真实 codebook。推迟；没有稳定数据字段前会扩大配置和测试面。

5. **loss 做成 opt-in physics bundle，普通 beam loss 不变。**

   新增 `physics_informed_beam_loss` 或训练 extension helper，读取 `ModelOutput.diagnostics` 与 batch auxiliary。总损失为 beam CE 加权叠加 `csi_reconstruction`、`path_consistency`、`array_consistency`、`beam_power_distribution` 和可选 `alignment`。所有分量都有 `enabled/weight/mask`，缺少 target 时返回 0 张量和 diagnostic reason。

   备选：修改现有 beam loss 主体。放弃，普通 supervised baseline 不应承担物理监督字段。

6. **inspection 作为包内 CLI 或 train debug summary，不建根脚本。**

   新增 `kd-sensing-inspect-dataset` 或 `python -m kd_sensing.cli.inspect_dataset`，也允许训练 debug 首 batch 写出 shape summary。输出只打印/写入当前 run metadata，不生成源码文件。

   备选：按用户附件创建 `scripts/inspect_dataset.py`。放弃，因为项目 spec 已删除 Python thin alias 支持。

7. **配置放在现有 fusion 目录，用 overlay 表达 ablation。**

   新增 `configs/fusion/physics_informed_mmw_hybrid.yaml`、`physics_informed_mmw_debug.yaml`、`physics_informed_mmw_{vision_only,partial_csi_multimodal,history_csi_multimodal,oracle_full_csi}.yaml` 和少量 ablation overlay。配置使用 `data.dataset.type: mmw`、显式 `data.use_csi_input`/`data.csi_input_mode`/`allow_oracle_full_csi_input`、`physical_label`/`path_semantic` 开关、`model.primary.type: pinn_multimodal_beam` 和 `loss.physics.*` 权重。

   备选：新增根 `configs/default.yaml`。放弃，因为当前仓库按 modality/fusion/diagnostics/preprocess 分类维护配置。

8. **实验结论先标记为 pending/supplementary，真实 claim 需后续跑数。**

   文档只记录运行方法、ablation matrix、metadata 字段和 claim eligibility，不写“效果更好”的结论。`docs/result_claims_registry.md` 中该 baseline 初始状态为 pending/unverified。

## Risks / Trade-offs

- **真实 MMW path payload 字段不稳定** -> adapter 支持字段映射、key diagnostics 和 unavailable mask；第一版只要求 synthetic tests 与真实数据 inspection smoke。
- **complex autograd/shape 对齐出错** -> 物理模块保留最小 ULA 公式，增加 synthetic finite-gradient/backward tests 和 CSI NMSE shape tests。
- **sensor-assisted 主结论被 oracle 污染** -> dataset/forward 边界强制 `csi_target` 不进模型；`oracle_full` 必须显式 `allow_oracle_full_csi_input=true` 并打印 warning；metadata 强制记录 `csi_input_mode`、`used_current_full_csi_as_input`、`used_path_label_for_training`、`used_beam_power_for_training` 和 `main_conclusion_eligible`。
- **whole-model exception 扩大 registry surface** -> 只新增一个注册名、一个模型文件和 focused architecture summary test；不新增兼容 alias。
- **物理 loss 权重导致训练不稳定** -> debug 配置默认小 batch/短 epoch，physics 权重可关；loss diagnostics 输出每个分量和可用样本数。
- **真实 codebook 缺失影响 physics logits** -> 第一版使用现有 DFT/ULA helper 或 AoD-bin fallback，metadata 记录 `codebook_source`，不把 fallback 结果升级为主 claim。

## Migration Plan

1. 先实现 physics helper、adapter、loss 和 synthetic tests，保证不触碰真实数据。
2. 注册 `pinn_multimodal_beam` 并加入默认组件导入、config load、model summary 和 architecture boundary tests。
3. 增加 MMW physics debug/canonical/ablation 配置和包内 inspection CLI/help smoke。
4. 更新 README 与四份主线实验文档，明确运行命令、claim 状态、sensitive usage 和产物边界。
5. 回滚时删除新增模型注册、配置和 docs 入口；现有 MMW/CSI/fusion 配置不依赖该注册名，普通训练路径不受影响。

## Open Questions

- 真实 Multimodal-Wireless path payload 的字段名、单位和 shape 是否稳定；实现时以 inspection 输出为准补充 field map。
- `H_hat` 是否需要同时建模 Tx/Rx 二维阵列；第一版默认 ULA，若真实数据强依赖 UPA，再单独扩展。
- physics logits 使用真实 codebook、现有 DFT codebook 还是 AoD-bin fallback；第一版按配置记录来源并允许 fallback。
