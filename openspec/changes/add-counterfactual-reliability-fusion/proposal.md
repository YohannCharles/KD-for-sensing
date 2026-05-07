## Why

当前项目已经支持 image、radar、GPS、LiDAR、mmWave 的单模态与可配置 fusion 实验，但现有 fusion 主要使用早期拼接后 GRU/attention 的固定结构。这个结构默认各模态同等可信，难以处理“某些模态单独训练有效、融合后反而拖累”或训练中模态贡献不稳定的失衡问题。

`模态失衡改进方案1.md` 提出的 CRAF 方向适合解决这个问题，但原方案按从零项目设计目录和训练入口。现在需要把它收敛为符合本仓库 registry、canonical config、DeepSense6G dataset、KD distiller 和统一训练/评估流程的增量改造。

## What Changes

- 新增 CRAF fusion 模型族：按固定模态顺序将启用模态编码为 token，加入时间/模态 embedding，通过可靠性估计器输出每样本每模态 gate，再送入 Transformer fusion 和 horizon prediction head。
- 新增可靠性相关模块：模态 token projector、tokenizer、单模态辅助 head、entropy/margin confidence、dataset-level reliability prior、reliability gate 和可解释输出字段。
- 新增反事实训练能力：在训练阶段可按配置进行 modality dropout / leave-one-out 或 sample-one counterfactual forward，用性能差异监督可靠性 gate，并保留普通 no-KD 训练路径。
- 新增 beam-aware soft label loss 和序列 CE/per-sample loss helper，使 future beam prediction 可同时使用硬标签与邻近 beam 软标签。
- 新增 CRAF baseline 配置：single-modal transformer、token transformer fusion、early concat transformer/GRU 和 CRAF no-KD 配置，优先覆盖当前可用的 canonical fusion 模态集合。
- 扩展训练、验证、日志与测试：支持模型返回 dict 或现有三元组，记录 reliability、effective modality mask、counterfactual loss 和单模态辅助 loss，并补充 smoke/unit 测试。
- 不复制 `train.py`、`eval.py`、`models/`、`losses/` 的从零目录结构；所有实现放入 `src/kd_sensing` 现有包、注册表和配置体系。

## Capabilities

### New Capabilities

- `counterfactual-reliability-fusion`: 描述 CRAF 模型、可靠性估计、反事实 gate 监督、beam-aware loss、缺失/强制 drop 模态 mask、可解释日志与 baseline 对比的行为契约。

### Modified Capabilities

- `configurable-multimodal-fusion`: 允许 fusion 配置选择 CRAF/Transformer 类 fusion 模型，并要求其复用现有 `modalities` 标准化、batch 输入准备和 canonical 配置语义。
- `experiment-workflow`: 训练和评估流程需要支持 CRAF 的 dict 输出、附加 loss、counterfactual 训练配置、可靠性日志和 smoke test。
- `component-registry`: 新增的模型、loss 和可选训练 helper 必须通过现有注册表或窄模块入口接入，保持扩展方式一致。

## Impact

- 影响代码：`src/kd_sensing/models/fusion/`、`src/kd_sensing/distillation/losses.py` 或新增 loss helper、`src/kd_sensing/engine/trainer.py`、`src/kd_sensing/engine/validator.py`、`src/kd_sensing/engine/batch.py`、`src/kd_sensing/config/canonical.py`、相关配置和测试。
- 影响配置：新增 CRAF/baseline fusion 配置字段，例如 `model.student.type: craf_fusion`、`model.student.reliability`、`training.modality_dropout`、`training.counterfactual`、`loss.beam_soft`。
- 影响输出：训练日志、TensorBoard 和 metrics 可记录 reliability gate、counterfactual loss、unimodal auxiliary loss、模态 drop 统计等可解释信息。
- 兼容性：默认 image-only、radar-only、GPS-only、LiDAR-only、mmWave-only 和既有 `fusion_teacher`/`fusion_student` 配置必须保持现有行为；CRAF 仅在显式配置时启用。
- 依赖：优先使用 PyTorch 现有 `TransformerEncoder`、`MultiheadAttention` 和项目已有依赖，不新增外部运行时依赖。
