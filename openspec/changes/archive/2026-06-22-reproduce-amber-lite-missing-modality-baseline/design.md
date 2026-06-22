## Context

AMBER 类论文关注缺失模态和退化模态下的鲁棒 beam prediction，但当前仓库的强对照主要是 clean 或评估期扰动。AMBER-lite 的目的不是完整复刻外部论文所有细节，而是提供一个可审计、最小、可训练的 full-multimodal missing-modality baseline。

现有项目已有 `modular_sequence`、多模态 encoder、token-aware core、difficulty profile 和 P0-P5 评估能力。第一版应复用这些能力，新增最少的 missing-mask 和 modality-dropout glue。

## Goals / Non-Goals

**Goals:**

- 支持 image、LiDAR、radar、GPS 四类模态的 missing-modality baseline。
- 训练时支持可配置 modality dropout，并在 metadata 中记录 dropout policy。
- forward 时能接收 modality availability/mask；缺失模态使用可学习或固定 mask token。
- 输出 clean 与 missing/degraded condition-level DBA summary，并可与当前 heatmap 合并。

**Non-Goals:**

- 不宣称完整 AMBER 官方复现或论文数值。
- 不新增外部依赖。
- 不复制训练循环或 dataset loader。
- 不把普通 baseline 变成必须消费 reliability metadata。

## Decisions

1. 采用 component baseline 路径。
   - 使用现有 `modular_sequence`，新增或复用 encoder/projector/core/head 组件。
   - 如现有 core 不支持 mask token，则新增一个窄 `missing_modality_token_transformer` representation core。

2. missing mask 是显式 opt-in metadata。
   - batch/runtime 可提供 modality availability/missing mask。
   - 只有 AMBER-lite 配置声明消费该 metadata 时才传给模型；普通 ResNet+GPS、JEPA 和 GPS-only 不受影响。

3. modality dropout 放在训练 transform/helper 层。
   - 训练 profile 随机 drop 模态输入并写出 dropout rate、seed、affected modalities 和 digest。
   - evaluation condition 使用确定性 missing/degraded profile，保持 target contract 不变。

4. summary adapter 对齐 P0-P5。
   - 输出 model、source、overall_clean、condition metrics、overall mean、strict comparability、dropout policy 和 missing-mask provenance。

## Risks / Trade-offs

- [Risk] mask token 与现有 batch shape 不匹配。→ Mitigation: synthetic forward test 覆盖全模态、单模态缺失和多模态缺失。
- [Risk] 训练 dropout 和评估 perturbation 混淆。→ Mitigation: metadata 分开记录 train dropout policy 和 eval condition digest。
- [Risk] LiDAR/radar 数据在本地不可用。→ Mitigation: config load 和 synthetic tests 不依赖真实数据；真实运行缺模态时标记 unavailable。
- [Risk] AMBER-lite 被误写成 AMBER 完整复现。→ Mitigation: docs、metadata 和 claim registry 均标记 `lite/local reproduction`。
