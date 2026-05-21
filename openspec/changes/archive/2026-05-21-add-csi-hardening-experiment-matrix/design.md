## Context

项目当前 CSI 路径为 `csi_batch [B,T,Nsc,Nant,2] -> _as_complex_csi -> train_rms normalization -> PilotCSIChannelEstimator -> frequency/delay dual-view -> CSIViewTokenizer -> symmetric_gate -> internal GRU -> modular_sequence representation_core -> beam_head`。已有 `add-csi-channel-degradation` change 已实现数据侧 destructive degradation，可作为 A2 negative control，但它会降低信息上限，不适合作为主实验叙事。

`CSI模态处理对比实验.md` 要求的是 information-preserving hardening：保留 CSI-only final ceiling，同时拖慢 CSI-only 的 E50/E80/E90，并在 GPS/其它 easy modality 的 joint training 中制造可观察的模态失衡。现有代码缺口主要是：没有 `csi_hardening` 算子、`PilotDualViewCSIEncoder` 不支持 `use_internal_gru: false`、view/delay warmup、`freq_only` 或 tokenizer 容量消融，也没有统一的 hardening sweep 分析脚本。

## Goals / Non-Goals

**Goals:**

- 增加默认关闭、可复现的信息保留型 CSI hardening，并把它放在 RMS 归一化之后、pilot estimation 之前。
- 支持 CSI encoder 架构消融：内部 GRU 开关、view gate warmup、delay view warmup、frequency-only、tokenizer hidden/dropout/second-conv 配置。
- 提供按 A/B/C/D/E 分组的 CSI hard-to-learn 控制变量配置，优先覆盖文档中建议的第一批、第二批和多模态验证组。
- 提供统一分析脚本，计算 final ceiling、best、E50/E80/E90、ceiling gap、E90 ratio，并输出候选排序。
- 确保 G2D-style 验证能处理包含 `csi` 的模态集合，例如 `gps+csi`。

**Non-Goals:**

- 不移除或替换已有 `csi_degradation`；它继续作为 destructive negative control。
- 不重新生成 beam label，不用 hardening 后的 CSI 反推 beam power。
- 不引入 Sionna、TensorFlow 或其它外部信道仿真依赖。
- 不一次性实现自动大规模调度器；首版只提供配置矩阵、推荐运行顺序和离线分析脚本。
- 不把 hardening 默认应用到所有 CSI 配置；未显式启用时现有实验行为不变。

## Decisions

1. 在 CSI encoder 内实现 hardening，而不是在 dataset 侧实现。

   hardening 的推荐位置是 RMS 归一化之后、pilot estimation 之前，这个位置只存在于 `PilotDualViewCSIEncoder.forward()` 内。数据侧 `csi_degradation` 表示输入质量退化，适合 path-level dropout/noise；hardening 表示信息保留型 nuisance transform，和模型看到的归一化复信道直接相关。因此实现应放在 `src/kd_sensing/models/csi.py` 或相邻 `models/csi_hardening.py` 中。

   为了兼容文档里的 YAML 形状，`data.dataset.csi_hardening` 可作为实验级别 alias，由配置归一化阶段复制到 `model.teacher.encoders.csi.csi_hardening` 和 `model.student.encoders.csi.csi_hardening`。若 encoder 内显式配置了 `csi_hardening`，以 encoder 配置为准。

2. hardening 算子必须保留 shape 和主要信息上限。

   首版只实现四类算子：common phase、subcarrier phase slope、antenna calibration、fixed antenna permutation。它们都作用于 complex tensor `[B,T,Nsc,Nant]`，输出 shape 不变。common phase 和 phase slope 可在 training 随机，在 eval 关闭或固定；antenna calibration 和 permutation 默认由 seed 固定，避免每个 batch 改变天线语义而退化成破坏性噪声。

   备选方案是直接增强已有 `csi_degradation` profile。该方案会混淆 destructive degradation 与 information-preserving hardening，难以解释 ceiling gap，因此不采用。

3. CSI encoder 架构消融只改变学习路径，不改变 batch contract。

   `use_internal_gru: false` 时，encoder 直接输出 fused per-step features `[B,T,D]`，由外层 `single_gru` 或 `early_concat_gru` 负责时序建模。`view_gate_warmup_epochs` 在 warmup 期强制 mean fusion，`delay_view_warmup_epochs` 在 warmup 期使用 frequency-only 表示。训练循环每个 epoch 调用模型的 `set_epoch(epoch)` 或递归 helper，让 encoder 能根据当前 epoch 切换行为。

   备选方案是在 trainer 中硬编码 CSI 分支行为。该方案会让配置不可组合，且无法被单元测试直接覆盖，因此不采用。

4. 配置矩阵以显式 YAML 为主，分析脚本负责跨 run 判断。

   首版提供 `configs/csi/hardening_matrix/*.yaml` 和 `configs/fusion/csi_hardening_matrix/*.yaml`。A2 直接复用已有 `csi_degradation.profile: medium`；B/D 使用 `csi_hardening`；C 使用 encoder 架构配置。分析脚本读取输出目录下的 `train_log.json`、`final_config.yaml` 或等价 metrics 文件，不依赖训练过程中知道其它 run 的结果。

5. G2D 应按配置模态集合工作，不应假设五模态固定集合。

   项目 `MODALITY_ORDER` 已包含 `csi`，但现有 G2D 示例主要覆盖 image/radar/gps/lidar/mmwave。GPS+CSI 验证要求 teacher ensemble、confidence ranking、SMP active modality 和 diagnostics 均能接受 `csi`。实现上优先使用 `model.student.modalities` 或 `distillation.g2d.modalities` 解析 teacher 列表，保留现有五模态配置兼容。

## Risks / Trade-offs

- [Risk] hardening 强度过大导致 ceiling 下降，变成 destructive degradation。→ Mitigation：配置矩阵先跑 B5/B6 与 C1/C2，筛选条件固定为 `ceiling_gap <= 0.03` 和 `E90_ratio >= 1.5`，`final_acc` 下降超过 0.05 的 run 只作为 negative control。
- [Risk] training 随机 hardening 导致 eval 不可复现。→ Mitigation：eval 默认 off 或 fixed；fixed 模式由 seed 生成稳定参数，并在 run metadata/final config 中记录 resolved hardening。
- [Risk] warmup 行为需要 epoch 状态，漏调用会让配置无效。→ Mitigation：训练循环在每个 epoch 开始递归调用 `set_epoch`；单元测试直接验证 epoch 0 与 epoch >= warmup 的输出/aux 差异。
- [Risk] GPS+CSI G2D teacher checkpoint 缺失会阻塞 E4。→ Mitigation：E4 放在第四批；配置和错误信息必须明确缺失的 `gps` 或 `csi` teacher 来源。
- [Risk] 运行矩阵过大。→ Mitigation：配置提供完整矩阵，但文档和 tasks 明确优先顺序：先 A0/A1/A2/B3/B4/B5/B6/C1/C2，再 D1-D4，之后 E0-E4。

## Migration Plan

1. 增加 CSI hardening 配置解析、算子和 encoder 集成，默认 disabled。
2. 增加 encoder 架构开关和 epoch-aware warmup，并在 trainer 中接入递归 epoch setter。
3. 扩展 G2D 对 `gps+csi` 等任意配置模态集合的 teacher/SMP 支持。
4. 增加 CSI-only 与 GPS+CSI hardening matrix YAML。
5. 增加 sweep 分析脚本与针对性测试。
6. 使用 `conda run -n kd_mm_beam pytest ...` 运行 CSI、G2D、配置和分析脚本测试。

Rollback 策略：关闭或删除 `csi_hardening`、warmup 和 `use_internal_gru: false` 配置即可回到当前行为；由于默认关闭且不改数据文件、CSV 和 label 生成，已有实验无需迁移。

## Open Questions

- GPS 是否一定是当前 MMW Town10 数据中收敛最快的 easy modality？首版以 GPS 为默认 E 组 easy modality，同时让配置结构可扩展到 image/lidar。
- 是否需要把 hardening resolved parameters 写成独立 JSON artifact？首版可先写入 final config/run metadata，若后续做大量随机 sweep 再拆成独立 artifact。
