## Context

项目当前已经具备 CSI 一等模态基础：`src/kd_sensing/data/transform_ops/csi.py` 能从 `.npy/.npz` 读取 complex 或 real/imag CSI，也能从 MMW `_paths` payload 的 path gain 与 AoD 派生等效 channel；`DeepSense6GDataset`/`MMWDataset` 按 `csi1..csiN` 懒加载历史 CSI；`PilotDualViewCSIEncoder` 在模型内执行 RMS 归一化、pilot-based estimation noise、frequency/delay 双视图编码。

`CSI模态加噪方案.md` 的核心不是普通 augmentation，而是把 MMW ray tracing 产生的高质量 path-level CSI 退化成更接近真实 estimated CSI 的输入。这个退化应发生在数据输入侧，并且必须保持现有监督标签、split、RMS normalizer 和 batch shape 契约稳定。

## Goals / Non-Goals

**Goals:**

- 为 MMW/CSI dataset 增加默认关闭的 `csi_degradation` 配置，支持 clean、medium、hard profile 和参数覆盖。
- 对 MMW path-level payload 优先执行物理含义明确的退化：复增益 AWGN、弱路径优先 dropout、主径衰减、delay/angle 扰动、天线相位误差、历史 CSI temporal shift。
- 保证退化后模型输入仍是现有 `[T, Nsc, Nant, 2]` real/imag 张量，训练、评估和 fusion batch 准备不需要新张量契约。
- 保证可复现：同一配置、seed、split 和样本在重复运行中得到相同 degraded CSI，并记录有效参数。
- 保持 CSI RMS 基于 clean 训练 split，future beam 标签保持 clean 监督来源。

**Non-Goals:**

- 不替换 `PilotCSIChannelEstimator`；模型内 pilot estimation noise 继续用于估计噪声消融。
- 不重新生成 beam label，不用 degraded CSI 反推新的 beam power。
- 不要求 MMW preparation 预先写出 degraded CSI 文件；首版以 runtime transform 为主。
- 不新增 Sionna、TensorFlow 或其它外部依赖。
- 不把所有加噪方案一次性扩展到 image、LiDAR、GPS 或 mmWave power vector。

## Decisions

1. 在 `transform_ops.csi` 增加退化配置和纯 NumPy transform，而不是在模型 encoder 内实现。

   数据退化代表输入质量，而 `PilotCSIChannelEstimator` 代表模型侧从 clean CSI 到 channel estimate 的估计过程。把 MMW path-level 的 delay、AoA/AoD、path dropout 放在 loader 侧，才能在派生等效 channel 前破坏几何传播结构；模型 encoder 继续只接收 real/imag CSI 张量。实现上增加 `CSIDegradationConfig`、profile resolver、`degrade_csi_payload()`/`degrade_csi_sequence()` 等小函数，并让 `read_csi_tensor()` 或 `load_csi_sequence()` 接收可选 degradation context。

   备选方案是在 `PilotDualViewCSIEncoder` 增加更多噪声参数。该方案可以处理 tensor-level AWGN，但无法自然处理 path-level delay/angle/dropout，也会让数据质量实验与模型结构绑定。

2. 默认保持 clean 行为，显式配置才返回 degraded `csi`。

   `data.dataset.csi_degradation.enabled` 默认为 `false`。未开启时，`load_csi_sequence()`、dataset 样本字段、现有配置和测试保持当前行为。开启时，样本字段仍命名为 `csi`，但语义变为“模型实际输入 CSI”，可由 metadata 标记为 degraded。这样不需要修改 `prepare_csi_inputs()` 和 modular model 的调用约定。

   备选方案是同时返回 `csi_clean` 和 `csi_degraded`。这更显式，但会牵动 batch、fusion、可视化和训练入口。首版可以通过 internal clean cache 支持 RMS，不暴露额外样本字段。

3. RMS normalizer 必须始终基于 clean CSI，并在 dataset 内分离 clean cache 与 degraded cache。

   `DeepSense6GDataset._prepare_csi_rms_normalizer()` 应继续扫描 clean sequence；开启 degradation 时，取样阶段再生成 degraded CSI。若继续复用当前 `_csi_cache` 作为唯一 cache，RMS 可能意外基于 degraded 输入。实现应明确区分 clean loader path 和 degraded loader path，或在 fit RMS 时强制禁用 degradation。

4. temporal shift 只重排当前历史窗口内的 CSI 路径，默认边界 clamp，不读取未来 CSI。

   现有 MMW sequence CSV 只有 `csi1..csiN` 历史列，future 侧只有 beam label/power 路径。为了遵守“不包含未来 CSI”的现有契约，`temporal_shift_choices` 只在历史列表内移动：目标位置越界时默认使用最近边界帧，也可后续扩展为 zero fill。这样 `±1` 或 `±2` shift 能模拟同步误差，同时不会引入未来信息泄漏。

   备选方案是扩展 sequence CSV 写出 `future_csi*`。这会增加 preparation 和数据泄漏风险，不适合作为首版默认。

5. 使用 profile 加参数覆盖的配置模型。

   推荐内置 profile：

   - `clean`: 所有退化关闭。
   - `medium`: SNR 10 dB、path dropout 20%、AoA/AoD noise 3 度、delay noise 0.5 ns、antenna phase error 10 度、temporal shift 从 `[-1, 0, 1]` 采样。
   - `hard`: SNR 5 dB、path dropout 30%、dominant path attenuation 0.5、AoA/AoD noise 5 度、delay noise 1 ns、antenna phase error 20 度、temporal shift 从 `[-2, -1, 0, 1, 2]` 采样。

   用户可通过 YAML 覆盖单项参数。profile resolver 必须把最终参数写入 diagnostics，避免实验只记录 profile 名而丢失实际配置。

6. 随机性以 base seed、split、样本 index 和路径摘要派生。

   退化应使用 `np.random.default_rng()`，seed 来自 `data.dataset.csi_degradation.seed` 或实验 seed，再结合 split、dataset index 和历史 CSI 路径摘要。这样多 worker dataloader、重复评估和 lazy loading 顺序变化不会改变同一样本的退化结果。训练期如需 epoch-varying augmentation，应作为后续扩展显式增加 `vary_by_epoch`，首版不默认启用。

7. path-level 优先，tensor fallback 可诊断。

   对含 path gain、delay、AoA/AoD 的 payload，先在 path-level 应用 dropout、主径衰减、delay/angle 扰动，再派生等效 channel，最后应用 tensor-level AWGN 和 antenna phase error。若 payload 只有 complex channel tensor，则只能应用 tensor-level AWGN 和 antenna phase error；不可执行的 path-level 算子必须记录为 skipped，而不是静默声称已执行。

## Risks / Trade-offs

- [Risk] MMW channel payload 字段名和 shape 不完全统一。→ Mitigation：沿用当前 `_first_present()` 字段候选策略，并把实际字段、skipped operators 和失败原因写入 diagnostics。
- [Risk] 退化过强会把 CSI 变成低信息模态，削弱“高上限但难学”的实验目标。→ Mitigation：默认主实验使用 medium profile，hard profile 仅作鲁棒性实验，severe 参数不作为内置默认。
- [Risk] 多 worker dataloader 下随机结果不稳定。→ Mitigation：不用全局 RNG，所有随机数从样本稳定 key 派生。
- [Risk] runtime degradation 增加数据加载开销。→ Mitigation：默认关闭；启用时复用 per-dataset cache，并保留后续增加 degraded cache 文件的空间。
- [Risk] temporal shift 边界 clamp 会重复边界 CSI。→ Mitigation：记录 shift 与 fill mode；由于首版禁止读取未来 CSI，这是可接受的保守近似。

## Migration Plan

1. 增加 CSI degradation 配置解析和纯函数实现，默认 disabled。
2. 修改 CSI sequence loader 和 dataset CSI 加载路径，确保 clean RMS 与 degraded sample 输出分离。
3. 将 `csi_degradation` 从 dataset YAML/data_factory 传入，并在 metadata 中记录 profile、seed 和有效参数。
4. 增加 `configs/csi/*degraded*.yaml` 和至少一个 fusion degraded CSI 示例配置。
5. 增加针对退化算子、确定性、temporal shift、RMS clean 统计和配置加载的测试。
6. 使用 `conda run -n kd_mm_beam pytest tests/test_csi_modality.py tests/test_training_io_workflow.py -q` 运行针对性验证。

Rollback 策略：删除或关闭 `data.dataset.csi_degradation` 配置即可恢复 clean CSI；由于默认关闭且不改 CSV/label 格式，已有实验配置不需要迁移。

## Open Questions

- 是否需要把 degraded CSI 持久化为 cache 文件以降低大规模训练开销？首版先 runtime transform，等真实训练 profiling 后决定。
- 是否需要 epoch-varying degradation 作为数据增强？首版优先可复现固定退化，后续可加显式开关。
- MMW 真实 `_paths` 文件中 delay/AoA/AoD 字段名称是否覆盖当前候选列表？实现后需要用真实 zip 做一次 smoke 检查。
