## Why

现有 Radio-Semantic Prototype enhanced HiST-Beam 主要从 beam power profile 派生传播语义，这会把知识单元绑定到 codebook 响应形状，难以充分利用 Multimodal-Wireless / Sionna 已提供的 path-level 物理传播参数。现在需要将 HiST-Beam 升级为 P3-HiST-Beam，使 shared branch 学习可迁移的 path-level propagation pattern，private branch 保留 town/scenario/weather/local geometry/codebook mapping 等场景私有修正，同时严格维持 target adaptation 防泄漏边界。

## What Changes

- 新增 Multimodal-Wireless path 数据巡检能力，自动扫描 CARLA sensor、Sionna channel/path、metadata 文件，报告 town/scenario/weather、模态可用性、beam/beam_power/CSI/path-level 参数可用性，并通过 `data.field_map` 适配不同字段名。
- 扩展 MMW dataset 输出契约，在保留现有 camera/radar/gps/lidar/imu、beam、beam_power、radio_semantic_label 和 domain metadata 的基础上，新增可选 `path_params`、`path_descriptor`、`path_semantic_label`、`path_valid`。
- 新增 `PathFeatureBuilder` 与 `PathSemanticLabelBuilder`，从 Sionna path gain、delay、AoD/AoA、mask 等字段构造 path descriptor，并支持 source-only KMeans path label、rule fallback、radio_power baseline 和 coarse baseline。
- 将 HiST-Beam 模型扩展为 P3-HiST-Beam：新增 path head、可选 path descriptor regression head、path embedding，并允许 beam head 使用 `concat(c, s_star)` 或 `concat(c, s_star, e_path)`；prototype 只作为 semantic anchor/condition，不能直接输出 beam。
- 扩展 source loss、source prototype artifact、target adaptation、inference、metrics 与 diagnostics，支持 `proto_type=path`、`mu_path_c`、target-private `nu_path_s`、path semantic accuracy、path descriptor regression MSE、prototype confidence/coverage 和 V5/V6/V8 对比。
- 增强 leakage guard：在 `label_budget=0` 或 unlabeled target training 时，任何使用 target beam、beam_power、CSI/channel、path_params、path_descriptor、path_semantic_label、radio_semantic_label 作为训练监督的行为必须失败，并在 `adapt_log.json` 记录使用标志。
- 保留旧 V5 coarse prototype、V6 radio-semantic prototype、hierarchical beam loss 和 full fine-tuning baseline；新增 V8 path prototype 配置与 path condition on/off、KMeans/rule 消融配置。
- 不把 CSI/channel/path_params 作为模型输入模态；它们只允许用于 source label construction、few-shot labeled target supervision、offline evaluation 或 diagnostics。

## Capabilities

### New Capabilities

- `path-prototype-hist-beam-adaptation`: 定义 P3-HiST-Beam 的 path-level physical propagation descriptor、path semantic label、path-conditioned model forward、source/target path prototype、target-private prototype bank、防泄漏和评估诊断契约。

### Modified Capabilities

- `hist-beam-cross-scene-adaptation`: 扩展 HiST-Beam 变体矩阵、模型输出、source loss、prototype artifact、target adaptation、inference 和 metrics，纳入 V8 path-level prototype 方法，同时保持 V0/V3/V5/V6/V7 兼容。
- `radio-semantic-hist-beam-adaptation`: 将 radio-semantic prototype 明确保留为 V6 baseline 和 fallback，而不是新的 full method 主知识单元，并要求与 V8 path prototype 在 summary 中可区分。
- `mmw-cross-scene-adaptation-protocol`: 扩展 MMW manifest/inspection/split protocol，使 Sionna path-level 参数、字段映射、path availability 和 path-derived diagnostics 可追踪。
- `dataset-runtime-contracts`: 扩展 runtime flat sample 与 target provider 边界，允许可选 path descriptor/label 输出，同时保证 path/CSI/channel 不进入 sensing modality 输入。

## Impact

- 影响数据检查与数据集：`scripts/inspect_dataset.py`、`src/kd_sensing/data/datasets/*multimodal*/*mmw*`、sample manifest、collate/mask 逻辑和 runtime metadata。
- 影响 path semantics：新增或扩展 `src/kd_sensing/data/path_semantics.py` 或 `src/kd_sensing/utils/path_semantics.py`，以及 KMeans/scaler artifact 保存与复用。
- 影响模型与 loss：`src/kd_sensing/models/hist_beam.py`、HiST-Beam output contract、source training loss、path regression loss 和 shared/private regularization 配置。
- 影响 prototypes 与 adaptation：`src/kd_sensing/utils/prototypes.py`、target adaptation loop、private adapter 更新边界、target-private prototype bank 和 `adapt_log.json`。
- 影响评估与实验矩阵：HiST-Beam LOSO runner、metrics、diagnostics、summary、`configs/hist_beam/exp_v8_*.yaml` 和 smoke tests。
- 影响安全边界：`src/kd_sensing/utils/leakage_guard.py` 或等价训练访问守卫；不得提交本地数据、Sionna path 文件、缓存、checkpoint 或实验输出。
