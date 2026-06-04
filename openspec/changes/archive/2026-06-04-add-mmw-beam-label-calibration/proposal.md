## Why

MMW Town10 的原始 64-beam label 是 channel-derived power vector 的 `argmax`，但从 GPS 角度与 raw beam label 对应关系看，不同 town/scene 存在方向、offset 和 0/63 边界语义不一致的问题。该问题会污染 source/target label 分布诊断、soft beam label 邻接关系、history anchor、coarse/fine 分组和 DBA 等距离敏感指标，因此需要把 MMW raw beam ID 与训练使用的 calibrated beam label space 显式区分。

## What Changes

- 新增 MMW beam label calibration 能力，支持基于配置的 64-class label 映射，例如 `mapped = (direction * raw + offset) mod num_classes`，并允许按 scene/town 指定映射参数。
- MMW dataset 在启用 calibration 时 MUST 将 `input_beam`、`target_beam`、显式 `future_beam_label*`、beam label cache 和 metadata 中的训练标签统一映射到 calibrated label space。
- 64 维 class-indexed 监督分布（例如 `target_beam_distribution`、`beamspace_power_label`）在作为训练监督或评估诊断消费时 MUST 按同一映射重排 class 维；传感器输入 `mmwave` 的原始 power vector 不作为 sensing feature 重排。
- 训练、评估、预测导出、分布诊断和 GPS-angle 可视化 MUST 在 metadata 中记录当前 label space，并在需要人类审计时保留 raw label 与 calibrated label 的可追溯关系。
- 默认行为保持 raw label space，不改变现有 DeepSense6G、Raymobtime、非 MMW dataset 或未显式启用 calibration 的 MMW 配置。

## Capabilities

### New Capabilities
- `mmw-beam-label-calibration`: 定义 MMW raw beam label 到 calibrated beam label space 的配置、数据集返回字段、分布重排、metadata 和评估诊断契约。

### Modified Capabilities
- `mmw-town10-dataset-preparation`: MMW prepared artifacts 需要记录 raw beam label、calibration 候选信息和可审计 metadata，不直接覆盖原始 power-vector 语义。
- `modality-aware-data-loading`: MMW dataset 在启用 calibration 时返回 calibrated hard labels，同时保持模态按需读取和张量 shape 稳定。
- `soft-beam-label-training`: soft target 生成需要使用 calibrated label topology 或按映射重排分布，避免 hard label 与 soft distribution class 顺序不一致。
- `beamspace-physical-labels`: beamspace physical label 作为 class-indexed 分布消费时需要匹配 calibrated label space，并保留 raw power vector 来源诊断。
- `beam-distribution-shift-diagnostics`: 分布诊断和图表需要声明 raw/calibrated label space，并支持基于 calibrated label 计算 histogram 与 ordered/circular distance。

## Impact

- 影响 MMW dataset、MMW preparation metadata、beam label cache、soft beam target 构造、beamspace physical label 构造/缓存、evaluation metrics metadata、prediction/diagnostic artifact 写出和分布诊断脚本。
- 需要新增配置字段和 runtime metadata 字段；默认关闭，因此现有配置、checkpoint 和 raw-label artifact 仍可复现。
- 不新增外部依赖；所有 Python 验证继续使用 `conda run -n kd_mm_beam`。
- 启用 calibration 后的新训练结果不应与旧 raw-label checkpoint 直接混比，除非评估报告明确执行 inverse mapping 或声明 label space。
