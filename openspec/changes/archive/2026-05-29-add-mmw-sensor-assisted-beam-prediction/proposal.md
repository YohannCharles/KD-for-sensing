## Why

上一轮 P3/V8 实验显示，`mmwave/beam_power` 相关输入和 `num_pred=1` 的强 temporal persistence 会让 beam prediction 结论变得不干净：`last_beam` baseline 过强，少样本 adaptation 也出现负迁移。需要新开一个更明确的 Multimodal-Wireless sensor-assisted beam prediction 验证路径，将模型输入限定为可部署的传感器模态 `gps+image+lidar+radar`，并用更小实验矩阵快速判断方向是否值得继续。

## What Changes

- 新增 MMW sensor-assisted beam prediction 实验契约：模型输入只允许 `gps`、`image`、`lidar`、`radar`，不得把 `mmwave`、CSI/channel、beam_power、path_params、path_descriptor 或 path label 作为 sensing input。
- 新增或调整 MMW HiST-Beam 配置，提供 `gps+image+lidar+radar` 输入 profile，继续保留 beam label 作为 supervised target，保留 beam_power/path/radio 仅用于 source auxiliary、prototype、evaluation diagnostics 或显式 baseline。
- 将快速验证矩阵限定为 `label_budget=10` 和 2 个 seeds，默认建议 seeds `[0, 1]`，以减少训练时间并便于迭代；后续只有 smoke 通过且结果有价值时再扩大到更多 budgets/seeds。
- 增加 sensor-assisted baseline 对比：至少覆盖 `v3_decoupled` source-only、`v4_adapter`、`v6_radio_proto`、`v8_path_proto`、`adapter_path_proto` 和 full fine-tuning baseline，并输出与 source-only、radio/path prototype、path condition on/off、full fine-tuning 的横向比较。
- 将 `last_beam` diagnostic baseline 显式纳入 summary，报告是否允许历史 beam 作为可比较 baseline；默认 sensor-assisted 主结论不得依赖 last-beam shortcut。
- 加强 few-shot negative-transfer 诊断：summary 必须报告 adapted-source delta、胜率、trainable ratio、adaptation time 和 leakage flags，避免只看 adapted absolute Top-K。
- 不新增本地数据或 checkpoint 到源码；LiDAR/radar 派生缓存、训练输出和实验结果仍只留在本地产物目录。

## Capabilities

### New Capabilities

- `mmw-sensor-assisted-beam-prediction`: 定义 MMW sensor-assisted beam prediction 的输入模态边界、快速验证矩阵、baseline/diagnostic 输出和负迁移判据。

### Modified Capabilities

- `hist-beam-cross-scene-adaptation`: 扩展 HiST-Beam MMW 变体验证，使其支持 `gps+image+lidar+radar` sensor-assisted 输入 profile、adapted-source delta 和 last-beam diagnostic 汇总。
- `mmw-cross-scene-adaptation-protocol`: 扩展 MMW LOSO protocol metadata，允许声明快速验证矩阵只使用 `label_budget=10` 和 2 个 seeds，并记录该矩阵不可替代完整 budget/seed sweep。

## Impact

- 影响配置：新增 `configs/hist_beam` 下 MMW sensor-assisted 快速验证配置，或扩展现有 LOSO 配置以支持 `model.modalities=[image,gps,lidar,radar]`、`budgets=[10]` 和 2 seeds。
- 影响数据加载：确认 MMW prepared split 中 `camera/gps/lidar/radar` 路径可被 dataset/loader 稳定解析，LiDAR/radar cache 策略可配置，`mmwave` 不进入 enabled modalities。
- 影响 LOSO runner 与 summary：增加 sensor-assisted matrix metadata、adapted-source delta、last-beam baseline、negative-transfer flags 和 modality profile 字段。
- 影响测试与 smoke：增加 dataset sample shape、config load、plan generation、single-run smoke 和 summary aggregation focused tests。
