## Context

当前 MMW HiST-Beam/P3 路径已经具备 Town10 scenario-LOSO runner、MMW prepared split、image/GPS/mmWave 输入、LiDAR/radar 数据读取、radio/path prototype、target leakage guard 和 summary 输出。上一轮完整矩阵显示：V8 path prototype 相比 path-condition-off 有明显收益，也比 full fine-tuning 更省参数；但 V8 top1 未超过 V6 radio-semantic，`label_budget=0` 与少样本 adaptation 存在负迁移，并且 `last_beam` diagnostic baseline 在 `num_pred=1` 下过强。

本 change 的目标不是继续调 path prototype，而是建立一个更干净的 sensor-assisted beam prediction 快速验证面：模型输入限定为可部署传感器 `image/gps/lidar/radar`，不再让 `mmwave` 或 beam power profile 作为 sensing input。beam label 仍作为 supervised target，beam_power/path/radio 只允许用于 source auxiliary、prototype、offline evaluation 或 diagnostics。

本地 MMW prepared CSV 已包含 `camera1..8`、`gps1..8`、`lidar1..8`、`beam1..8`、`future_beam*` 等字段；radar maps 已可由 `_ensure_radar_columns` 派生为 `radar1..8`。实际 sample smoke 已确认 `image [8,3,224,224]`、`gps [8,3]`、`lidar [8,3,224,224]`、`radar_ra/radar_da [8,128,64]` 能被 dataset 返回。

## Goals / Non-Goals

**Goals:**

- 新增 MMW sensor-assisted HiST-Beam 配置，默认输入为 `image`、`gps`、`lidar`、`radar`。
- 禁止 `mmwave`、CSI/channel、beam_power、path_params、path_descriptor 和 path labels 进入 sensor-assisted 模型输入。
- 快速验证矩阵只使用 `label_budget=10` 和 2 个 seeds，默认 seeds `[0, 1]`，减少训练时间。
- 在 summary 中输出 adapted-source delta、negative-transfer flags、last-beam diagnostic baseline、trainable ratio、adaptation time 和 leakage flags。
- 保留 V6 radio 和 V8 path prototype 作为 auxiliary/prototype baseline，但要求 run metadata 明确它们是否使用了 beam_power/path label 作为输入或监督。
- 提供 smoke/quick validation，先证明 `gps+image+lidar+radar` 数据路径、模型 forward 和小矩阵 runner 可用。

**Non-Goals:**

- 不把 LiDAR/radar 预处理产物、cache、训练输出或 checkpoint 纳入源码提交。
- 不在本 change 中扩大到 budgets `[0,5,10,20,50]` 或 3+ seeds；完整矩阵留给后续确认方向后再开。
- 不声称 sensor-assisted 配置已解决 leave-one-town-out 或 weather-shift；当前本地 ready data 仍主要支持 sunny/Town10 scenario-LOSO。
- 不移除既有 `image+gps+mmwave` 配置或 V6/V8 路径，只新增更干净的对照 profile。
- 不默认使用 target path/radio labels 做 few-shot target supervision，除非显式配置并被 leakage guard 记录。

## Decisions

1. **新 profile 固定为 `image+gps+lidar+radar`，而不是在旧 MMW config 上临时覆盖。**

   这样 runner、summary 和 metadata 能明确区分 sensor-assisted 结果与旧 `image+gps+mmwave` 结果，避免把 beam_power-like 特征混入主结论。备选方案是用命令行 `-o model.modalities=...` 覆盖旧 config，但历史结果和新结果容易混淆。

2. **快速矩阵只跑 `budget=10` 和 2 个 seeds。**

   上一轮实验已经证明 0/5 labels 容易负迁移，且四卡矩阵耗时较高。先用 `budget=10` 判断传感器输入是否带来有意义的 source/adaptation 性能；若结果为正，再扩大预算和 seed。备选方案是直接保留 `[0,5,10] x 3 seeds`，但反馈周期太慢。

3. **`last_beam` 保留为 diagnostic baseline，而不是默认模型输入或主比较方法。**

   `last_beam` 在短 horizon 下接近饱和，能暴露任务的 temporal persistence，但不适合作为 sensor-assisted 主模型输入。summary 必须报告它，论文式结论必须说明是否允许历史 beam。后续 sensor-assisted 配置采用 `seq_len=5`、`num_pred=3`，用较短历史窗口和更长预测 horizon 弱化 last-beam shortcut。备选方案是删除该 baseline，但会掩盖任务难度和潜在 shortcut。

4. **V6/V8 在 sensor-assisted profile 中只能使用 radio/path 作为 auxiliary/prototype/diagnostics。**

   V6 radio label 可由 source beam_power 派生，V8 path label 可由 source path 派生，但这些字段不得成为 sensing input。target adaptation 的使用必须由 leakage flags 记录，尤其是 `used_target_beam_power_for_training`、`used_target_path_label_for_training` 和 `used_target_radio_label_for_training`。备选方案是完全禁用 V6/V8，但会失去与上一轮方法的桥接对照。

5. **LiDAR/radar 性能风险通过 cache 和 smoke 控制。**

   LiDAR BEV 投影和 radar map 读盘会显著增加 I/O。首版配置应支持 `lidar_cache_policy`、`lidar_use_cache`、`lidar_write_cache`、`data.dataloader.num_workers` 和 CPU thread overrides，并先跑 single-run smoke。备选方案是只用 `gps+image`，但无法验证用户希望的 LiDAR/radar 辅助价值。

## Risks / Trade-offs

- [Risk] LiDAR/radar I/O 让训练吞吐明显下降。→ Mitigation：提供 smoke 配置、缓存开关、低 worker 默认和后续 throughput change 的复用点。
- [Risk] `radar` 派生 CSV 或 cache 不完整导致部分 scenario 失败。→ Mitigation：在 dataset smoke 中校验三类 ready scenario 的 `radar_ra/radar_da` shape，并在 plan preflight 中报告缺失。
- [Risk] 少样本 adaptation 继续负迁移。→ Mitigation：summary 强制输出 adapted-source delta 和 negative-transfer flags，不把 adapted absolute top1 当成唯一成功标准。
- [Risk] V6/V8 auxiliary supervision 被误解为输入泄漏。→ Mitigation：sensor-assisted spec 明确输入边界，run metadata 输出 enabled modalities 与 sensitive-field usage flags。
- [Risk] `budget=10`、2 seeds 结论方差较大。→ Mitigation：该 change 只声明 rapid validation；若方向有效，再新开完整矩阵。
- [Risk] `last_beam` baseline 远高于 sensor-only model。→ Mitigation：报告为 diagnostic，后续可考虑更长 horizon 或禁用历史 beam 信息的任务定义。

## Migration Plan

1. 增加 sensor-assisted YAML 和 smoke YAML，默认 `model.modalities=[image,gps,lidar,radar]`、`budgets=[10]`、`seeds=[0,1]`。
2. 增加 dataset/config focused tests，验证 MMW sample 返回 image/GPS/LiDAR/radar shape，且 `mmwave` 不在模型输入中。
3. 增强 LOSO plan/summary metadata，记录 modality profile、budgets/seeds、last-beam baseline、adapted-source delta 和 negative-transfer flags。
4. 跑单 run smoke，再跑 2-seed budget10 快速矩阵。
5. 根据结果决定是否扩展到更长 horizon、更多 budgets/seeds 或 sensor modality ablation。

## Open Questions

- 是否将 `seed` 默认设为 `[0, 1]` 还是 `[1, 2]`？首版建议 `[0, 1]`，与现有配置惯例一致。
- `last_beam` 是否应该在 sensor-assisted 报告中作为不可比 diagnostic，还是另开一个 explicit temporal baseline variant？首版只做 diagnostic，不进模型输入。
- 是否需要同步增加 `num_pred>1` 的快速 smoke 来削弱 last-beam persistence？已将 sensor-assisted quick/smoke 配置统一为 `seq_len=5`、`num_pred=3`，后续重新运行矩阵时需单独记录为新历史窗口/horizon 结果。
