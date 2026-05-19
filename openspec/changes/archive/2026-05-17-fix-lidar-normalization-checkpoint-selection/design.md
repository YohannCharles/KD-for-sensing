## Context

LiDAR BEV 目前有两套配置入口：旧布尔字段 `lidar_normalize` 和结构化字段 `lidar_normalization`。多个配置和默认合并路径会生成 `lidar_normalize: false` 但 `lidar_normalization.enabled: true` 的最终配置；dataset 只要看到结构化字段就启用 streaming stats，导致用户读配置时以为是 raw BEV，实际训练输入却是 z-score 后的稀疏 BEV。

训练侧还同时保存 `best.pth` 和 `best_top1.pth`。`best.pth` 跟随 early stopping metric，`best_top1.pth` 跟随验证 Top-1；teacher registry 当前偏向 Top-1 checkpoint 和 Top-1 epoch 指标，容易把多数类水平的 LiDAR checkpoint 注册成 teacher。

## Goals / Non-Goals

**Goals:**

- 让 LiDAR normalization 配置在最终配置、dataset 属性、metadata 和诊断报告中保持一致。
- 让 raw BEV 成为未显式启用 normalization 时的默认行为。
- 保留显式 streaming stats profile，用于后续 `BEV raw` vs `BEV + streaming_stats` 对照实验。
- 让 teacher registry 默认引用与训练选择目标一致的 checkpoint，避免 LiDAR teacher 被多数类 Top-1 checkpoint 污染。
- 让 LiDAR quality summary 同时暴露 raw BEV 稀疏度和模型实际输入统计。

**Non-Goals:**

- 不重写 LiDAR BEV 构造、CNN encoder 或 fusion 架构。
- 不引入新的 LiDAR 点云语义分割、目标检测或背景建模算法。
- 不自动修复历史 run 的 `teacher_registry.json`；历史 registry 需要用修复后的逻辑重建。

## Decisions

1. 以结构化 `lidar_normalization` 作为新配置模型，但禁止与旧布尔字段静默冲突。

   `lidar_normalization` 为 `null` 或缺失时，dataset 使用 `lidar_normalize` 推导 enabled/mode；结构化字段存在时，`enabled` 和旧布尔字段必须一致，或由配置加载阶段同步成一致值。实现时应在 config 解析/合并后的边界做一次 canonicalization，并在 dataset 直接构造时保留冲突检查，防止绕过配置加载的调用再次静默启用 normalization。

   替代方案是继续让结构化字段无条件覆盖旧布尔字段，但这正是本次问题根源，会让 `lidar_normalize: false` 失去诊断意义。

2. 修正默认和显式配置，而不是在训练循环里特殊判断 LiDAR。

   `MODALITY_SPECS["lidar"].dataset_field_defaults` 不应默认注入 `{"enabled": true}`。已有 YAML 如果写了 `lidar_normalize: false`，必须将 `lidar_normalization.enabled` 改为 `false` 或删除结构化块；需要 stats 的配置必须明确设置两处为 true，或只保留 canonical 结构化字段并由 loader 写回一致的 legacy 值。

   替代方案是在训练时根据 run name 推断 raw/stats profile，但这会让同一配置在不同入口下语义不同。

3. teacher registry 使用 objective checkpoint 优先，Top-1 checkpoint 只作为显式 Top-1 registry 的候选。

   训练仍可保存 `best_top1.pth` 作为诊断和兼容产物，但 teacher registry 构建、teacher metrics 选择和 checkpoint metadata 应记录 `selection_metric`、`selection_mode`、`selected_epoch` 和 `checkpoint_path`。未显式要求 Top-1 teacher 时，LiDAR teacher 应优先使用 `best.pth` 或 sidecar 中声明的 objective checkpoint。

   替代方案是继续保留 `best_top1.pth` 优先级并只降低 LiDAR teacher prior，但这会把错误 checkpoint 传播到后续分析，问题更难定位。

4. LiDAR quality 诊断分成 raw 与 model_input 两个视角。

   现有 `LidarQualityAccumulator` 只看模型实际输入；z-score 后原始零值会变成非零，`zero_ratio` 失去稀疏度含义。实现应在 LiDAR loader 或训练/eval batch 生成路径上保留 raw BEV 统计，至少输出 `raw.zero_ratio`、`raw.nonempty_frame_ratio`、`raw.channel_mean/std`，并继续输出 `model_input` 统计用于发现归一化后异常幅值。

   替代方案是只把字段重命名为 `normalized_zero_ratio`，但不能回答 raw BEV 是否几乎为空的问题。

## Risks / Trade-offs

- [Risk] 旧配置因 `lidar_normalize` 与 `lidar_normalization.enabled` 冲突而报错。→ 修复仓库内配置，并在错误信息中说明如何选择 raw 或 streaming stats。
- [Risk] 修复后 LiDAR run 的指标不可与旧 run 直接比较。→ 在 `final_config.yaml`、metrics 和 registry metadata 中记录 normalization profile，并建议重跑 ablation。
- [Risk] registry 优先 objective checkpoint 后，部分已有 Top-1 最优 teacher 的复现实验路径变化。→ 保留显式 Top-1 选择和 `best_top1.pth` 产物，但默认 teacher reliability registry 使用 objective checkpoint。
- [Risk] raw/model_input 双质量统计增加少量开销。→ 只在已有 LiDAR batch 读取路径上累计统计，不新增全量 dataset 扫描。

## Migration Plan

1. 更新默认 modality 配置和仓库内 LiDAR YAML，消除 false/true 冲突。
2. 增加 config canonicalization 和 dataset 冲突检查，确保 direct dataset 构造也不会静默启用 normalization。
3. 调整 trainer 和 teacher registry metadata，使 registry 默认选择 objective checkpoint。
4. 扩展 LiDAR diagnostics 输出 raw 与 model_input 统计。
5. 用 `conda run -n kd_mm_beam pytest ...` 跑覆盖测试，并用一个短 epoch LiDAR 配置确认 `final_config.yaml` 中 normalization profile 与实际 dataset 属性一致。

## Open Questions

无需要阻塞实现的问题。后续是否将 streaming stats 作为单独配置文件命名规范的一部分，可在实现时按现有配置组织方式决定。
