## 1. LiDAR Normalization 配置一致性

- [x] 1.1 将 `MODALITY_SPECS["lidar"].dataset_field_defaults` 的默认 normalization 改为 raw BEV 语义，避免默认注入 `lidar_normalization.enabled: true`。
- [x] 1.2 在配置解析或 dataset 参数准备路径增加 LiDAR normalization canonicalization，使 `lidar_normalize` 与 `lidar_normalization.enabled` 在最终配置中保持一致。
- [x] 1.3 在 `DeepSense6GDataset._resolve_lidar_normalization()` 增加冲突检查，直接构造 dataset 时遇到 false/true 冲突必须抛出清晰错误。
- [x] 1.4 修复仓库内 LiDAR 和含 LiDAR fusion YAML，将默认 raw profile 设为 disabled，并只在显式 stats profile 中启用 streaming stats。
- [x] 1.5 确认 `final_config.yaml`、LiDAR preprocessing metadata 和 normalization artifacts 对 raw/stats profile 的记录一致。

## 2. Teacher Checkpoint 与 Metrics 选择

- [x] 2.1 调整训练指标导出，使 teacher metrics 记录 `selection_metric`、`selection_mode`、selected epoch 和 checkpoint 路径，并让 `best_epoch` 指向默认 teacher checkpoint。
- [x] 2.2 调整 `archive_best_checkpoint()` 调用或 sidecar metadata，使 objective checkpoint 与 Top-1 checkpoint 的来源可区分。
- [x] 2.3 修改 teacher registry checkpoint 查找逻辑：优先使用 metrics/metadata 中声明的 checkpoint，其次使用 `best.pth`，仅在显式 Top-1 selection 时使用 `best_top1.pth`。
- [x] 2.4 对 LiDAR teacher 缺少 objective checkpoint 且未显式选择 Top-1 的情况抛出可诊断错误。

## 3. LiDAR Quality 诊断

- [x] 3.1 扩展 LiDAR 读取或 batch 构造路径，使 quality accumulator 能同时接收 raw BEV 和模型实际输入张量。
- [x] 3.2 更新 `LidarQualityAccumulator` 或新增包装结构，输出 `raw` 与 `model_input` 两组非空率、通道均值、标准差和零值比例。
- [x] 3.3 更新训练、评估和 teacher registry 扩展 metadata，保留 raw 稀疏度并继续记录 normalization 后的输入统计。
- [x] 3.4 增加异常幅值或 raw 极端稀疏度的 degradation reason，并在报告中包含 ROI、FoV、normalization 和 cache 参数。

## 4. 测试与验证

- [x] 4.1 添加配置解析单元测试，覆盖 raw 默认、显式 stats、false/true 冲突拒绝和最终配置同步。
- [x] 4.2 添加 teacher registry 单元测试，覆盖 LiDAR 同时存在 `best.pth`/`best_top1.pth` 时默认选择 `best.pth`，以及显式 Top-1 时选择 `best_top1.pth`。
- [x] 4.3 添加 LiDAR quality 单元测试，验证 z-score 后 `model_input.zero_ratio` 不会覆盖 `raw.zero_ratio`。
- [x] 4.4 使用 `conda run -n kd_mm_beam pytest <相关测试路径>` 运行新增和受影响测试。
- [x] 4.5 使用 `conda run -n kd_mm_beam <训练或配置解析命令>` 对一个短 epoch LiDAR 配置做 smoke test，确认 `final_config.yaml` 中 normalization profile 与实际 dataset 行为一致。
