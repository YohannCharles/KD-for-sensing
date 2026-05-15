## 1. Profile 与诊断增强

- [x] 1.1 扩展 `scripts/profile_training_io.py`，输出 train/test split 的 batch size、num_workers、persistent_workers、prefetch_factor、pin_memory、progress 状态和 cache policy。
- [x] 1.2 为 profile 增加模态级 `__getitem__` 计时，分别汇总 image、radar、gps、lidar、mmwave 和 auxiliary targets 的均值、P50、P95。
- [x] 1.3 增加 DataLoader wait 与 GPU step 对比字段，明确输出 wait/forward/backward/transfer 的比例和 P95 尖峰。
- [x] 1.4 使用 `conda run -n kd_mm_beam python scripts/profile_training_io.py --config configs/fusion/image_radar_gps_lidar_mmwave_beam_no_kd.yaml --samples 64 --output outputs/profile/five_modal_gpu_utilization.json` 验证 profile 可运行。

## 2. DataLoader worker 策略

- [x] 2.1 扩展 DataLoader 配置解析，支持 train/test split 专用 `num_workers`、`persistent_workers` 和 `prefetch_factor`，并保持旧 `data.dataloader.num_workers` 兼容。
- [x] 2.2 调整训练流程或 DataLoader factory，避免 test DataLoader worker 在训练阶段无条件长期驻留；验证阶段仍能正常创建和复用 test loader。
- [x] 2.3 增加单元测试覆盖旧配置兼容、test split 独立 worker 参数和 persistent worker 行为。

## 3. 并行训练推荐与日志降噪

- [x] 3.1 新增并行训练推荐 helper 或脚本，根据 parallel runs、CPU 数、启用模态和 cache 状态输出推荐覆盖参数。
- [x] 3.2 推荐器在四个五模态后台任务场景下输出 `output.progress.enabled=false`、合理 worker/prefetch、cache policy 和可选 AMP 建议。
- [x] 3.3 调整训练 progress 配置，确保 `output.progress.enabled=false` 时不写 batch 级 tqdm，但 epoch metrics、TensorBoard、checkpoint 和 `training_outputs.npz` 正常。
- [x] 3.4 增加测试覆盖后台低噪声 progress 不影响训练 artifacts。

## 4. Cache 复用与文档

- [x] 4.1 增加 LiDAR BEV cache 覆盖率检查或预热说明，避免四任务并行时重复 cache write。
- [x] 4.2 记录并复用 LiDAR normalizer、GPS scaler、mmWave scaler 和 occlusion stats 的推荐流程，说明何时可使用 `read_only`。
- [x] 4.3 更新 README 或 `docs/training_throughput.md`，写明四任务并行推荐命令和 GPU 低利用率排查顺序。

## 5. 验证

- [x] 5.1 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_lidar_modality.py tests/test_student_configs.py`。
- [x] 5.2 运行 `openspec status --change improve-training-gpu-utilization`，确认 artifacts apply-ready。
- [x] 5.3 在一个五模态配置上对比优化前后 profile 输出，确认 DataLoader wait 和 GPU step 分解可解释 GPU 利用率。
