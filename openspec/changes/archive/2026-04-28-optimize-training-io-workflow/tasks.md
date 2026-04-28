## 1. 模态选择与 Dataset 懒加载

- [x] 1.1 在 `src/kd_sensing/engine/builders.py` 中新增启用模态推导 helper，覆盖 image、radar、gps、lidar 和 fusion 默认/显式 `modalities`。
- [x] 1.2 在 builder 中校验 fusion teacher/student 模态一致性，并将 `enabled_modalities` 传入 dataset 配置。
- [x] 1.3 扩展 `src/kd_sensing/data/samples.py`，让 `create_samples` 可按启用模态校验/解析列，未启用模态缺列时不失败。
- [x] 1.4 更新 `src/kd_sensing/data/datasets/scenario9.py`，使 `__getitem__` 只调用启用模态的加载函数，并只返回启用模态字段。
- [x] 1.5 修复 `target_beam` 构造，确保 `num_pred=1` 时单样本维度为 `[1]`、batch 维度为 `[B, 1]`。
- [x] 1.6 确认 `src/kd_sensing/engine/batch.py` 的单模态和 fusion 输入准备只依赖启用模态字段，并为缺失必需字段输出清晰错误。

## 2. DataLoader 与运行目录

- [x] 2.1 新增共享 DataLoader 参数构造 helper，支持 `pin_memory`、`persistent_workers`、`prefetch_factor`、`drop_last`，并在 `num_workers=0` 时过滤 worker-only 参数。
- [x] 2.2 更新 `build_dataloaders` 和 `evaluate`，让训练与评估复用同一 DataLoader 参数解析逻辑。
- [x] 2.3 更新 `src/kd_sensing/config/defaults.py` 和相关 YAML 默认值，提供并行训练更稳的 DataLoader 配置字段。
- [x] 2.4 更新 `create_run_dir`，让固定 `output.run_name` 默认创建唯一目录，并支持显式 overwrite 与 resume 复用目录。
- [x] 2.5 更新评估输出目录策略，避免默认覆盖 `outputs/evaluation/test_report.json`。
- [x] 2.6 在训练日志、评估报告或最终配置中记录实际 train/test CSV 路径和 split 样本数。

## 3. LiDAR Cache 与 Split 配置

- [x] 3.1 更新 LiDAR-only 和包含 LiDAR 的 fusion 配置，提供稳定 `lidar_cache_dir`、`lidar_use_cache` 和可选 `lidar_write_cache` 默认值。
- [x] 3.2 为 LiDAR cache 增加参数隔离策略，确保 BEV size、ROI、FoV、ground filter 或背景过滤参数变化时不会误用旧 cache。
- [x] 3.3 确认 LiDAR cache 读取/写入只在取样阶段发生，dataset 初始化不遍历 cache 目录或全量读取 `.npy`。
- [x] 3.4 统一所有默认单模态和 fusion 配置，使其引用同一 train/test CSV。
- [x] 3.5 移除历史 split 特殊口径，并确保输出记录统一 split 路径和样本数。

## 4. 测试与验证

- [x] 4.1 新增单元测试，使用 monkeypatch 验证 GPS-only、LiDAR-only、radar-only、image-only 和 selected fusion 不调用未启用模态加载函数。
- [x] 4.2 新增 `create_samples` 测试，覆盖未启用模态缺列、启用 GPS/LiDAR 缺列时报清晰错误。
- [x] 4.3 新增 `num_pred=1` 标签维度测试，覆盖 dataset 样本、DataLoader batch 和 `prepare_labels`。
- [x] 4.4 新增 DataLoader 参数测试，覆盖 `num_workers=0` 和多 worker 参数透传。
- [x] 4.5 新增输出目录测试，覆盖固定 `run_name` 不覆盖、resume 复用、评估多次运行不覆盖。
- [x] 4.6 新增 LiDAR cache 测试，覆盖 cache hit、cache miss 写入、参数隔离和初始化不全量读取。
- [x] 4.7 运行 `conda run -n kd_mm_beam pytest` 验证全部测试通过。
- [x] 4.8 运行 `conda run -n kd_mm_beam pytest tests/test_gps_modality.py tests/test_lidar_modality.py tests/test_student_configs.py` 验证既有关键测试通过。
- [x] 4.9 使用 `conda run -n kd_mm_beam python -m kd_sensing.cli.train --config <smoke-config> --override training.epochs=1 --override data.dataset.portion=0.01` 或项目等价 smoke 命令验证短训练路径。
