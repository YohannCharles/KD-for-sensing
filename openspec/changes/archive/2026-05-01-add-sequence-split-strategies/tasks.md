## 1. 预处理 split 协议

- [x] 1.1 重构 `src/kd_sensing/preprocessing/sequences.py`，把窗口生成和 train/test seq 选择拆成可测试 helper。
- [x] 1.2 实现单一 `balanced_seq` 选择器，按完整 `seq_index` 分配 train/test，并使用 `split_seed` 处理并列候选顺序。
- [x] 1.3 支持 `training_set_pct`、`min_test_sequences` 和 `test_sequence_count` 或等价数量控制，并对冲突配置给出清晰错误。
- [x] 1.4 移除旧的隐式 `all_seq_idx[:80%]` 顺序切分逻辑，默认预处理配置不再暴露旧兼容路径。
- [x] 1.5 为每次序列预处理写出 split metadata sidecar，包含 `split_protocol: balanced_seq`、seed、CSV 路径、train/test seq 列表、窗口数和 label 分布摘要。

## 2. 配置与运行 metadata

- [x] 2.1 更新 `configs/preprocess/sequences_ra*.yaml`，使用新的 `balanced_seq` 协议字段并设置适合 Scene 32 的最少 test seq 约束。
- [x] 2.2 增加 split metadata 发现/读取 helper，让训练和评估 metadata 能记录 sidecar 路径或核心字段。
- [x] 2.3 更新 `dataset_run_metadata()`、训练 final config、`train_log.json` 和评估报告，使其记录 split protocol、seed、seq 数量和样本数。
- [x] 2.4 当默认统一 split CSV 缺少 `balanced_seq` metadata 时，训练或评估入口必须给出清晰错误或显式警告。

## 3. 数据再生成与文档

- [x] 3.1 使用 `conda run -n kd_mm_beam python scripts/preprocess.py --config configs/preprocess/sequences_ra_gps_lidar.yaml` 重新生成 Scene 32 统一 split，并检查 train/test seq 列表和 label 分布。
- [x] 3.2 使用 `conda run -n kd_mm_beam python scripts/preprocess.py --config configs/preprocess/sequences_ra_gps_lidar.yaml data.dataset.scene=9` 或等价配置重新生成 Scene 9 统一 split。
- [x] 3.3 更新 README 预处理说明，明确旧顺序 split 与新 `balanced_seq` split 是不同实验协议，结果不可直接混比。
- [x] 3.4 记录推荐的 Scene 32 验证方案：先重跑图中 7 种 image/radar/LiDAR 组合，再观察 train/test label 分布和验证曲线。

## 4. 测试与验证

- [x] 4.1 添加单元测试，验证 `balanced_seq` 不把同一 `seq_index` 分到 train 和 test 两侧。
- [x] 4.2 添加单元测试，验证相同 `split_seed` 可复现，不同 seed 可改变并列候选选择。
- [x] 4.3 添加单元测试，验证小 seq 数场景满足 `min_test_sequences`，且冲突配置会报错。
- [x] 4.4 添加单元测试，验证 split metadata 行数、seq 列表和 label 分布摘要与输出 CSV 一致。
- [x] 4.5 使用 `conda run -n kd_mm_beam pytest tests/test_preprocessing_formats.py tests/test_training_io_workflow.py` 验证预处理与运行 metadata 行为。
- [x] 4.6 运行 `openspec validate add-sequence-split-strategies --strict`，确认提案、设计、spec 和任务合法。
