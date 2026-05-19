## 1. 模型与输出契约

- [x] 1.1 在 `src/kd_sensing/models/modular.py` 中实现并注册 `snapshot_frame` representation core，支持单模态 `[B, 1, D]` 与多模态 `[B, K, 1, D]` 输入，输出 `[B, 1, D_out]`。
- [x] 1.2 为 `snapshot_frame` 增加时间维校验，收到 `T != 1` 时抛出包含 `seq_len=1`、`num_pred=1` 提示的清晰错误。
- [x] 1.3 确保 `snapshot_frame` 不创建 GRU/RNN/LSTM/TCN 或跨时间 attention；多模态分支只在当前帧模态 token 间融合。
- [x] 1.4 验证 `snapshot_frame` 与现有 beam head、`TemporalAuxiliaryHeads`、`adapt_model_output()` 和 `select_prediction_slots()` 兼容。

## 2. Snapshot 数据预处理与统计 artifact

- [x] 2.1 新增 `configs/preprocess/sequences_snapshot_next_frame.yaml`，默认 Scenario 31、`in_len: 1`、`out_len: 1`、`training_set_pct: 0.8`，并启用 GPS、LiDAR、mmWave 列输出。
- [x] 2.2 扩展序列预处理，写出 `train_seqs_SNAPSHOT_NEXT_FRAME.csv` 和 `val_seqs_SNAPSHOT_NEXT_FRAME.csv`，按完整 `seq_index` 做 80/20 balanced split。
- [x] 2.3 扩展 split metadata，记录 `split_protocol: snapshot_next_frame_balanced_seq`、`in_len: 1`、`out_len: 1`、train/validation `seq_index`、窗口数、label 分布、输出 CSV 路径和 split seed。
- [x] 2.4 支持 snapshot position 目标列：启用 `include_position_targets: true` 时写出 `future_gps1` 和 `future_bs_gps1`。
- [x] 2.5 确保 mmWave scaler、GPS scaler、LiDAR streaming stats、occlusion threshold 和 position target scaler 都只基于 snapshot train split 拟合，并在 artifact metadata 中记录 snapshot split fingerprint。
- [x] 2.6 允许帧级 LiDAR BEV cache 复用，但禁止 split-dependent normalizer/stat artifact 复用历史窗口 run 的 fingerprint。

## 3. Snapshot 配置生成

- [x] 3.1 扩展配置解析，支持 `configs/<modality>/snapshot_next_frame_no_kd.yaml` 单模态入口，覆盖 `seq_len=1`、`num_pred=1`、`model.seq_length_* = 1`、`model.num_pred=1`。
- [x] 3.2 扩展 fusion virtual canonical 解析，支持 `configs/fusion/<slug>_snapshot_next_frame_no_kd.yaml`，并复用现有 canonical slug 校验和固定模态顺序。
- [x] 3.3 增加 `configs/fusion/all_modalities_snapshot_next_frame_no_kd.yaml` 五模态兼容别名或等价 virtual alias。
- [x] 3.4 生成的 snapshot 配置必须默认指向 `train_seqs_SNAPSHOT_NEXT_FRAME.csv` 和 `val_seqs_SNAPSHOT_NEXT_FRAME.csv`，找不到时提示先运行 snapshot 预处理。
- [x] 3.5 生成的 snapshot 配置必须设置 `distillation.type: no_kd`、`distillation.teacher_model_name: null`、无时序 `model.student`，并设置清晰的 `experiment.name` 与 `output.run_name`。
- [x] 3.6 在配置校验或加载后补全中记录 `experiment.variant: snapshot_next_frame` 或等价 metadata，并防止 snapshot 配置被命令行覆盖成 `seq_len>1` 后仍伪装为 snapshot。

## 4. Workflow Metadata 与文档

- [x] 4.1 扩展运行 metadata/final config 写入，记录 `variant`、`uses_history_window`、`uses_temporal_core`、`seq_len`、`num_pred`、enabled modalities、objective、scene、train/validation CSV、样本数和 split metadata 路径。
- [x] 4.2 更新结果汇总或模态失衡分析工具，使其输出包含 `variant`、`seq_len`、`num_pred`、`uses_temporal_core` 和 `split_protocol`，避免 snapshot 与历史窗口结果被静默合并。
- [x] 4.3 更新 README 或实验文档，加入 snapshot 预处理命令、单模态和五模态 snapshot baseline 命令，并说明 snapshot 使用 80/20 train/validation split。
- [x] 4.4 文档中明确 mmWave snapshot 输入语义为“当前帧 mmWave power 预测下一帧”，避免与完全无 radio context 的 ablation 混淆。

## 5. 测试与验证

- [x] 5.1 添加模型单元测试：单模态 snapshot forward shape、五模态 snapshot forward shape、`T>1` 报错、模型模块树不含 GRU/RNN/LSTM。
- [x] 5.2 添加辅助 head 单元测试：snapshot occlusion 输出 `[B, 1]`，position 输出 `[B, 1, 2]`，并能被 objective-aware loss/metric 路径消费。
- [x] 5.3 添加预处理测试：snapshot 配置写出 train/validation CSV、80/20 sequence split metadata、`in_len=1/out_len=1` 和必要模态列。
- [x] 5.4 添加配置加载测试：五个 `configs/<modality>/snapshot_next_frame_no_kd.yaml` 和至少 `configs/fusion/image_radar_gps_lidar_mmwave_snapshot_next_frame_no_kd.yaml` 能生成有效 final config。
- [x] 5.5 添加标签对齐测试：snapshot 配置下 labels 来自第一个 future target，不包含当前帧 `input_beam`。
- [x] 5.6 添加 artifact 隔离测试：snapshot mmWave/LiDAR/target stats artifact fingerprint 与历史窗口 split 不匹配时不得复用。
- [x] 5.7 使用 `conda run -n kd_mm_beam pytest tests/test_snapshot_next_frame_baselines.py tests/test_preprocessing_formats.py tests/test_student_configs.py tests/test_training_io_workflow.py` 运行相关测试。
- [x] 5.8 使用 `conda run -n kd_mm_beam python scripts/preprocess.py --config configs/preprocess/sequences_snapshot_next_frame.yaml` 生成 snapshot train/validation CSV。
- [x] 5.9 使用 `conda run -n kd_mm_beam python scripts/train.py --config configs/gps/snapshot_next_frame_no_kd.yaml training.epochs=1 data.dataset.portion=0.02` 执行单模态 smoke test。
- [x] 5.10 使用 `conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/all_modalities_snapshot_next_frame_no_kd.yaml training.epochs=1 data.dataset.portion=0.02` 执行五模态 fusion smoke test。
