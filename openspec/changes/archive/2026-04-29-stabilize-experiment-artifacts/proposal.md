## Why

当前训练输出目录会因固定 `run_name` 自动追加时间戳，KD 和评估配置仍按固定路径解析 teacher checkpoint，导致已有最佳模型不一定能被复用。与此同时，序列生成、portion 采样、GPS/LiDAR 归一化状态和部分模型默认 GRU 层数规格存在可复现性或文档契约偏差，需要集中修正，避免后续实验横向比较失真。

## What Changes

- 新增统一的最高精度 checkpoint 归档目录，用于保存每个配置当前验证精度最高的 checkpoint，并在 KD/评估加载权重时优先按配置和模态从该目录解析。
- 训练完成或验证集最佳指标刷新时，将当前配置对应的最佳 checkpoint 复制到归档目录，并按 `<slug>_teacher_no_kd_acc_<val_top1>.pth` 这类包含配置 slug、角色/KD 模式和精度的格式命名。
- 持久化 GPS scaler 和 LiDAR normalizer/stats 等训练集归一化工件，评估时优先加载随训练产物保存的归一化状态，而不是重新构建 train dataset 重新 fit。
- 修正 Scenario 9 序列生成窗口边界，包含每个 `seq_index` 的最后一个合法窗口。
- 明确 `portion` 采样语义，避免默认使用 CSV 头部连续样本作为代表性小比例实验。
- 删除未使用的 `gps_smooth_window` 配置、参数传递和文档引用，避免误导用户以为 GPS 平滑参与了特征构造。
- 修正 GPS、RadarStudent 和 LiDAR 模型规格中已过期的默认单模态 GRU 层数要求，使其与当前 README、配置和测试保持一致。

## Capabilities

### New Capabilities
- `experiment-artifact-registry`: 覆盖最佳 checkpoint 归档、优先解析、归一化工件持久化和运行日志记录。

### Modified Capabilities
- `experiment-workflow`: 调整训练/评估权重解析、输出记录和默认参数契约，纳入稳定 artifact 优先级。
- `modality-aware-data-loading`: 修正序列窗口生成边界和小比例 `portion` 采样语义。
- `gps-preprocessing`: 持久化 GPS scaler，并移除未使用的 `gps_smooth_window` 死配置。
- `lidar-preprocessing`: 持久化 LiDAR normalizer/stats，并保证评估复用训练统计量。
- `gps-modality-model`: 将默认 GPS teacher/student/KD 单模态 `gru_params` 更新为 `[64, 64, 1]`。
- `radar-student-model`: 将默认 RadarStudent 单模态 KD `gru_params` 更新为 `[64, 64, 1]`。
- `lidar-modality-model`: 将默认 LiDAR teacher/student/KD 单模态 `gru_params` 更新为 `[64, 64, 1]`。

## Impact

- 影响训练、评估、KD teacher checkpoint 解析和输出目录管理代码，尤其是 `src/kd_sensing/engine/` 下 builder、trainer、evaluator 相关逻辑。
- 影响 Scenario 9 预处理和数据采样逻辑，尤其是 `src/kd_sensing/preprocessing/sequences.py` 与 `src/kd_sensing/data/samples.py`。
- 影响 GPS/LiDAR 预处理、归一化状态保存和加载路径。
- 影响配置文件、README/OpenSpec 规格和相关测试；验证命令必须通过 `conda run -n kd_mm_beam ...` 执行。
