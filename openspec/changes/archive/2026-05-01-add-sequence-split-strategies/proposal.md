## Why

Scene 32 的原始帧数不少，但只有很少的 `seq_index` 轨迹段；当前预处理按 `seq_index` 原始顺序做 80/20 切分，导致验证集只覆盖末尾少数轨迹段，image/radar/LiDAR 这类强场景相关模态容易出现训练精度上升、验证精度下降。

需要让序列 CSV 生成过程支持可复现、可配置且更有代表性的 seq-level split，避免 Scene 32 默认实验被窄验证域主导，同时保持不同模态仍使用同一组 train/test 窗口。

## What Changes

- **BREAKING**: `sequence_csv` 预处理不再使用旧的按 `seq_index` 原始顺序 80/20 切分协议；所有默认统一 split 需要用新的协议重新生成。
- 将 `sequence_csv` 预处理改为单一的确定性标签分布感知 seq-level split 协议，使用 `split_seed` 处理并列选择和可复现实验。
- 对 Scene 32 和 Scene 9 都使用同一套新 split 协议，确保 train/test 都覆盖更完整的轨迹与 beam label 分布，而不是只把最后 20% `seq_index` 留作 test。
- 保持滑动窗口生成语义不变：窗口仍只在单个 `seq_index` 内生成，不跨轨迹拼接历史输入或未来标签。
- 保持跨模态比较语义不变：image、radar、GPS、LiDAR、mmWave 和 fusion 继续引用同一组生成后的 train/test CSV。
- 在预处理输出、训练 final config 或运行 metadata 中记录 split protocol、seed、train/test `seq_index` 列表和标签分布摘要，方便判断实验是否可比较。
- 不改变模型结构、loss、训练循环或评估指标计算。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `modality-aware-data-loading`: 序列 CSV 生成流程需要支持可配置的 seq-level split 策略，并记录 split 元数据。
- `experiment-workflow`: 统一 split 配置和运行输出需要记录 split 策略信息，保证不同模态实验可复现、可横向比较。

## Impact

- 受影响代码：`src/kd_sensing/preprocessing/sequences.py`、预处理 CLI 配置、相关测试和文档。
- 受影响配置：`configs/preprocess/sequences_ra*.yaml`，尤其是 `configs/preprocess/sequences_ra_gps_lidar.yaml`；旧的隐式顺序切分配置语义将被移除。
- 受影响产物：需要重新生成 `train_seqs_*.csv`、`test_seqs_*.csv` 以及 split metadata sidecar。
- 不引入新的运行时依赖；训练入口和 dataset 读取接口保持兼容。
