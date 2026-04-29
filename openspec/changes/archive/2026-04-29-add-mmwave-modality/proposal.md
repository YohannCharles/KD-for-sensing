## Why

当前项目已经支持 image、radar、GPS、LiDAR 及其可选融合，但还不能把 DeepSense Scenario 9 中的 60GHz mmWave beam-training receive-power vector 作为独立 sensing modality 使用。新增 mmWave 模态可以直接利用 `unit1_pwr_60ghz` / `unit1/mmWave_data/mmWave_power_*.txt` 中的 64 维接收功率向量，补齐低成本向量模态的单模态和多模态波束预测实验路径。

## What Changes

- 新增 mmWave 数据预处理能力：从序列 CSV 生成 `mmwave1..mmwaveN` 历史路径列，读取 64 维 power vector，做 finite/NaN 清洗、dB 压缩和训练集 z-score 归一化。
- 扩展 `SequenceSamples`、`Scenario9Dataset` 和模态推导逻辑，使启用 mmWave 时样本返回 `mmwave` 字段，形状为 `[seq_len, 64]`，并且不影响现有 `beam1..future_beamN` 标签路径。
- 新增 mmWave 模型族：`MmWaveFeatureExtractor`、`MmWaveModalityNet` 和 `MmWaveStudentModalityNet`，使用向量序列 MLP/GRU 风格建模，并保持 `(pred, features, output_features)` 输出契约。
- 扩展训练、验证、评估和 batch 准备流程，支持 `experiment.task: mmwave`，并支持 mmWave 在 fusion `modalities` 中被选择。
- 新增 mmWave-only teacher no-KD、student no-KD、logits KD、RKD 配置，并扩展 canonical fusion 配置矩阵覆盖包含 `mmwave` 的组合。
- 新增测试覆盖 mmWave power vector 读取、归一化、dataset 字段、模型 forward、batch 准备、配置构建、KD 兼容和 fusion 模态校验。

## Capabilities

### New Capabilities
- `mmwave-preprocessing`: 定义 mmWave 序列列生成、64 维 receive-power vector 读取、dB 压缩、训练集归一化、scaler 工件和 dataset batch 字段契约。
- `mmwave-modality-model`: 定义 mmWave-only teacher/student 模型、输入输出契约、batch 预测窗口补齐、KD 兼容性和默认实验配置。

### Modified Capabilities
- `modality-aware-data-loading`: 将 `mmwave` 纳入启用模态推导、按模态懒加载和未启用模态不读取的契约。
- `configurable-multimodal-fusion`: 将 `mmwave` 纳入 fusion 可选模态，要求 teacher/student fusion 分支、输入校验和 canonical 配置矩阵支持 mmWave。
- `experiment-workflow`: 将 `mmwave` 纳入配置驱动训练、评估、预处理、默认配置和统一 split 检查。
- `component-registry`: 要求 mmWave 模型和预处理组件可通过现有注册机制发现与构建。
- `experiment-artifact-registry`: 要求 checkpoint metadata 能记录并复用 mmWave scaler 工件，避免评估阶段重新 fit 或扫描训练 split。

## Impact

- 影响代码：`src/kd_sensing/preprocessing/sequences.py`、`src/kd_sensing/data/`、`src/kd_sensing/engine/batch.py`、`src/kd_sensing/engine/builders.py`、`src/kd_sensing/models/`、`src/kd_sensing/models/fusion/`、`src/kd_sensing/config/defaults.py`、`configs/`、`tests/`。
- 数据接口：序列 CSV 需要可选 `mmwave1..mmwaveN` 列；启用 mmWave 时 dataset 从这些路径加载 64 维 receive-power feature，同时继续使用 beam 路径生成历史和未来标签。
- 运行接口：新增 `experiment.task: mmwave` 和 fusion `modalities: [..., "mmwave"]`；旧 image/radar/GPS/LiDAR/fusion 配置必须保持兼容。
- 依赖影响：优先使用 NumPy/PyTorch 实现读取、dB 压缩和 z-score 归一化，不引入新的外部依赖。
