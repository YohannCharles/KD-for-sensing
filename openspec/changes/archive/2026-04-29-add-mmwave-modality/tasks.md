## 1. 数据与预处理

- [x] 1.1 更新 `src/kd_sensing/preprocessing/sequences.py`，新增 `include_mmwave`、`mmwave_column` 和 fallback 配置，并在序列 CSV 中输出 `mmwave1..mmwaveN`
- [x] 1.2 更新 `configs/preprocess/sequences_ra_gps_lidar.yaml` 或新增等价预处理配置，使用户可生成带 mmWave 输入列的统一 train/test sequence CSV
- [x] 1.3 更新 `src/kd_sensing/data/samples.py`，让 `SequenceSamples` 保存可选 `mmwave_paths`，并在启用 mmWave 时校验 `mmwave1..N` 列
- [x] 1.4 在 `src/kd_sensing/data/transforms.py` 实现 `read_mmwave_power_vector`、dB 压缩特征构造和 `load_mmwave_feature_sequence`
- [x] 1.5 在 `src/kd_sensing/data/transforms.py` 实现 `MmWaveStandardScaler`，支持 train fit、transform、save/load，并覆盖 64 维 mean/scale 校验

## 2. Dataset、Builder 与归一化工件

- [x] 2.1 更新 `src/kd_sensing/data/datasets/scenario9.py`，将 `mmwave` 加入合法模态、dataset 参数、列校验、样本读取和 `[seq_len, 64]` float32 返回字段
- [x] 2.2 更新 `Scenario9Dataset` 的 mmWave scaler 准备逻辑，确保只在 train split fit，test/eval split 必须复用传入 scaler 或工件
- [x] 2.3 更新 `src/kd_sensing/engine/builders.py`，将 `mmwave` 加入启用模态推导、`use_mmwave` 冲突校验、train/test dataset scaler 传递和 run metadata
- [x] 2.4 更新 `save_normalization_artifacts` 与 `load_normalization_artifacts`，保存和加载 `mmwave_scaler.npz`
- [x] 2.5 更新 `src/kd_sensing/utils/artifact_registry.py`，确保 checkpoint sidecar metadata 记录 mmWave scaler，并能供评估入口复用

## 3. mmWave 模型

- [x] 3.1 新增 `src/kd_sensing/models/mmwave.py`，实现并注册 `MmWaveFeatureExtractor`
- [x] 3.2 实现并注册 `MmWaveModalityNet`，注册名为 `mmwave_teacher`，保持 `(pred, features, output_features)` 输出契约
- [x] 3.3 实现并注册 `MmWaveStudentModalityNet`，注册名为 `mmwave_student`，使用轻量 MLP/GRU 路径
- [x] 3.4 为 mmWave teacher/student 添加 `gru_params` 长度、`gru_input_size == feature_size` 和 `mmwave_input_size == 64` 校验
- [x] 3.5 更新 `src/kd_sensing/models/__init__.py` 和相关包导出，确保 mmWave 模型类可被导入并自动注册

## 4. 输入准备与训练评估流程

- [x] 4.1 在 `src/kd_sensing/engine/batch.py` 新增 `prepare_mmwave_inputs`，按预测窗口补齐 `num_pred - 1` 个未来 zero padding
- [x] 4.2 更新 `forward_model`，支持 `task: mmwave` 和 fusion mmWave 多输入分发
- [x] 4.3 更新 `src/kd_sensing/engine/trainer.py`，支持 mmWave-only teacher/student forward 和 KD forward
- [x] 4.4 更新 `src/kd_sensing/engine/validator.py`，支持 mmWave-only 验证和包含 mmWave 的 fusion 验证
- [x] 4.5 更新 `src/kd_sensing/engine/evaluator.py`，确保 mmWave-only 评估能加载 registry scaler 并只准备 mmWave 输入

## 5. Fusion 扩展

- [x] 5.1 更新 `src/kd_sensing/models/fusion/networks.py`，将 `mmwave` 加入 `VALID_FUSION_MODALITIES` 和模态校验
- [x] 5.2 在 `FusionModalityNet` 中新增 mmWave teacher 分支，使用 `MmWaveFeatureExtractor` 并动态更新 fusion projection 维度
- [x] 5.3 在 `StudentModalityNet` 中新增 mmWave student 分支，使用轻量 MLP/projection 生成固定维度 embedding
- [x] 5.4 更新 `prepare_fusion_inputs`，使包含 mmWave 的 fusion 配置只准备启用模态输入，未启用 mmWave 时不要求 `mmwave` 字段
- [x] 5.5 补齐 fusion forward 缺少 mmWave 输入、维度不对齐和非法 `modalities` 的清晰错误信息

## 6. 配置与文档

- [x] 6.1 新增 `configs/mmwave/teacher_no_kd.yaml` 和 `configs/mmwave/no_kd.yaml`，使用 `mmwave_teacher`、`mmwave_input_size: 64`、`mmwave_normalize: true`
- [x] 6.2 新增 `configs/mmwave/student_no_kd.yaml`，使用 `mmwave_student` 并保持默认 `gru_params: [64, 64, 1]`
- [x] 6.3 新增 `configs/mmwave/logits_kd.yaml` 和 `configs/mmwave/rkd.yaml`，默认从 mmWave teacher no-KD 输出或 registry 加载 frozen teacher
- [x] 6.4 扩展 `configs/fusion/` canonical 矩阵，覆盖包含 `mmwave` 的 10 个双模态、10 个三模态、5 个四模态和 1 个五模态 slug，每个 slug 提供 teacher no-KD、student no-KD、logits KD 和 RKD
- [x] 6.5 更新 `src/kd_sensing/config/defaults.py`，补齐 mmWave dataset、model 和归一化默认字段，且不改变旧 image/radar/GPS/LiDAR 默认行为
- [x] 6.6 更新 README 或 `docs/extension_guide.md`，说明 mmWave 输入列生成、模型注册名、scaler 工件和 fusion `modalities` 用法

## 7. 测试与验证

- [x] 7.1 新增 mmWave transform 单元测试，覆盖 64 维读取、NaN/inf/非正值清洗、dB 压缩和非法维度错误
- [x] 7.2 新增 mmWave scaler 测试，断言 test split 复用 train scaler，`mmwave_scaler.npz` 可保存和加载
- [x] 7.3 新增 `Scenario9Dataset` mmWave 测试，覆盖返回 `[seq_len, 64]` 张量、旧 CSV 未启用 mmWave 兼容和缺少列时的错误
- [x] 7.4 新增 mmWave teacher/student 构建、参数校验和 forward contract 测试
- [x] 7.5 新增 mmWave batch、trainer/validator/evaluator smoke 测试，覆盖 `experiment.task: mmwave`
- [x] 7.6 新增 fusion mmWave 测试，覆盖默认值、单模态 mmWave、包含 mmWave 的多模态 forward、非法 modalities 和缺少输入错误
- [x] 7.7 新增配置构建测试，覆盖 `configs/mmwave/*.yaml` 和包含 mmWave 的 canonical fusion 配置
- [x] 7.8 运行 `conda run -n kd_mm_beam pytest tests/test_student_configs.py tests/test_gps_modality.py tests/test_lidar_modality.py tests/test_training_io_workflow.py`
- [x] 7.9 运行 `conda run -n kd_mm_beam pytest` 完整验证测试套件
- [x] 7.10 运行 `openspec status --change add-mmwave-modality` 确认 change artifacts 已达到可实施状态
