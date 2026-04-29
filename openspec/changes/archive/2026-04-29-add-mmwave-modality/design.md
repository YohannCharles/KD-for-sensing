## Context

项目目前通过注册表、YAML 配置和统一训练入口支持 image、radar、GPS、LiDAR 及可选 fusion。`Scenario9Dataset` 根据 `experiment.task` 和 fusion `modalities` 推导启用模态，只读取启用模态的字段；GPS 使用 `[T, 3]` 向量序列和训练集 scaler，LiDAR 使用 `[T, C, H, W]` BEV 与可选 normalizer。

mmWave 方案建议使用 DeepSense Scenario 9 中 `unit1_pwr_60ghz` / `unit1/mmWave_data/mmWave_power_*.txt` 的 64 维 receive-power vector。当前项目已经把同一类 power txt 用作 `beam1..future_beamN` 标签来源，并通过 `argmax` 得到 beam label。新增 mmWave 模态时需要把输入特征路径与标签路径分开表达：`mmwave1..mmwaveN` 表示历史输入，`beam1..future_beamN` 仍表示历史和未来标签，避免以后替换标签来源或做消融时语义混淆。

## Goals / Non-Goals

**Goals:**
- 支持序列生成流程输出 `mmwave1..mmwaveN` 历史路径列，默认来源为 `unit1_pwr_60ghz`。
- 支持读取 64 维 mmWave receive-power vector，进行无效值清洗、dB 压缩和训练集 z-score 归一化。
- 支持 `Scenario9Dataset` 在启用 mmWave 时返回 `[seq_len, 64]` 的 `mmwave` float32 张量。
- 新增 `MmWaveFeatureExtractor`、`MmWaveModalityNet` 和 `MmWaveStudentModalityNet`，注册名为 `mmwave_feature_extractor`、`mmwave_teacher`、`mmwave_student`。
- 支持 `experiment.task: mmwave` 的训练、验证、评估、no-KD、logits KD 和 RKD。
- 支持 fusion `modalities` 包含 `mmwave`，并扩展包含 mmWave 的 canonical fusion 配置。
- 保存并复用训练集 fit 后的 mmWave scaler，评估 registry checkpoint 时不扫描训练 split。

**Non-Goals:**
- 不把 mmWave 当作 radar RA/DA 图像处理，不新增 FFT、range-angle、Doppler 或 CNN 默认路径。
- 不把 `beam1..future_beamN` 标签列直接复用为公开的 mmWave 输入字段；第一版显式生成 `mmwave1..mmwaveN`。
- 不对每个样本做 softmax、min-max 或只保留相对排序作为默认输入，因为这会丢失绝对功率、SNR 和 path-loss 信息。
- 不支持未来 mmWave power vector 作为输入；预测窗口内只对未来时隙补零，避免泄漏未来观测。
- 不引入新的外部依赖。

## Decisions

1. **mmWave 输入使用独立 `mmwave1..mmwaveN` 列**

   序列生成在 `include_mmwave: true` 时从 `unit1_pwr_60ghz` 复制历史路径到 `mmwave1..mmwaveN`。标签列继续使用 `beam1..future_beamN`，即使第一版两者来自同一原始列。这样 dataset 可以清楚地区分“输入 power vector”和“beam label 文件”，也避免启用 mmWave 时把标签读取逻辑绑定到输入模态。

2. **特征构造采用 dB 压缩 + 训练集 z-score**

   reader 将 txt 解析为 64 维 float32，非法维度、缺失文件或无法解析时抛出带路径的错误；NaN/inf 会在 log 前清洗，非正功率会以小 epsilon 裁剪，随后计算 `10 * log10(power)`。归一化只在训练 split 上 fit `MmWaveStandardScaler`，并把同一 scaler 传给 test/eval split。相比按样本 softmax 或 min-max，dB+z-score 保留绝对接收功率差异，同时让 MLP/GRU 训练更稳定。

3. **模型沿用 GPS 风格的向量序列网络**

   mmWave 输入是 `[B, T, 64]` 向量序列，不需要 CNN。teacher 使用 `Linear(64 -> hidden) + LayerNorm + GELU + Dropout + Linear(hidden -> feature_size)`、LayerNorm、GRU、时序 attention 或 MHA 增强、classifier；student 使用更窄 hidden、单层 GRU 和小型 classifier。两者返回 `(pred, features, output_features)`，便于 no-KD、logits KD、RKD 与现有训练循环复用。

4. **batch 准备按历史输入补零预测窗口**

   `prepare_mmwave_inputs` 从 batch 读取 `mmwave`，截取最近 `seq_length` 个历史时隙，并追加 `num_pred - 1` 个全零未来占位时隙，输出 `[B, seq_length + num_pred - 1, 64]`。这与 GPS、radar、LiDAR 的预测窗口契约一致，损失和指标继续对最后 `num_pred + 1` 个输出时隙与标签对齐。

5. **fusion 扩展为第五个可选模态**

   `VALID_FUSION_MODALITIES`、数据构建和模型 forward 校验都增加 `mmwave`。fusion teacher 使用 `MmWaveFeatureExtractor` 作为分支；fusion student 使用轻量投影层输出固定维度 embedding，再与其它启用模态拼接。未启用 mmWave 时 dataset 不要求 `mmwave1..N` 列，batch 准备和模型 forward 也不要求 `mmwave` 字段。

6. **归一化工件进入现有 artifact registry metadata**

   `save_normalization_artifacts` 保存 `mmwave_scaler.npz`，checkpoint sidecar 记录该路径。评估入口加载 registry checkpoint 时通过 `load_normalization_artifacts` 恢复 scaler，并传入测试 dataset；缺少 scaler 且测试配置启用 mmWave normalize 时抛出清晰错误，而不是重新 fit 或扫描训练 CSV。

## Risks / Trade-offs

- **mmWave 与 beam 标签来自同一原始列，语义容易混淆** -> 通过独立 `mmwave1..N` 输入列和 `beam/future_beam` 标签列隔离语义，并在错误信息中提示需要重新生成带 mmWave 列的序列 CSV。
- **训练集 scaler 需要读取训练 split 的 mmWave 文件** -> 单帧仅 64 维，内存和 I/O 成本远低于 LiDAR；只在启用 mmWave normalize 时执行，并保存工件供评估复用。
- **power txt 可能存在非正值或异常值** -> reader 在 dB 前做有限值清洗和 epsilon 裁剪，同时保留严格的 64 维长度校验，避免静默吞掉坏数据。
- **canonical fusion 组合数量增加** -> 使用脚本或模板化生成配置，固定模态顺序为 `image`、`radar`、`gps`、`lidar`、`mmwave`，减少手工维护错误。
- **旧 CSV 不包含 `mmwave1..N`** -> 只有启用 mmWave 时才校验该列；旧 image/radar/GPS/LiDAR 配置继续运行。启用 mmWave 的用户需要重新运行序列预处理。

## Migration Plan

1. 更新序列生成、samples、transforms、dataset、builder 和 artifact registry，先跑通 mmWave-only dataset 与 scaler。
2. 新增 mmWave 模型和 batch/forward 路径，补齐 teacher/student forward、参数校验和 KD smoke 测试。
3. 扩展 fusion 模型和配置矩阵，确保旧默认 image+radar fusion 入口行为不变。
4. 新增 `configs/mmwave/*.yaml`、更新含 mmWave 的 preprocess/fusion 配置，并补充 README 或扩展文档说明。
5. 使用 synthetic/fixture power txt 跑单元测试，再使用 `conda run -n kd_mm_beam pytest` 验证完整测试套件。

## Open Questions

- 本地 `dataset/scenario9/scenario9_RA.csv` 是否已稳定包含 `unit1_pwr_60ghz`，还是部分数据包使用其它列名；实现时可提供 `mmwave_column` 与 fallback 配置。
- mmWave scaler 默认是否总是启用；第一版建议默认启用，因为 64 维 dB 特征量纲差异会影响 MLP/GRU 收敛。
- 是否需要给 mmWave 增加离线 `.npy` 特征缓存；第一版不需要，除非真实训练 I/O 显示 power txt 读取成为瓶颈。
