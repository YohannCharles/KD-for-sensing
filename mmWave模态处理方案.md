我按 **mmWave** 理解；你最后一句写了 LiDAR，但仓库里 LiDAR 已经有完整路径了。因为你指定了 `openspec-explore`，我这轮只做探索和方案，不直接改代码。

**结论**
建议新增的 mmWave 模态不要做 radar 那种 FFT/RA/DA 预处理，而是使用 DeepSense 的 `unit1_pwr_60ghz` / `unit1/mmWave_data/mmWave_power_*.txt` 里的 **64 维接收功率向量**：

```text
mmWave txt -> 64-D power vector -> finite/NaN 清洗 -> dB 压缩 -> train-split per-beam z-score -> [T, 64]
```

理由：
- DeepSense 论文说明每个时刻有 64-element mmWave beam-training power vector。
- 最新 ISAC/DeepSense 工作也把 64-D mmWave receive-power vector 当作显式 sensing modality，用 MLP encoder 处理。
- 本地 `dataset/scenario9/unit1/mmWave_data/mmWave_power_0.txt` 正好是 64 行正数，argmax 也就是当前 beam label 的来源。
- 不推荐每条样本做 softmax/min-max 到纯相对分布，因为会丢掉绝对功率/SNR/path-loss 信息；dB + train scaler 更稳，也更保留通信语义。

**应新增的代码面**
当前集成点主要在这些文件：

- [src/kd_sensing/preprocessing/sequences.py](/root/projects/KD-for-sensing/src/kd_sensing/preprocessing/sequences.py): 增加 `include_mmwave`，从 `unit1_pwr_60ghz` 生成 `mmwave1..mmwaveN`，同时保留 `beam1..future_beamN` 作为标签路径。
- [src/kd_sensing/data/samples.py](/root/projects/KD-for-sensing/src/kd_sensing/data/samples.py): `SequenceSamples` 增加 `mmwave_paths`，列校验支持 `mmwave1..N`。
- [src/kd_sensing/data/datasets/scenario9.py](/root/projects/KD-for-sensing/src/kd_sensing/data/datasets/scenario9.py): `VALID_MODALITIES` 加 `mmwave`，返回 `sample["mmwave"]`，shape `[seq_len, 64]`。
- [src/kd_sensing/data/transforms.py](/root/projects/KD-for-sensing/src/kd_sensing/data/transforms.py): 增加 `read_mmwave_power_vector`、`build_mmwave_features`、`load_mmwave_feature_sequence`、`MmWaveStandardScaler`。
- [src/kd_sensing/engine/batch.py](/root/projects/KD-for-sensing/src/kd_sensing/engine/batch.py): 增加 `prepare_mmwave_inputs`，历史 `[B,T,64]` 后面 pad `num_pred-1` 个零步，避免 future mmWave 泄漏。
- 新增 `src/kd_sensing/models/mmwave.py`: `MmWaveFeatureExtractor`、`MmWaveModalityNet`、`MmWaveStudentModalityNet`。
- [src/kd_sensing/models/fusion/networks.py](/root/projects/KD-for-sensing/src/kd_sensing/models/fusion/networks.py): fusion teacher/student 增加 mmWave 分支。
- configs 增加 `configs/mmwave/{teacher_no_kd,student_no_kd,logits_kd,rkd}.yaml`，fusion 配置矩阵扩展包含 `mmwave` 的组合。

**模型建议**
单模态 mmWave 走 GPS 风格的向量序列网络，而不是 CNN：

```text
[B,T,64]
   -> MmWaveFeatureExtractor: Linear(64 -> hidden) + LayerNorm/GELU/Dropout + Linear(hidden -> feature_size)
   -> LayerNorm
   -> GRU
   -> teacher: temporal attention 或 MHA
   -> classifier
```

Student 版用更窄 hidden、单层 GRU、不加 teacher 的增强模块。Fusion student 中 mmWave 分支可以类似 GPS projection：`Linear(64 -> 96) -> ReLU -> fusion_layer`。

**需要注意**
`beam1..N` 现在本质上也是指向这些 mmWave power txt，只是 dataset 读取后取 `argmax` 当标签。新增 mmWave 模态时最好仍单独生成 `mmwave1..N`，语义清晰，也方便以后换 label 来源或做消融。

资料依据：
- DeepSense 6G dataset paper: https://arxiv.org/pdf/2211.09769
- in-band mmWave power as modality: https://arxiv.org/abs/2601.01033
- short-term mmWave power history for multimodal beam prediction: https://arxiv.org/abs/2603.25799
- Scenario 9 real-data summary: https://www.nature.com/articles/s41598-024-70651-9