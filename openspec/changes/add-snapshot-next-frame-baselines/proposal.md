## Why

当前单模态和多模态实验默认使用历史窗口与 GRU/时序 core，无法把“当前帧感知能力”与“历史窗口带来的收益”分开评估。为了和 arXiv:2603.25799 的 snapshot 式多模态基线对齐，并研究历史窗口对强弱模态结论的影响，需要新增无历史窗口、只预测下一帧、无 GRU 的可复现实验族。

## What Changes

- 新增 snapshot next-frame baseline 语义：输入只包含当前帧 `seq_len=1`，预测目标只包含下一帧 `num_pred=1`，模型不得使用 GRU、RNN、TCN 或跨时间 attention。
- 新增论文对齐的 snapshot 专用预处理配置，使用 `in_len=1/out_len=1` 在 Scenario 31 上重新生成 snapshot 窗口，保留 RGB、radar、GPS/BS GPS、LiDAR、mmWave、future beam 和可选 future position target 列。
- 新增 snapshot 专用序列级 train/validation split metadata，默认按 80%/20% 的完整 `seq_index` 划分，避免复用 `seq_len=8` CSV 丢失序列开头样本。
- 为 `image`、`radar`、`gps`、`lidar`、`mmwave` 提供单模态 snapshot no-KD 配置入口，沿用现有模态 loader、encoder、normalization 和 metric 契约。
- 为合法多模态组合，至少包括五模态 `image_radar_gps_lidar_mmwave`，提供 snapshot fusion no-KD 配置入口；默认融合只在当前帧的模态 token 间交互，不建模历史时间。
- 在模块化模型中新增无时序 representation core，用于单模态 pass-through/MLP 和多模态当前帧 token fusion，输出仍兼容现有 `[B, H, C]` future-slot 训练与评估流程。
- 在 canonical/virtual 配置解析中新增 snapshot 命名规则，生成的配置必须显式记录 `data.dataset.seq_len: 1`、`data.dataset.num_pred: 1`、`model.seq_length_*: 1`、`model.num_pred: 1` 和 snapshot/no-history 元数据。
- snapshot 配置默认指向 snapshot 专用 CSV，不再默认复用历史窗口统一 CSV；LiDAR/mmWave/occlusion/position 等 split-dependent cache 或统计 artifact 必须基于 snapshot train split 重新 fit 并记录 fingerprint。
- 文档和测试增加“历史窗口 GRU baseline vs snapshot no-GRU baseline”的可比性要求，确保两类实验明确标记各自 split 协议、objective 和指标口径。
- 不移除现有 GRU 历史窗口模型、KD 配置、CRAF/MARF/G2D 或多任务 objective。

## Capabilities

### New Capabilities

- `snapshot-next-frame-baselines`: 定义无历史窗口、只预测下一帧、无 GRU 的单模态和多模态融合 baseline 契约、配置命名和可比性要求。

### Modified Capabilities

- `modular-sequence-model`: 模块化模型需要支持无时序 snapshot representation core，用于单模态当前帧预测和多模态当前帧融合。
- `configurable-multimodal-fusion`: fusion 配置语义需要新增 snapshot fusion no-KD 入口，并保证多模态融合不依赖 GRU 或历史窗口。
- `modality-aware-data-loading`: 序列预处理、split metadata 和 split-dependent cache/stat artifact 需要支持论文对齐的 `in_len=1/out_len=1` snapshot 协议。
- `canonical-config-resolution`: virtual/canonical 配置解析需要识别并生成 snapshot next-frame 单模态和 fusion 配置。
- `experiment-workflow`: 训练、验证、评估和输出 metadata 需要记录 snapshot/no-history 语义，并支持 snapshot 协议下 80/20 train/validation 的评估口径。

## Impact

- 影响代码：
  - `src/kd_sensing/models/modular.py`
  - `src/kd_sensing/models/fusion/cls_token_transformer.py` 或新增轻量 snapshot fusion core
  - `src/kd_sensing/config/canonical.py`
  - `src/kd_sensing/config/canonical_recipes/`
  - `src/kd_sensing/config/io.py`
  - `src/kd_sensing/preprocessing/sequences.py`
  - `src/kd_sensing/engine/data_factory.py`
  - `src/kd_sensing/engine/normalization_artifacts.py`
  - `src/kd_sensing/engine/run_metadata.py`
  - README/docs 中的实验命令与结果解释
- 影响配置：
  - 新增或虚拟生成 `configs/<modality>/snapshot_next_frame_no_kd.yaml`
  - 新增或虚拟生成 `configs/fusion/<slug>_snapshot_next_frame_no_kd.yaml`
  - 新增 `configs/preprocess/sequences_snapshot_next_frame.yaml`
  - snapshot 默认 Scenario 31 split 使用 `train_seqs_SNAPSHOT_NEXT_FRAME.csv` / `val_seqs_SNAPSHOT_NEXT_FRAME.csv`
- 影响测试：需要覆盖 snapshot core forward shape、单模态与 fusion 配置加载、无 GRU 校验、snapshot 预处理输出、`seq_len=1/num_pred=1` 标签对齐、cache/stat artifact 隔离、训练/验证/评估 smoke test 和 metadata 记录。
- API 兼容：现有配置、checkpoint 和输出路径语义保持兼容；snapshot 配置是新增入口，不改变默认历史窗口实验。
