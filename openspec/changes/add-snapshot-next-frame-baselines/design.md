## Context

现有默认实验以 `seq_len=8`、`num_pred=3` 为主，单模态 `modular_sequence` 使用 `single_gru`，legacy fusion 使用 GRU，推荐 fusion 虽然已可使用 `cls_token_transformer_fusion`，但仍消费历史窗口 token。用户现在需要一组“无历史窗口、只预测下一帧、去掉 GRU”的单模态和多模态基线，用来隔离历史窗口对模态强弱排序的影响。

arXiv:2603.25799 的对比对象是 Scenario 31 上的 snapshot 式多模态框架：每个样本以同步 RGB、LiDAR、radar、GNSS 和短期 mmWave power history 构成观测。本项目的 snapshot baseline 应保持现有 future-only label 语义：输入当前帧 `t`，预测 `t+1`，即 `seq_len=1` 且 `num_pred=1`。为了更严格对齐 snapshot 数据口径，同时按当前实验需要采用 80/20 train/validation，本方案不默认复用 `seq_len=8/out_len=3` 生成的历史窗口 CSV，而是从 Scenario 31 原始/RA CSV 重新生成 `in_len=1/out_len=1` 的 snapshot 专用 CSV。

## Goals / Non-Goals

**Goals:**

- 提供可通过配置运行的五个单模态 snapshot no-KD baseline：`image`、`radar`、`gps`、`lidar`、`mmwave`。
- 提供多模态 snapshot fusion no-KD baseline，至少覆盖五模态 `image_radar_gps_lidar_mmwave`，并复用现有合法 fusion slug 解析能力。
- 提供 snapshot 专用预处理配置，生成 `in_len=1/out_len=1` 的 Scenario 31 窗口、完整 `seq_index` 级 80/20 train/validation split 和可机器读取 metadata。
- 让 snapshot 配置默认使用 snapshot 专用 CSV，并拒绝在未明确标记的情况下回退到历史窗口 CSV。
- 明确禁止 snapshot baseline 使用 GRU、RNN、TCN、跨时间 attention 或 `seq_len>1` 的历史输入。
- 复用现有 Dataset、模态 loader、encoder、目标、loss、metric、evaluation pass 和输出目录语义，避免新增训练脚本。
- 在输出 metadata/final config 中记录 snapshot/no-history 语义、split 协议、CSV 名称、样本数和统计 artifact fingerprint，便于与历史窗口实验按明确口径横向比较。

**Non-Goals:**

- 不复现论文中的完整三任务 Transformer、SLAM-style LiDAR map 可视化或论文表格的全部训练超参。
- 不删除现有 GRU、CLS-token Transformer 历史窗口、KD、CRAF、MARF、G2D 或多任务 objective。
- 不改变默认 canonical 历史窗口配置；snapshot 是新增实验入口。
- 不把 `seq_len=1` 解释成“无当前帧”，而是解释成“只使用当前观测帧，没有更早历史帧”。
- 不强制历史窗口 baseline 改用 snapshot split；历史窗口结果和 snapshot 结果必须通过 metadata 标记不同 split 协议。

## Decisions

### 1. 使用 snapshot 专用预处理生成 `in_len=1/out_len=1` 数据口径

新增 `configs/preprocess/sequences_snapshot_next_frame.yaml`，默认读取 `dataset/scenario31/scenario31_RA.csv`，设置 `in_len: 1`、`out_len: 1`、`include_gps: true`、`include_lidar: true`、`include_mmwave: true`，并在需要 position objective 时启用 `include_position_targets: true`。输出建议命名为：

- `train_seqs_SNAPSHOT_NEXT_FRAME.csv`
- `val_seqs_SNAPSHOT_NEXT_FRAME.csv`
- `split_metadata_SNAPSHOT_NEXT_FRAME.json`

split 必须以完整 `seq_index` 为单位，默认目标比例为 80% train 和 20% validation。metadata 必须记录 source CSV、scene、`in_len`、`out_len`、split seed、train/validation `seq_index`、窗口数、label 分布摘要、启用列和输出 CSV 路径。这样比复用 `seq_len=8` CSV 更接近 per-snapshot 评估口径，也不会丢掉每条序列开头可作为 snapshot 输入的样本。

替代方案是继续复用现有 `train_seqs_RA_GPS_LIDAR.csv` / `test_seqs_RA_GPS_LIDAR.csv` 并只在 config 中设置 `seq_len=1`。该方案可以快速消融历史窗口，但会继承 `in_len=8` 预处理产生的窗口集合，样本数和序列覆盖不符合更严格的 snapshot 口径，因此不作为默认方案。

### 2. 使用 `seq_len=1` 与 `num_pred=1` 表达模型输入输出

snapshot 配置将 `data.dataset.seq_len`、`data.dataset.num_pred`、`model.seq_length_teacher`、`model.seq_length_student`、`model.num_pred` 全部设置为 `1`。现有 batch preparer 在 `num_pred=1` 时不会追加 future zero padding，因此模型实际只接收一个当前帧 slot。labels 继续来自 `future_beam1`，occlusion 和 position 目标来自同一个 future slot。

### 3. 新增模块化 `snapshot_frame` representation core

在 `modular_sequence` 中新增 `snapshot_frame` core。它接收单模态 `[B, 1, D]` 或多模态 `[B, K, 1, D]`，输出 `[B, 1, D_out]`，供现有 beam head 和 auxiliary heads 生成 `[B, 1, C]`、`[B, 1]`、`[B, 1, 2]`。该 core 必须在 forward 时拒绝 `T != 1`，并且不包含 GRU/RNN/TCN 或任何跨时间 attention。

单模态路径默认使用 LayerNorm/Dropout/MLP 或 identity-style projection 处理当前帧 embedding。多模态路径默认使用当前帧模态 token 融合，可选 `concat_mlp` 或只在 `K` 个模态 token 间运行的浅层 self-attention；该 attention 不带 time embedding，也不接收多个时间步。

替代方案是直接用 `cls_token_transformer_fusion` 并把 `seq_len` 设为 1。它可作为可行实现的一部分，但单模态仍需要无 GRU core，而且现有 CLS 模型包含面向历史窗口的 time embedding 语义。新增 `snapshot_frame` 能把实验语义写进模型契约和错误检查里。

### 4. snapshot 配置作为 no-KD baseline family

新增配置命名：

- 单模态：`configs/<modality>/snapshot_next_frame_no_kd.yaml`
- 多模态：`configs/fusion/<slug>_snapshot_next_frame_no_kd.yaml`
- 五模态兼容别名：`configs/fusion/all_modalities_snapshot_next_frame_no_kd.yaml`

这些配置默认 `distillation.type: no_kd`，`experiment.objective` 默认为 `beam`，设置 `experiment.variant: snapshot_next_frame` 或等价 metadata，并默认指向 snapshot 专用 CSV。若用户显式配置 `occlusion`、`position` 或 `multitask`，snapshot 模型必须继续通过现有 auxiliary head 和 target provider 契约工作，并要求对应 snapshot CSV 包含 future target 列。

替代方案是通过命令行覆盖现有配置的 `seq_len=1` 并手工改模型 core。该方案不可发现、不可审计，也无法保证“去掉 GRU”，因此不作为正式入口。

### 5. 复用现有 canonical/virtual config 解析

`build_virtual_config()` 继续作为缺失 canonical 配置的入口，新增 snapshot stem 解析逻辑。解析顺序应避免和现有 objective/fusion mode 冲突：先识别 advanced overlay，再识别 snapshot suffix，再识别 objective suffix和普通 KD mode。实体 YAML 仍优先于 virtual 生成。

单模态 snapshot 可以通过实体 YAML 或新增 virtual builder 实现；fusion snapshot 应优先通过 virtual builder 覆盖所有合法多模态 slug，减少 26 组组合的实体 YAML 维护成本。

### 6. split-dependent cache/stat artifact 必须按 snapshot 协议隔离

原始帧级 cache 可以复用：例如 LiDAR BEV cache 只依赖 LiDAR 文件路径和 BEV 参数时，不需要因为 snapshot split 复制一份。但所有依赖训练 split 的统计量必须按 snapshot 协议重新 fit 并保存，包括：

- mmWave z-score scaler
- GPS scaler
- LiDAR streaming stats normalizer
- occlusion threshold `tau`
- position target scaler

artifact metadata 必须记录 source CSV、split metadata path、train CSV path、`seq_len`、`num_pred`、enabled modalities、相关预处理参数和一个稳定 fingerprint。评估 snapshot checkpoint 时必须优先加载同一运行保存的 snapshot artifact，不得复用历史窗口 run 的 scaler/threshold。

### 7. 比较口径由 metadata 明确记录

训练与评估输出需要记录 `prediction_setup` 或等价字段，包括 `variant=snapshot_next_frame`、`uses_history_window=false`、`uses_temporal_core=false`、`seq_len=1`、`num_pred=1`、`modalities`、`objective`、`scene`、split protocol、train/validation CSV 与样本数、split metadata 路径和统计 artifact fingerprint。这样后续汇总脚本可以拒绝把 snapshot 与历史窗口结果误合并，或者显式分组比较模态强弱。

## Risks / Trade-offs

- [Risk] 新增显式 validation CSV 会触及当前以 `test_csv_name` 承担验证集的流程。→ Mitigation：snapshot 配置必须明确 `val_csv_name`；实现可兼容地将 validation loader 映射到现有验证路径，但 metadata 中不得把 validation 称为 test。
- [Risk] snapshot split 与历史窗口 split 不再是同一窗口集合，直接比较会混入数据口径差异。→ Mitigation：结果表必须展示 split protocol 和样本数；需要“同窗口消融”时可额外运行旧 CSV override，但不得标记为 paper-aligned snapshot。
- [Risk] mmWave 当前帧输入预测下一帧仍是“一步 radio context”，可能强于完全无 radio history 的设定。→ Mitigation：把该设定明确记录为 `current_frame_mmwave_to_next_frame`，后续可追加 `no_mmwave_history` ablation，但不混入本 change。
- [Risk] split-dependent scaler 或 threshold 误复用历史窗口 artifact 会造成数据泄漏或指标漂移。→ Mitigation：artifact metadata 必须包含 snapshot split fingerprint；不匹配时评估入口拒绝加载。
- [Risk] 多模态 snapshot token attention 可能被误认为时间 Transformer。→ Mitigation：core 命名、配置和 forward 校验必须强调只在 `K` 个当前帧模态 token 间融合，并拒绝 `T>1`。
- [Risk] 复制单模态配置会造成 YAML 漂移。→ Mitigation：优先复用 canonical recipe/helper 生成 snapshot 配置，实体 YAML 只保留常用入口或别名。
- [Risk] 与论文指标仍不可逐项等价，因为模型、loss 和部分任务定义不同。→ Mitigation：文档中说明本 change 是 paper-aligned data protocol 与 no-history/no-GRU baseline，不声称完全复现论文训练协议。

## Migration Plan

1. 新增 snapshot 专用预处理配置和 80/20 train/validation split metadata，补预处理测试。
2. 新增或适配 validation split dataloader 支持，确保 early stopping 和训练期验证使用 `val_seqs_SNAPSHOT_NEXT_FRAME.csv`。
3. 新增 `snapshot_frame` core 与配置校验，补 forward shape 单元测试。
4. 新增单模态和 fusion snapshot 配置生成逻辑，默认指向 snapshot 专用 CSV，补配置加载测试。
5. 补 split-dependent normalization/cache artifact fingerprint 和 README 命令示例。
6. 运行最小 smoke test：单模态至少 `gps` 或 `mmwave`，fusion 至少五模态，均使用 `conda run -n kd_mm_beam`。
7. 回滚策略：snapshot 是新增预处理配置、注册名和配置入口，若出现问题可删除/禁用 snapshot 配置解析，不影响既有历史窗口配置。

## Open Questions

- 是否需要第一批就提供 `occlusion`、`position`、`multitask` 的 snapshot canonical 命名矩阵？默认先保证 beam baseline；模型输出契约保留 objective 兼容能力。
- snapshot 预处理的默认 split seed 是否沿用当前项目的 `42`，还是单独设置 paper-aligned seed 并固定在文档中？默认沿用 `42`，metadata 必须记录。
