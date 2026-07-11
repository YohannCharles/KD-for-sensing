# temporal-window-missing Specification

## Purpose
定义训练、评估和 H5/P1 temporal matrix 使用的显式历史/预测窗口、时序缺失掩码、在线分层采样及可复用评估缓存契约。
## Requirements
### Requirement: 显式窗口配置
系统 MUST 支持 `history_window` 和 `prediction_window` 作为一等配置与 CLI 参数。`history_window` MUST 与现有 dataset `seq_len` 和 model `seq_length` 对齐；`prediction_window` MUST 与 dataset/model `num_pred` 对齐。未显式设置时，默认窗口 MUST 为 `history_window=5`、`prediction_window=1`。

#### Scenario: CLI 窗口参数同步
- **WHEN** 用户运行训练或评估入口并传入 `--history_window 5 --prediction_window 1`
- **THEN** resolved config MUST 记录 `history_window=5` 和 `prediction_window=1`
- **AND** dataset `seq_len`、model `seq_length`、dataset/model `num_pred` MUST 分别同步为 5 和 1

#### Scenario: 新实验默认启用 temporal missing
- **WHEN** 配置不包含 `history_window`、`prediction_window` 或 temporal missing 字段
- **THEN** 系统 MUST 使用 `history_window=5` 和 `prediction_window=1`
- **AND** `temporal_missing_mode` MUST resolve to `modality_frame_bernoulli`
- **AND** `temporal_missing_prob` MUST resolve to `0.2`
- **AND** 用户 MUST 可以通过 `temporal_missing_mode=none` 显式关闭时序缺失

### Requirement: 时序缺失 mask contract
系统 MUST 支持 `temporal_missing_mode` 为 `none`、`frame_bernoulli`、`modality_frame_bernoulli`、`block` 或 `stratified_modality_temporal`。除 `none` 外，operator MUST 产生 `temporal_mask [B,T]`、`modality_temporal_mask [B,T,M]`、`available_modalities [B,M]` 和统计 metadata，并将不可用输入位置置零。

#### Scenario: 整帧随机缺失
- **WHEN** `temporal_missing_mode=frame_bernoulli`
- **THEN** 被采样为缺失的时间步 MUST 对所有启用模态置零
- **AND** 对应 `temporal_mask` 与 `modality_temporal_mask` MUST 为 false

#### Scenario: 模态-时间粒度缺失
- **WHEN** `temporal_missing_mode=modality_frame_bernoulli`
- **THEN** 系统 MUST 允许某一历史帧只缺失某个模态
- **AND** `temporal_mask[t]` MUST 等于该时间步任一模态可用
- **AND** `available_modalities[m]` MUST 等于该模态任一历史时间步可用

#### Scenario: 连续块缺失
- **WHEN** `temporal_missing_mode=block` 且 `temporal_missing_block_len=2`
- **THEN** 系统 MUST 在历史窗口内采样连续 2 个时间步置为不可用
- **AND** block MUST 不越界

#### Scenario: 在线分层模态-时序缺失
- **WHEN** `mask_sampler=stratified_modality_temporal` 或 `temporal_missing_mode=stratified_modality_temporal`
- **THEN** 每个训练样本 MUST 近似均匀采样 `drop_count`、`temporal_missing_rate` 和 `temporal_missing_type`
- **AND** sampler MUST 支持 `modality_level`、`frame_level`、`modality_frame` 和 `block`
- **AND** sampler MUST 不预生成或枚举 20 个 modality-time cell 的所有组合

### Requirement: 时序缺失保底与组合
系统 MUST 避免全窗口全模态缺失。若 `ensure_at_least_one_frame=true` 且采样结果全缺失，系统 MUST 恢复至少一个时间步和一个模态或最后一帧的可用模态，并在 metadata 记录 `num_all_missing_fixed`。已有 modality missing MUST 先作用，temporal missing MUST 后作用，最终 mask MUST 表达两者组合。

#### Scenario: 保底修复全缺失
- **WHEN** temporal missing 参数会导致样本全窗口全模态缺失
- **THEN** 系统 MUST 修复该样本使至少一个模态时间位置可用
- **AND** metadata MUST 记录修复次数

#### Scenario: 与模态缺失组合
- **WHEN** 既启用 random modality dropout 又启用 temporal missing
- **THEN** 被模态 mask 掉的模态 MUST 在所有时间步不可用
- **AND** 被 temporal mask 掉的时间步 MUST 在所有模态不可用
- **AND** target 字段 MUST 保持不变

### Requirement: H5/P1 temporal matrix workflow
项目 SHALL 提供 `scripts/launch_h5_p1_temporal_models_v1.py`、`scripts/eval_h5_p1_temporal_matrix_v1.py` 和 `scripts/summarize_h5_p1_temporal_matrix_v1.py` 作为 local/manual research workflow。该 workflow MUST 默认覆盖 `ours_c2_main`、`ours_b4_nonrouter_soft_jepa`、`ours_e5_low_lr_pcpg`、`amber_full` 和 `rmbp_mm`，并写入 ignored `outputs/h5_p1_temporal_models_v1/`。

#### Scenario: launcher dry-run 生成 5 方法作业
- **WHEN** 用户运行 H5/P1 launcher 并传入 `--dry_run --seeds 1`
- **THEN** manifest MUST 包含 5 个方法、`history_window=5` 和 `prediction_window=1`
- **AND** GPU 分配 MUST 尊重 `--gpus`、`--max_jobs` 和 `--per_gpu`

#### Scenario: 固定 eval mask cache 可复用
- **WHEN** eval script 对 `(missing_rate, drop_count)` cell 求值
- **THEN** 系统 MUST 从 `eval_fixed_mask_cache` 读取或生成固定 JSON mask cache
- **AND** 同一 cache MUST 被不同 method/seed 复用
- **AND** cache MUST 包含 seed、checksum、模态组合覆盖和 `modality_temporal_mask [5,4]`

#### Scenario: summary 输出三类矩阵
- **WHEN** summary script 汇总 eval matrix 输出
- **THEN** 每个方法 MUST 输出 Top1、Within@3 和 MAE 的 5x4 CSV/Markdown 矩阵
- **AND** summary MUST 保留 pattern metrics，并生成方法对比表与自动分析段落
