## ADDED Requirements

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

### Requirement: temporal missing 本地脚本
项目 SHALL 提供 `scripts/check_temporal_window_missing.py`、`scripts/launch_temporal_missing_v1.py` 和 `scripts/summarize_temporal_missing_v1.py` 作为 local/manual 或 research diagnostic 脚本。脚本 MUST 遵守 outputs 边界，launcher MUST 支持 dry-run 输出命令，summary MUST 写出 CSV/Markdown 汇总。

#### Scenario: 数据检查脚本输出 mask 统计
- **WHEN** 用户运行 `conda run -n kd_mm_beam python scripts/check_temporal_window_missing.py --history_window 5 --prediction_window 1 --temporal_missing_mode modality_frame_bernoulli --temporal_missing_prob 0.3 --num_samples 16`
- **THEN** 脚本 MUST 输出 batch shape、mask shape、可用率、全缺失检查、history indices 和 target index 示例

#### Scenario: launcher dry-run
- **WHEN** 用户运行 temporal missing launcher 并传入 `--dry_run`
- **THEN** 脚本 MUST 打印可运行训练命令
- **AND** MUST 不启动训练或写入 checkpoint

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

### Requirement: S1-S4 temporal router workflow
项目 SHALL 提供 `scripts/launch_temporal_router_s1_s4_v1.py`、`scripts/eval_temporal_router_s1_s4_matrix_v1.py` 和 `scripts/summarize_temporal_router_s1_s4_v1.py` 作为 local/manual research workflow。该 workflow MUST 默认覆盖 `s1_temporalagg_modality_router`、`s2_pertime_modality_router`、`s3_two_level_router`、`s4_global_modality_time_router`、`amber_full` 和 `rmbp_mm`，并写入 ignored `outputs/temporal_router_s1_s4_v1/`。

#### Scenario: S1-S4 显式 opt-in
- **WHEN** 用户未配置 `temporal_router_type`
- **THEN** `u_mask_beam_jepa` MUST 保持既有时间 mean 行为
- **AND** S1-S4 temporal routing MUST 只在配置显式设置对应 `temporal_router_type` 时启用

#### Scenario: S1-S4 gate mask 语义
- **WHEN** 模型接收 `modality_temporal_mask [B,5,4]`
- **THEN** S1 gate MUST 形如 `[B,M]`
- **AND** S2/S3 modality gate MUST 形如 `[B,T,M]`
- **AND** S3 temporal gate MUST 形如 `[B,T]`
- **AND** S4 global gate MUST 形如 `[B,T,M]`
- **AND** 所有不可用 modality/time/cell 权重 MUST 为 0，单可用候选权重 MUST 为 1

#### Scenario: S1-S4 oracle fallback 可审计
- **WHEN** `router_supervision=oracle`
- **THEN** oracle target MUST 只选择 mask 标记可用的 modality、time 或 cell
- **AND** tie MUST deterministic
- **AND** 若使用 hard target 而不是 soft target，diagnostics MUST 记录 hard-target fallback

#### Scenario: S1-S4 launcher dry-run
- **WHEN** 用户运行 S1-S4 launcher 并传入 `--dry_run --seeds 1 --methods s1,s2,s3,s4,amber_full,rmbp_mm`
- **THEN** manifest MUST 包含 6 个方法、`history_window=5` 和 `prediction_window=1`
- **AND** GPU 分配 MUST 尊重 `--gpus`、`--max_jobs` 和 `--per_gpu`

#### Scenario: S1-S4 固定 eval mask cache
- **WHEN** eval script 对六个方法求值
- **THEN** 所有 method/seed MUST 使用同一个 `eval_fixed_mask_cache`
- **AND** 每个 `(rate, drop_count)` cell MUST 至少包含配置数量的 mask
- **AND** cache MUST 记录 seed 和 checksum，且不允许全缺

#### Scenario: S1-S4 summary 输出
- **WHEN** summary script 汇总 S1-S4 eval matrix 输出
- **THEN** 每个方法 MUST 输出 Top1、Within@3 和 MAE 的 5x4 CSV/Markdown 矩阵
- **AND** summary MUST 输出方法对比总表、router diagnostics、pattern metrics 和自动分析段落
