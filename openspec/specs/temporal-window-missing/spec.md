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
项目 SHALL 提供 `scripts/launch_h5_p1_temporal_models_v1.py`、`scripts/eval_h5_p1_temporal_matrix_v1.py` 和 `scripts/summarize_h5_p1_temporal_matrix_v1.py` 作为 local/manual research workflow。默认 profile MUST 继续覆盖 `ours_c2_main`、`ours_b4_nonrouter_soft_jepa`、`ours_e5_low_lr_pcpg`、`amber_full` 和 `rmbp_mm`；显式 S1 lightweight profile MUST 在同一组参数化脚本内覆盖原有八个筛选方法，并写入独立 ignored output root。用户通过 `--methods` 显式选择时，launcher MUST 额外支持本 change 的 `S1-LG`、`T2-LG`、`S1-CLS` 和 `T2-CLS`。所有 profile 与显式候选 MUST 继续遵守 `H5/P1 跨方法统一数据划分` 的 group-safe sequence split、稳定 identity 和跨 split 重叠审计契约。系统 MUST 不新增 S1-S4 派生 wrapper 或实体配置族。

#### Scenario: 默认 launcher 保持五方法
- **WHEN** 用户运行 H5/P1 launcher 并传入 `--dry_run --seeds 1` 且未选择 S1 profile
- **THEN** manifest MUST 包含原有 5 个方法、`history_window=5` 和 `prediction_window=1`
- **AND** GPU 分配 MUST 尊重 `--gpus`、`--max_jobs` 和 `--per_gpu`

#### Scenario: S1 profile 生成八卡矩阵
- **WHEN** 用户运行 launcher 的 S1 lightweight profile、`--seeds 1 --gpus 0,1,2,3,4,5,6,7 --max_jobs 8 --per_gpu 1 --dry_run`
- **THEN** manifest MUST 包含原有 8 个独立 S1 筛选任务
- **AND** 每个任务 MUST 分配到不同 GPU0-7
- **AND** generated config MUST 使用独立 output dir、相同 group-safe split、epoch、optimizer 和 temporal sampler contract

#### Scenario: S1 profile 使用实测资源默认值
- **WHEN** 用户未显式覆盖 batch、CPU thread 或 DataLoader persistence 并运行 S1 lightweight profile
- **THEN** generated config MUST 使用 train batch64、intra-op 12、inter-op 1 和 persistent workers
- **AND** manifest MUST 记录实际 thread 与 persistent worker 配置
- **AND** 默认 H5/P1 profile MUST 继续使用 intra-op 1 和 non-persistent workers

#### Scenario: 显式生成 T2 几何与 head 候选
- **WHEN** 用户运行 S1 lightweight launcher 并通过 `--methods S1-LG,T2-LG,S1-CLS,T2-CLS --seeds 1` 显式选择候选
- **THEN** manifest MUST 只包含四个候选方法
- **AND** 默认 profile 方法列表 MUST 不改变
- **AND** 所有 config MUST 由现有 base config 与 launcher override 生成并写入 ignored output root
- **AND** 所有候选 MUST 继承相同的 group identity policy，且 train、validation 与 test 的 sequence group、sample、历史帧和 target 帧 identity 审计 MUST 在训练前通过

#### Scenario: 固定 eval mask cache 可复用
- **WHEN** eval script 对 `(missing_rate, drop_count)` cell 求值
- **THEN** 系统 MUST 从 `eval_fixed_mask_cache` 读取或生成固定 JSON mask cache
- **AND** 同一 cache MUST 被不同 method/seed 复用
- **AND** cache MUST 包含 seed、checksum、模态组合覆盖和 `modality_temporal_mask [5,4]`

#### Scenario: summary 输出三类矩阵
- **WHEN** summary script 汇总 eval matrix 输出
- **THEN** 每个方法 MUST 输出 Top1、Within@3 和 MAE 的 5x4 CSV/Markdown 矩阵
- **AND** S1 profile summary MUST 额外保留 Top3、ADBA、teacher/ranking、pooling 和 router diagnostics 中实际可用的字段
- **AND** summary MUST 保留 pattern metrics，并生成方法对比表、guardrail 状态与自动分析段落

### Requirement: H5/P1 跨方法统一数据划分
H5/P1 temporal matrix launcher MUST 为所有方法生成相同的 Scene31-34 数据范围，并 MUST 对 train、validation 和 test 使用相同的场景列表、split protocol、group-safe split strategy、split seed、split source 和 split fractions。方法基配置中的 Scene31-only 字段 MUST 被公共划分覆盖。重叠 temporal window MUST 按稳定 sequence identity 整组分配；逐样本分层拆分 MUST NOT 用于可比较 temporal evidence。

#### Scenario: AMBER 与 RMBP-MM dry-run 使用 Scene31-34
- **WHEN** 用户对 `amber_full` 和 `rmbp_mm` 运行 launcher dry-run
- **THEN** 两个生成配置的 `scenes`、`train_scenes`、`validation_scenes` 和 `test_scenes` MUST 均为 `[31, 32, 33, 34]`
- **AND** 两个配置 MUST 使用相同的 `stratified_80_10_10` group-safe sequence split strategy

#### Scenario: 不同方法共享相同 split contract
- **WHEN** launcher 同时生成 U-Mask、AMBER 和 RMBP-MM 配置
- **THEN** 所有生成配置的场景列表、split seed、source splits、fractions 和 group identity policy MUST 完全一致

#### Scenario: Temporal split 身份不相交
- **WHEN** H5/P1 workflow 生成 train、validation 和 test split artifact
- **THEN** 任意两个 split 的 sequence group identity MUST 两两不相交
- **AND** 任意两个 split 实际引用的 sample、历史输入帧和 target 帧 identity MUST 两两不相交
- **AND** 发现交集时 workflow MUST 在训练前失败并报告交集类型和有限示例

#### Scenario: 逐样本 temporal split 被拒绝
- **WHEN** H5/P1 或其它重叠 temporal window 配置请求逐样本 label-stratified split
- **THEN** 配置解析或 dataloader 构建 MUST 拒绝该策略用于 validation/test evidence
- **AND** 错误 MUST 指向 group-safe sequence strategy

### Requirement: Temporal superset observation preservation
temporal missing operator MUST 在显式 opt-in 时保存 zero-fill 前的历史输入 tensor 引用和采样前已有有效性 base mask，供同一 primary model 的在线 stop-gradient superset forward 使用。Student batch MUST 继续按 partial mask 零填充；保存行为 MUST 不 clone tensor、不读取 future/target，也不得在未启用时增加 batch payload。

#### Scenario: Partial 与 superset 具有包含关系
- **WHEN** temporal sampler 产生 student mask `M-` 且 superset preservation 启用
- **THEN** operator MUST 暴露 `M+` 且逐 cell 满足 `M- subseteq M+`
- **AND** 两个分支 MUST 使用同一 sample、history window 和 target
- **AND** `M+` 与 `M-` 均 MUST 至少包含一个有效 cell

#### Scenario: 保存原输入不复制 storage
- **WHEN** operator 保存 zero-fill 前输入
- **THEN** 保存值 MUST 与 operator 输入 tensor 共享 storage
- **AND** student batch 对应模态 MUST 仍在缺失 cell 为零

