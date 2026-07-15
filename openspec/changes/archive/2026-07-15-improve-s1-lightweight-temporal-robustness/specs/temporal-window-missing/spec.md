## ADDED Requirements

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

## MODIFIED Requirements

### Requirement: H5/P1 temporal matrix workflow
项目 SHALL 提供 `scripts/launch_h5_p1_temporal_models_v1.py`、`scripts/eval_h5_p1_temporal_matrix_v1.py` 和 `scripts/summarize_h5_p1_temporal_matrix_v1.py` 作为 local/manual research workflow。默认 profile MUST 继续覆盖 `ours_c2_main`、`ours_b4_nonrouter_soft_jepa`、`ours_e5_low_lr_pcpg`、`amber_full` 和 `rmbp_mm`；显式 S1 lightweight profile MUST 在同一组参数化脚本内覆盖本 change 的筛选方法，并写入独立 ignored output root。系统 MUST 不新增 S1-S4 派生 wrapper。

#### Scenario: 默认 launcher 保持五方法
- **WHEN** 用户运行 H5/P1 launcher 并传入 `--dry_run --seeds 1` 且未选择 S1 profile
- **THEN** manifest MUST 包含原有 5 个方法、`history_window=5` 和 `prediction_window=1`
- **AND** GPU 分配 MUST 尊重 `--gpus`、`--max_jobs` 和 `--per_gpu`

#### Scenario: S1 profile 生成八卡矩阵
- **WHEN** 用户运行 launcher 的 S1 lightweight profile、`--seeds 1 --gpus 0,1,2,3,4,5,6,7 --max_jobs 8 --per_gpu 1 --dry_run`
- **THEN** manifest MUST 包含 8 个独立 S1 筛选任务
- **AND** 每个任务 MUST 分配到不同 GPU0-7
- **AND** generated config MUST 使用独立 output dir、相同 split、epoch、optimizer 和 temporal sampler contract

#### Scenario: S1 profile 使用实测资源默认值
- **WHEN** 用户未显式覆盖 batch、CPU thread 或 DataLoader persistence 并运行 S1 lightweight profile
- **THEN** generated config MUST 使用 train batch64、intra-op 12、inter-op 1 和 persistent workers
- **AND** manifest MUST 记录实际 thread 与 persistent worker 配置
- **AND** 默认 H5/P1 profile MUST 继续使用 intra-op 1 和 non-persistent workers

#### Scenario: 固定 eval mask cache 可复用
- **WHEN** eval script 对 `(missing_rate, drop_count)` cell 求值
- **THEN** 系统 MUST 从 `eval_fixed_mask_cache` 读取或生成固定 JSON mask cache
- **AND** 同一 cache MUST 被不同 method/seed 复用
- **AND** cache MUST 包含 seed、checksum、模态组合覆盖和 `modality_temporal_mask [5,4]`

#### Scenario: summary 输出三类矩阵
- **WHEN** summary script 汇总 eval matrix 输出
- **THEN** 每个方法 MUST 输出 Top1、Within@3 和 MAE 的 5x4 CSV/Markdown 矩阵
- **AND** S1 profile summary MUST 额外保留 Top3、ADBA、teacher/ranking、pooling 和 router diagnostics 中实际可用的字段
- **AND** summary MUST 保留 pattern metrics，并生成方法对比表与 guardrail 状态
