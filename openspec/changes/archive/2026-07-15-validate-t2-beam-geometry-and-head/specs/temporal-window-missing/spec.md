## MODIFIED Requirements

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
