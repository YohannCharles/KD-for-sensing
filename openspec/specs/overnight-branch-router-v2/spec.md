# overnight-branch-router-v2 Specification

## Purpose
记录 overnight branch-router v2 的 soft hard-subset weighting、supervised router、launcher、summary 和 focused regression 边界，作为 post-C2 历史/支撑实验上下文。
## Requirements
### Requirement: Soft hard-subset weighting
系统 MUST 支持显式 opt-in 的 `soft_static` hard subset weighting。该模式 MUST 按缺失 pattern 名称或 drop-count alias 返回固定权重：`full=0.75`、`drop1/miss1=1.0`、`drop2/miss2=1.15`、`drop3/miss3=1.35`、`radar_only=1.50`、`missing_image=1.35`，其它已知或未知 pattern 默认 `1.0`，且不得产生 NaN。

#### Scenario: 解析 soft static 权重
- **WHEN** 用户启用 `--hard_subset_weighting soft_static`
- **THEN** 训练侧 MUST 使用 soft static 权重表计算样本权重
- **AND** unknown pattern MUST fallback 到 `1.0`
- **AND** run config 与 summary MUST 记录 `hard_subset_weighting=soft_static`

### Requirement: Supervised pattern-aware router
系统 MUST 支持显式 opt-in 的 `supervised_router` fusion。该 fusion MUST 支持 `router_supervision=oracle|pattern_best|none`、`router_distill_weight`、`router_distill_temperature`、`router_focus_patterns` 和 `router_fuse_level=logits`，并且 MUST 保持普通 PCPG/BPRR 默认行为不变。

#### Scenario: oracle target 只选可用模态
- **WHEN** 训练或验证样本包含 unimodal logits、真实 beam label 和 available mask
- **THEN** oracle target MUST 只从可用模态中选择 beam 距离最小的模态
- **AND** tie MUST 按 CE loss、confidence 和固定模态顺序 deterministic 处理
- **AND** oracle target MUST 只用于 router distill，不得参与最终预测作弊

#### Scenario: masked gate 合法
- **WHEN** router 计算 gate
- **THEN** 不可用模态 gate MUST 为 `0`
- **AND** 单模态可用时该模态 gate MUST 为 `1`
- **AND** 多模态可用时 gate MUST sum 为 `1`
- **AND** gate 不得产生 NaN

#### Scenario: focus pattern distill
- **WHEN** `router_focus_patterns` 包含 `missing_image,miss2,drop2`
- **THEN** router distill MUST 只在这些 focus pattern alias 上启用
- **AND** full 与 radar_only 单模态 pattern MUST 不启用 distill

### Requirement: Router diagnostics artifacts
supervised router run MUST 保存真实 router gate 诊断、oracle target distribution 和 router accuracy。诊断指标至少 MUST 覆盖 mean gate、gate entropy、router oracle accuracy、focus pattern oracle accuracy、oracle target rate，以及 missing_image/drop2 下的 radar gate。

#### Scenario: supervised router metrics 可汇总
- **WHEN** supervised router run 完成评估
- **THEN** output dir 中 MUST 存在可被 summary 解析的 router diagnostics
- **AND** oracle eval 或 oracle target 诊断 MUST 不混入真实方法 ranking

### Requirement: Overnight launcher matrix
系统 MUST 提供 `scripts/launch_overnight_branch_router_v2.py`，默认生成 A/B/C 三组 40 个 job：A 组 `a1/a2` 使用 anchor seeds `1,2,3,4,5`，B/C 组使用 explore seeds `1,2,3`。launcher MUST 默认只使用 GPU `1,2`，每张 GPU 最多 2 个进程，总并发最多 4 个。

#### Scenario: dry-run manifest
- **WHEN** 用户以默认 seed 与 `--dry_run` 运行 launcher
- **THEN** launcher MUST 写出 `job_manifest.csv`
- **AND** manifest MUST 包含 40 个 job
- **AND** job 只能分配到 GPU 1 或 GPU 2
- **AND** 每张 GPU 并发计划不得超过 2，总并发不得超过 4

#### Scenario: job 失败不杀其它 job
- **WHEN** 某个已启动 job 返回非零
- **THEN** launcher MUST 继续等待其它已启动 job 完成
- **AND** 所有 job 结束后 MUST 写出 `failed_jobs.csv`
- **AND** 存在失败 job 时 launcher MUST 返回非零 exit code

### Requirement: Overnight summary outputs
系统 MUST 提供 `scripts/summarize_overnight_branch_router_v2.py`，能读取当前 overnight root 和 baseline roots，写出 `summary.csv`、`summary.md`、`drop_count_summary.csv`、`pattern_metrics.csv` 和 `router_diagnostics.csv`。

#### Scenario: 生成自动结论
- **WHEN** summary 脚本读取完成的 run metrics
- **THEN** `summary.md` MUST 包含 mean/std 主表、delta 表、e6 来源拆解、supervised router 诊断和推荐结论
- **AND** 缺失指标 MUST 保持为空或明确标记 unavailable，不得伪造结果

### Requirement: Focused regression coverage
本 change MUST 提供 focused tests 覆盖 soft static weighting、oracle router target、masked softmax、focus pattern alias、launcher dry-run 和 summary parser。测试 MUST 使用合成或 fake artifact，不得读取真实 `dataset/` 或写入受保护 runtime 产物。

#### Scenario: focused pytest
- **WHEN** 开发者运行 `conda run -n kd_mm_beam pytest -q tests/test_overnight_branch_router_v2.py`
- **THEN** 测试 MUST 覆盖新增 helper、launcher dry-run 和 summary parser
- **AND** 测试产物 MUST 写入 pytest 临时目录或 ignored output root
