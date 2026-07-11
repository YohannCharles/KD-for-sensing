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

### Requirement: Overnight summary outputs
项目 MUST 保留 `scripts/summarize_overnight_branch_router_v2.py` 作为 final C2 summary 直接消费的 read-only supporting parser。该 parser MUST 能读取既有 overnight root 和 baseline roots，并继续提供 final C2 所需的 summary、drop-count、pattern 和 router diagnostics 数据；它 MUST 不重新启动训练或要求历史 launcher 存在。

#### Scenario: Final C2 复用历史 summary parser
- **WHEN** `scripts/summarize_final_c2_ablation_v1.py` 读取 overnight branch-router 结果
- **THEN** retained parser MUST 提供 final C2 当前使用的解析函数和缺失值语义
- **AND** parser MUST 只读取显式输入并将生成内容写入 ignored output root
- **AND** current docs MUST 将其标记为 supporting/historical parser，而不是推荐训练入口

### Requirement: Focused regression coverage
项目 MUST 保留 focused regression 覆盖 soft static weighting、oracle router target、masked softmax、focus pattern alias 和 retained summary parser。测试 MUST 使用合成或 fake artifact，不得读取真实 `dataset/`、依赖已删除 launcher 或写入受保护 runtime 产物。

#### Scenario: Retained parser focused pytest
- **WHEN** 开发者运行 `conda run -n kd_mm_beam pytest -q tests/test_overnight_branch_router_v2.py tests/test_final_c2_ablation_v1.py`
- **THEN** 测试 MUST 覆盖保留的 router helper、summary parser 和 final C2 消费路径
- **AND** 测试产物 MUST 写入 pytest 临时目录或 ignored output root
