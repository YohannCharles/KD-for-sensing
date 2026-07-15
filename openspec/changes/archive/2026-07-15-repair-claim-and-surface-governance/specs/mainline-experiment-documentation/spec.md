## ADDED Requirements

### Requirement: Claim registry schema 与外键必须可验证
claim registry MUST 使用固定结构化字段记录 claim id、subject/method、dataset/split、metric/value、status、provenance、caveat、seed_count、baseline、mean/std 或 CI、comparability、stress status、candidate flag 和 upgrade gate。主线 catalog 中每个非空 claim id MUST 唯一引用 registry 中存在的 claim。

#### Scenario: Pending claim 缺少数值
- **WHEN** claim status 为 pending、unverified、not_comparable 或 invalidated
- **THEN** metric value 和统计摘要 MAY 为空
- **AND** provenance、caveat、comparability 和 upgrade gate MUST 解释缺失或 blocker

#### Scenario: Reviewed claim 字段完整
- **WHEN** claim status 请求进入 reviewed paper main table
- **THEN** method、dataset/split、metric/value、seed_count、baseline、统计摘要、comparability、stress、provenance 和 caveat MUST 全部非空且合法

#### Scenario: Catalog claim 外键断裂
- **WHEN** mainline model catalog 引用 registry 不存在或重复的 claim id
- **THEN** architecture test MUST 失败并报告 model id 与 claim id

### Requirement: 仓库主线与数据 campaign 术语必须分离
current README、navigation、catalog、matrix、protocol 和 claim docs MUST 将 final C2 / U-MaskBeamJEPA 描述为仓库默认模型/研究主线，并将 MMW/CSI 描述为 current supporting dataset workflow。MMW MAY 被描述为当前数据实验 campaign，但 MUST NOT 被描述为已替换默认主线，除非 active transition change 同步全部 current contracts。

#### Scenario: Current 文档描述 MMW
- **WHEN** current 文档列出 MMW all-weather、Town GPS 或 BPA/CMA workflow
- **THEN** 文档 MUST 标明 supporting dataset/campaign、入口、输出边界和 claim status
- **AND** final C2 默认主线表述 MUST 保持一致

## MODIFIED Requirements

### Requirement: baseline 报告状态分类
项目 MUST 在 current claim、protocol 和历史账本中使用统一状态分类，将 official blocked、local substitute、strict-validation、upper-bound、mock/smoke 和 historical ablation 分开。已失去运行 owner 的 root 复现报告 MUST 在迁移唯一有效结论后删除，不得继续充当 current summary 或命令入口。

#### Scenario: 旧 BeamBench root 报告退出 current surface
- **WHEN** BeamBench package、CLI、配置和脚本已经退役
- **THEN** `README_REPRODUCE.md`、`BASELINE_REPORT.md`、`DATASET_STRUCTURE.md`、`PATCH_NOTES.md`、`TODO_FOR_ATTENTION_MODULE.md` 和 `results/reproduce_baseline.md` MUST 被删除
- **AND** blocked/not-comparable 结论 MUST 迁入 `docs/mainline_experiment_history.md` 与 claim registry，且不得保留旧推荐命令

#### Scenario: Current claim 状态有唯一权威来源
- **WHEN** 开发者查找当前可引用结果或复现 gate
- **THEN** README MUST 指向 `docs/result_claims_registry.md`、`docs/experiment_protocols.md` 和 `docs/mainline_experiment_history.md`
- **AND** 历史输出、checkpoint、metrics 和日志 MUST 继续留在 ignored 本地产物边界

### Requirement: Paper export and literature documentation
主线实验文档 MUST 索引 paper artifact export、当前只读 dataset inspection 和 literature matrix，并说明它们的 claim 状态与本地产物边界。

#### Scenario: 文档索引 paper export
- **WHEN** README 或 `docs/experiment_matrix.md` 给出 paper export 说明
- **THEN** 文档 MUST 说明 export 只消费满足 reviewed gate 的 claim rows
- **AND** pending、mock、historical、unknown 和 candidate-only rows 默认不得进入 main table

#### Scenario: 文档索引 dataset inspection
- **WHEN** current 文档给出数据检查入口
- **THEN** 文档 MUST 只引用 on-disk current script lifecycle 或 public package CLI 中存在的入口
- **AND** inspection MUST 只读、不移动数据，也不代表 official reproduction 已完成

#### Scenario: inventory 记录 literature matrix
- **WHEN** 存在 `docs/literature_matrix.md` 或 `paper/references.bib`
- **THEN** inventory MUST 记录其文档生命周期和职责
- **AND** 文档 MUST 不把本地 PDF 或外部论文下载物纳入源码产物要求
