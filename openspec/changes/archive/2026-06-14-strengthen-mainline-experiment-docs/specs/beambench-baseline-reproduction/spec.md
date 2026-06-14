## ADDED Requirements

### Requirement: BeamBench current summary 与历史日志分层
BeamBench/Arnold22 baseline 报告 MUST 提供 current summary，并将历史流水账、旧命令、ablation、dry-run、mock 和 upper-bound 记录放入明确标记的 historical 或 appendix 区域。current summary MUST 是读者判断当前推荐口径的优先入口。

#### Scenario: current summary 位于报告开头
- **WHEN** 开发者打开 `BASELINE_REPORT.md`
- **THEN** 文档开头 MUST 给出当前推荐的 Table III 本地 substitute 口径、当前推荐命令、claim status、metric profile 和 caveat
- **AND** 文档 MUST 指向 `README_REPRODUCE.md` 或主线模型目录中的当前入口

#### Scenario: reproduce baseline 流水账不覆盖 current summary
- **WHEN** 开发者打开 `results/reproduce_baseline.md`
- **THEN** 文档开头 MUST 声明该文件是按时间记录的运行日志
- **AND** 文档 MUST 指向 `BASELINE_REPORT.md` current summary 作为当前可引用结论来源

### Requirement: Table III 本地 substitute 当前口径
Arnold22 BeamBench Table III `Camera=AE, GPS=Direct, Fusion=Yes` 本地 substitute 的当前推荐口径 MUST 使用当前 beam selection、`seq_len=1`、`num_pred=1`、GPS `paper_distance_angle`、scene paper calibration angle、linear/non-circular DBA 和 Top-1/3/5 指标。缺少官方权重、官方 exact test packaging 或官方完整训练搜索流程时，报告 MUST 将结果标记为 local substitute，而不是 official reproduction。

#### Scenario: 当前命令使用 current target
- **WHEN** `README_REPRODUCE.md`、`BASELINE_REPORT.md` 或主线模型目录给出当前 Table III 本地 substitute 命令
- **THEN** 命令 MUST 设置 `--target-beam-source current` 或继承配置中等价的 `target_beam_source: current`
- **AND** 文档 MUST 说明 `future` target 只允许作为历史 sequence-prediction ablation

#### Scenario: 官方复现条件未满足时不冒充官方结果
- **WHEN** 本地结果未使用官方 pretrained AE/fusion 权重、官方 exact test packaging、官方环境或官方完整训练搜索流程
- **THEN** 报告 MUST 将 claim status 标记为 local substitute、local strict-validation、upper-bound 或 historical ablation
- **AND** 报告 MUST NOT 声称该数值等同官方 Table III 复现

### Requirement: BeamBench 历史 ablation 标记
BeamBench 报告中保留的旧 `future` target、旧 AE 维度、旧 GPS 公式、旧校准角、`test_as_validation`、scene31-only、dry-run 或 mock 记录 MUST 有显式状态标记和不可引用 caveat。

#### Scenario: future target 历史记录被降级
- **WHEN** 历史日志包含 `--target-beam-source future` 或 `target_beam_source: future`
- **THEN** 同一段落或相邻说明 MUST 标记该记录为 historical ablation 或 sequence horizon ablation
- **AND** 文档 MUST 声明它不得作为 Table III strict setup 或当前推荐结果

#### Scenario: upper-bound 结果不可写成正式结果
- **WHEN** 报告包含 `test_as_validation` 或其它用 test split 选择 checkpoint 的结果
- **THEN** 报告 MUST 将其标记为 upper-bound
- **AND** current summary MUST 不把该结果作为 official 或 strict-validation 主结论

#### Scenario: mock 和 dry-run 结果不可用于论文比较
- **WHEN** 报告包含 mock、dry-run、synthetic 或极小样本 smoke metrics
- **THEN** 报告 MUST 标记 `mock_data: true`、dry-run 或 smoke
- **AND** 文档 MUST 声明这些数值只验证代码路径，不能用于论文或官方结果比较

### Requirement: BeamBench 结果进入统一 claim 账本
BeamBench/Arnold22 的当前结果、官方 blocked 状态、本地 substitute、strict-validation、upper-bound 和 historical ablation 摘要 MUST 被记录到统一结果/claim 账本或等价文档中。

#### Scenario: current BeamBench claim 可追溯
- **WHEN** 报告引用当前 BeamBench 本地 substitute 数值
- **THEN** 结果账本 MUST 记录 config、命令或 runner、train/eval scenes、target source、selection split、metric field、claim status、checkpoint provenance 和 caveat
- **AND** 账本 MUST 不提交 checkpoint、predictions、feature cache 或 metrics CSV

#### Scenario: 官方 blocked 状态可追溯
- **WHEN** 官方 BeamBench 权重、数据、源码或环境缺失导致 strict official reproduction blocked
- **THEN** 结果账本或 baseline current summary MUST 记录 blocked reason
- **AND** 文档 MUST 不填入伪造官方复现结果
