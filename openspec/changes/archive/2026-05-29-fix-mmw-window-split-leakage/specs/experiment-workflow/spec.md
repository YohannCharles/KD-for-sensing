## MODIFIED Requirements

### Requirement: 实验输出记录 split 协议
训练和评估流程 MUST 在运行产物中记录足够的 split 协议信息，用于判断不同实验是否使用同一数据协议并可横向比较。记录 MUST 包含实际 CSV 路径、样本数和 split metadata 路径或核心字段。对于 MMW Town10 或其它滑窗 sequence 数据，记录还 MUST 包含 `split_strategy`、`split_protocol_version`、`strict_validation_eligible`、`eligibility_reasons` 和可用的 leakage diagnostics 摘要，避免把 unknown 或高重叠 split 误当成 strict validation 结果。

#### Scenario: 训练输出包含 split metadata 引用
- **WHEN** 训练入口构建 train/test dataset
- **THEN** `final_config.yaml`、`train_log.json` 或等价运行产物 MUST 记录 split metadata 路径或核心字段
- **AND** 记录 MUST 包含 split 策略、seed、train/test `seq_index` 数量和 train/test 样本数
- **AND** 当 split metadata 包含 strict eligibility 或 leakage diagnostics 时，运行产物 MUST 记录这些字段

#### Scenario: 评估输出包含 split 协议
- **WHEN** 评估入口构建 test dataset
- **THEN** 评估报告 MUST 记录实际使用的 test CSV 和可用的 split 协议信息
- **AND** 当当前 CSV 缺少 split metadata 时，系统 MUST 给出清晰错误或显式警告，避免把未知 split 协议误当成新协议结果
- **AND** 当 split metadata 标记 `strict_validation_eligible=false` 时，评估报告 MUST 保留指标但标记其不适合作为 strict 主结论

#### Scenario: 跨模态 split 可比较
- **WHEN** 用户使用同一组 train/test CSV 运行 image、radar、GPS、LiDAR、mmWave 或 fusion 实验
- **THEN** 各运行产物中的 split 协议信息 MUST 能显示它们使用相同 CSV 和相同 split metadata
- **AND** 如果 CSV 路径、split metadata、split strategy 或 strict eligibility 不同，用户 MUST 能从运行产物中看出这些结果不应直接作为同一 split 协议比较

## ADDED Requirements

### Requirement: 主结论过滤 split eligibility
实验 summary、quick conclusion 和横向比较工具 MUST 消费 split eligibility metadata。任何使用 unknown 或 leakage diagnostics 失败的 split 的 run MUST 不被用于 strict validation 主结论，除非用户显式请求 debug/sanity 汇总。

#### Scenario: strict split run 可进入主结论
- **WHEN** run metadata 记录 `strict_validation_eligible=true`
- **THEN** summary MAY 将该 run 纳入 strict validation 横向比较
- **AND** summary MUST 保留 split strategy、split metadata 路径和样本数，便于复核可比性

#### Scenario: strict-ineligible split run 被排除
- **WHEN** run metadata 记录 `strict_validation_eligible=false`
- **THEN** summary MUST 将该 run 排除出 strict 主结论
- **AND** summary MUST 记录 exclusion reason 和 split metadata 路径
- **AND** 用户仍 MAY 在 debug/sanity 视图中查看该 run 的原始指标

#### Scenario: split metadata 缺失时保守处理
- **WHEN** summary 读取到没有 split metadata 的 MMW Town10 run
- **THEN** summary MUST 标记该 run 的 split eligibility 为 unknown
- **AND** strict 主结论 MUST 默认排除该 run
- **AND** 输出 MUST 给出生成或引用 strict split metadata 的修复提示
