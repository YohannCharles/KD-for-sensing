# target-shot-domain-splitting Specification

## Purpose
定义 source-target domain、target-shot labeled/unlabeled/test 拆分和 split artifact 复用契约，用于支持跨场景或跨天气快速适应实验，并防止 target_test 泄漏、sample overlap 与不可复现的 target labeled selection。
## Requirements
### Requirement: 可配置 source-target domain 定义
系统 MUST 支持通过配置定义多场景/多天气 beam prediction 的 source domain 和 target domain。domain 类型 MUST 至少支持 `scenario`、`weather`、`scenario_weather` 和 `town_scenario_weather`；系统 MUST 使用 sample metadata 中的明确字段构造 domain key，不得依赖不可解析的 sample_id 字符串猜测 domain。

#### Scenario: 按 scenario_weather 选择 domain
- **WHEN** 配置设置 `split.domain_type: scenario_weather`、`split.source_domains` 和 `split.target_domains`
- **THEN** split builder MUST 从样本 metadata 的 scenario 与 weather/condition 字段构造 domain key
- **AND** 只把匹配 source domain 的样本放入 source split
- **AND** 只把匹配 target domain 的样本放入 target split

#### Scenario: domain 字段缺失
- **WHEN** 配置要求 `town_scenario_weather` 但样本 metadata 缺少 town、scenario 或 weather/condition 中任一字段
- **THEN** split builder MUST 拒绝生成 split
- **AND** 错误信息 MUST 包含缺失字段名、dataset type 和可执行修复提示

### Requirement: 5% target-shot target 拆分
系统 MUST 将 target domain 确定性拆分为 `target_labeled`、可选 `target_unlabeled` 和 `target_test`。`target_labeled` MUST 只从 target adaptation pool 中采样，默认目标比例为 `target_label_fraction=0.05`；`target_test` MUST 与 source、target_labeled 和 target_unlabeled 无 sample id 交集。

#### Scenario: target_labeled 比例可复现
- **WHEN** 用户使用相同输入 manifest/CSV、domain 配置、`target_label_fraction: 0.05`、selection strategy 和 seed 生成 split 两次
- **THEN** 两次生成的 `target_labeled` sample ids MUST 完全一致
- **AND** `target_labeled` 数量 MUST 等于可用 target adaptation pool 的 5% 四舍五入策略结果或 metadata 中声明的最小样本修正规则

#### Scenario: target split 无交集
- **WHEN** split builder 完成 source、target_labeled、target_unlabeled 和 target_test 拆分
- **THEN** 任意两个 split 的 sample id 集合 MUST 无交集
- **AND** 对序列窗口数据，split metadata MUST 继续记录 frame/window overlap 与 guard band leakage diagnostics

### Requirement: target labeled subset 分层采样
系统 MUST 支持 `random`、`stratified_by_beam`、`stratified_by_geo_sector` 和 `stratified_by_weather` target labeled selection strategy。无法满足某个分层桶的最小样本数时，系统 MUST 记录降级原因并保持整体采样可复现。

#### Scenario: 按 beam 分层采样
- **WHEN** 配置设置 `split.target_label_selection: stratified_by_beam`
- **THEN** split builder MUST 基于 target adaptation pool 中的 beam label 分布选择 target_labeled
- **AND** sampling manifest MUST 记录每个 beam 的候选数、选中数和 seed

#### Scenario: 按 geo_sector 分层但 geometry 不可用
- **WHEN** 配置设置 `split.target_label_selection: stratified_by_geo_sector` 且 target adaptation pool 缺少 geo_sector
- **THEN** split builder MUST 拒绝该策略或按配置声明的 fallback 策略降级
- **AND** split metadata MUST 记录 fallback reason

### Requirement: split artifact 持久化与复用
系统 MUST 将 split indices/sample ids、配置摘要、输入 fingerprint、domain metadata、label histogram、weather/scenario histogram、target labeled sampling manifest 和 leakage diagnostics 写入 JSON 或 NPZ artifact。后续训练、适配、评估和诊断 MUST 能通过 artifact 复用同一 split。

#### Scenario: split artifact 匹配当前输入
- **WHEN** 用户传入已有 split artifact 且输入 sample ids、配置摘要和 fingerprint 匹配
- **THEN** 系统 MUST 复用 artifact 中的 split
- **AND** 不得重新随机采样 target_labeled

#### Scenario: split artifact 不匹配
- **WHEN** 已有 split artifact 的 sample id fingerprint、target domain、seed 或 target_label_fraction 与当前配置不匹配
- **THEN** 系统 MUST 拒绝复用该 artifact
- **AND** 错误信息 MUST 指出不匹配字段，并提示 regenerate 或 overwrite

### Requirement: target_unlabeled 监督字段防泄漏
系统 MUST 将 target_unlabeled 标记为无监督 target subset。训练 loss、adaptation、threshold selection、temperature fitting、prototype update 和 early stopping MUST NOT 访问 target_unlabeled 的 beam、residual、beam_power、CSI/channel、path 或 radio supervision 字段。

#### Scenario: target_unlabeled 访问 beam supervision 失败
- **WHEN** target_unlabeled batch 被用于 adaptation 且 loss 代码尝试读取 beam 或 residual label 作为监督
- **THEN** runtime guard MUST raise error
- **AND** 错误信息 MUST 包含 split、subset、field name、label fraction 和修复提示

#### Scenario: target_labeled 允许 beam supervision
- **WHEN** batch 来自 `target_labeled` 且 `target_label_fraction > 0`
- **THEN** supervised beam 或 residual loss MAY 读取该 batch 的 beam/residual label
- **AND** metadata MUST 记录该监督只来自 target_labeled subset

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

### Requirement: 序列 CSV 使用 balanced_seq split 协议
DeepSense6G 序列 CSV 生成流程 MUST 使用单一的 `balanced_seq` train/test split 协议。split MUST 以完整 `seq_index` 为最小单位，MUST 不把同一 `seq_index` 的滑动窗口同时分配到 train 和 test，且 MUST 保持每个窗口仍只在单个 `seq_index` 内生成。

#### Scenario: split 可复现
- **WHEN** 用户使用相同原始 CSV、`training_set_pct`、`split_seed` 和 seq 数量控制配置
- **THEN** 序列生成流程 MUST 在重复运行时产生相同的 train/test `seq_index` 集合
- **AND** 系统 MUST 允许不同 `split_seed` 产生不同集合

#### Scenario: 标签分布感知选择 test seq
- **WHEN** 用户运行序列 CSV 预处理
- **THEN** 序列生成流程 MUST 基于生成后的窗口标签统计选择完整 test seq
- **AND** test 窗口数量 MUST 尽量接近配置的目标测试比例或显式 test seq 数
- **AND** test label 分布 MUST 尽量接近全量窗口 label 分布

#### Scenario: 小 seq 数场景的最少验证 seq
- **WHEN** 用户配置 `min_test_sequences` 且可用 `seq_index` 数量足以满足该约束
- **THEN** 序列生成流程 MUST 至少选择该数量的 test seq
- **AND** 如果该约束与显式 `test_sequence_count` 冲突，系统 MUST 抛出清晰错误或按文档定义的优先级处理

### Requirement: 序列 split metadata 可追踪
序列 CSV 预处理 MUST 为生成的 train/test split 记录可机器读取的 metadata。metadata MUST 足以解释当前 split 的策略、seed、seq 分配、窗口数和主要 label 分布。

#### Scenario: 写出 split metadata
- **WHEN** 用户运行序列 CSV 预处理
- **THEN** 系统 MUST 写出 split metadata sidecar
- **AND** metadata MUST 包含 `split_protocol: balanced_seq`、`split_seed`、`training_set_pct`、train/test `seq_index` 列表、train/test 窗口数和输出 CSV 路径

#### Scenario: 记录标签分布摘要
- **WHEN** 序列 CSV 中包含 beam 标签路径
- **THEN** split metadata MUST 记录 train/test 的 label 分布摘要
- **AND** 摘要 MUST 至少覆盖当前时隙标签或所有训练目标时隙中的一种明确口径

#### Scenario: 新统一 split 必须有 metadata
- **WHEN** 用户使用新预处理配置生成默认统一 split CSV
- **THEN** train/test CSV 旁 MUST 存在 split metadata sidecar
- **AND** metadata 中的 train/test 窗口数 MUST 与输出 CSV 行数一致

### Requirement: target-shot split 字段隔离
数据加载流程 MUST 根据 split artifact 中的 subset 标记构建 source、target_labeled、target_unlabeled 和 target_test dataloader。target_unlabeled loader MUST 能提供 sensing input 和非监督 metadata，但训练 payload MUST 不暴露可作为监督的 target labels。

#### Scenario: target_unlabeled loader 隔离监督字段
- **WHEN** 构建 target_unlabeled adaptation loader
- **THEN** batch metadata MUST 标记 subset 为 `target_unlabeled`
- **AND** training payload MUST 不允许 loss 访问 beam/residual supervision 字段

#### Scenario: target_test loader 只用于评估
- **WHEN** 构建 target_test loader
- **THEN** batch MAY 包含 evaluation metrics 所需 label
- **AND** run metadata MUST 标记 target_test labels 只可在 evaluation scope 使用

### Requirement: Target-shot split 作为 MMW supporting owner 保留
`target-shot-domain-splitting` MUST 分类为 supporting capability，因为 `kd_sensing.data.mmw.protocol` 直接复用其 deterministic split、leakage guard、artifact write/load 和 metadata contract。项目 MUST 保留这些 helper，但 MUST 不恢复已退役 standalone target-shot CLI、console script 或独立 quickstart workflow。

#### Scenario: MMW protocol 复用 target-shot helper
- **WHEN** MMW cross-scene protocol 构建 target labeled/unlabeled/test split artifact
- **THEN** protocol MUST 继续复用 `target_shot_splits.py` 的 canonical helper
- **AND** split determinism、sample overlap、target-test leakage 和 artifact fingerprint 行为 MUST 保持

#### Scenario: Supporting helper 不扩大为 public workflow
- **WHEN** pyproject、CLI help、README 和 current workflow 被枚举
- **THEN** 项目 MUST 不声明 standalone target-shot 命令或推荐入口
- **AND** lifecycle inventory MUST 将 capability 标记为 `supporting` 而不是 `retired-tombstone`
