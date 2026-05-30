## MODIFIED Requirements

### Requirement: Sequence CSV and split artifacts are generated
系统 MUST 从有效 frame manifest 生成 beam 预测可用的序列 CSV。序列窗口 MUST 在同一 CAV agent 和连续 frame 片段内生成，不得跨 agent 或跨不连续 frame 拼接历史输入和未来标签。CSV MUST 至少包含 `seq_index`、历史 `camera*`、`lidar*`、`gps*`、`mmwave*`、`beam*` 列和 `future_beam*` 标签列，并 MUST 写出 train/test split metadata。默认 train/test split MUST 使用 group-safe 协议，按连续片段、agent、时间块或等价 group 分配窗口，并保留足够 guard band，避免 train/test 共享相邻滑窗上下文；公开准备流程和 split builder 不再支持随机窗口切分。

#### Scenario: 生成历史 8 帧和未来 3 帧窗口
- **WHEN** 配置 `seq_len=8` 且 `pred_len=3`
- **THEN** 每个输出样本 MUST 包含 `beam1..beam8`、`mmwave1..mmwave8` 和 `future_beam1..future_beam3`
- **AND** `future_beam1` MUST 对应当前历史窗口后的第一个未来帧
- **AND** 所有历史和未来 frame MUST 属于同一 CAV agent 和同一连续片段

#### Scenario: split 以 group-safe 单位生成
- **WHEN** 系统生成默认 train/test CSV
- **THEN** 同一个 `seq_index`、连续片段 group、time block 或 guard band 内的窗口 MUST 不得同时出现在 train 和 test
- **AND** train/test 之间 MUST 不共享完整历史+未来窗口中的 frame id
- **AND** split metadata MUST 记录 split strategy、protocol version、group key、guard band、split seed、比例、train/test group 列表、窗口数和 beam label 分布摘要
- **AND** split metadata MUST 标记 `strict_validation_eligible=true`

#### Scenario: unsupported split strategy 不生成随机窗口 split
- **WHEN** 用户通过配置或内部 API 请求不在支持列表中的 split strategy
- **THEN** 系统 MUST 按普通 unsupported strategy 处理
- **AND** 系统 MUST NOT 生成随机窗口 train/test CSV
- **AND** 系统 MUST NOT 为旧随机窗口策略写出专门 metadata 分支

## ADDED Requirements

### Requirement: MMW split leakage diagnostics
MMW Town10 split metadata MUST 包含可机器读取的泄漏诊断，用于判断当前 train/test CSV 是否可作为 strict validation 协议。诊断 MUST 至少覆盖 train/test frame overlap、test window 与 train window 的最大 frame overlap、相邻窗口跨 split 比例和未来标签序列复用比例。

#### Scenario: group-safe split 诊断通过
- **WHEN** 系统使用默认 group-safe 协议生成 split
- **THEN** leakage diagnostics MUST 记录 train/test frame overlap count 为 0
- **AND** test window 与任一 train window 的最大 frame overlap MUST 小于完整窗口长度
- **AND** summary MUST 包含 guard band frames、window length、train/test window counts 和 diagnostics 生成时间或版本

#### Scenario: 诊断发现高重叠
- **WHEN** leakage diagnostics 发现 test window 与 train window 共享完整或近完整历史+未来上下文
- **THEN** split metadata MUST 标记 `strict_validation_eligible=false`
- **AND** metadata MUST 包含超阈值统计和可执行修复提示
- **AND** 训练或评估产物消费该 metadata 时 MUST 能显示该 split 不适合作为 strict 主结论

### Requirement: MMW prepared split strategy is auditable
MMW Town10 数据准备和公开 split builder MUST 在输出 metadata 中记录实际生效的 split strategy。未显式设置时，默认 MUST 使用 group-safe 策略；当前公开准备协议只支持 `group_safe_time_block`，不提供旧随机窗口切分兼容路径。

#### Scenario: 默认使用 group-safe strategy
- **WHEN** 用户运行 MMW Town10 preparation 或 `build_sequence_splits_from_manifest` 且未指定 split strategy
- **THEN** 系统 MUST 使用 group-safe split strategy
- **AND** 输出 split metadata MUST 记录默认来源和策略参数

#### Scenario: 公开 split builder 只暴露 group-safe strategy
- **WHEN** 用户查看 MMW Town10 preparation 或 `build_sequence_splits_from_manifest` 的公开参数
- **THEN** 可用 split strategy MUST 限定为 `group_safe_time_block`
- **AND** 输出 split metadata MUST 记录该策略和策略参数
