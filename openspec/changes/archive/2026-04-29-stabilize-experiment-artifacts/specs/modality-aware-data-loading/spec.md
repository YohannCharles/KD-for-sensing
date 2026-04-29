## ADDED Requirements

### Requirement: Scenario 9 序列窗口完整生成
Scenario 9 序列 CSV 生成流程 MUST 包含每个 `seq_index` 内所有合法的滑动窗口。对长度为 `N` 的单个 `seq_index`，输入长度为 `in_len`、预测长度为 `out_len` 时，合法窗口数 MUST 为 `max(N - in_len - out_len + 1, 0)`。

#### Scenario: 包含最后一个合法窗口
- **WHEN** 某个 `seq_index` 包含 `N` 行，且 `N == in_len + out_len`
- **THEN** 序列生成 MUST 为该 `seq_index` 产生 1 个窗口
- **AND** 输出窗口 MUST 使用前 `in_len` 行作为历史输入，后 `out_len` 行作为未来目标

#### Scenario: 多个 seq_index 分别计算窗口
- **WHEN** 原始 CSV 包含多个 `seq_index`
- **THEN** 系统 MUST 在每个 `seq_index` 内独立生成窗口
- **AND** 系统 MUST 不跨 `seq_index` 拼接历史输入或未来目标

### Requirement: 小比例 portion 采样代表性
Dataset 样本构建流程 MUST 明确 `portion` 小比例采样语义。默认 `portion < 1.0` 时，系统 MUST 使用确定性、可复现且覆盖 CSV 全局分布的采样策略，不得默认只取 CSV 头部连续样本。采样策略、seed 和最终样本数 MUST 可记录到运行 metadata。

#### Scenario: portion 不取连续头部样本
- **WHEN** 用户设置 `portion: 0.05` 且 CSV 样本数大于 20
- **THEN** 默认采样结果 MUST 不等价于 `head(int(len * portion))`
- **AND** 采样结果 MUST 使用稳定 seed 或确定性索引，保证重复运行样本集合一致

#### Scenario: portion 保留 seq_index 覆盖
- **WHEN** 序列 CSV 包含 `seq_index` 列且 `portion < 1.0`
- **THEN** 采样策略 MUST 尽可能覆盖完整 `seq_index` 范围
- **AND** 运行 metadata MUST 记录采样后的样本数和涉及的 `seq_index` 范围

#### Scenario: portion 全量采样
- **WHEN** 用户设置 `portion: 1.0`
- **THEN** Dataset MUST 使用 CSV 中全部样本
- **AND** 样本顺序 MUST 与 CSV 原始顺序保持兼容
