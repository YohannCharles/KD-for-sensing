## ADDED Requirements

### Requirement: CSI hardening sweep validity gate
CSI hardening sweep 的分析流程 MUST 在候选排序和设计结论前执行有效性 gate。有效性 gate MUST 至少检查 A0 clone parity、pilot 噪声量级、C1/C2 单变量健康状态和必需 diagnostics 是否存在。未通过 gate 的 sweep MUST 标记为 invalid 或 pending-debug，不得被解释为 hardening 设计失败。

#### Scenario: A0 parity 未通过
- **WHEN** `A0_clone_generated` 未通过与 `A0_original` 的关键配置 diff 或短跑曲线 parity
- **THEN** 分析输出 MUST 将 full sweep 状态标记为 pending-debug 或 invalid
- **AND** 系统 MUST 不输出 slow-high-ceiling 候选结论

#### Scenario: pilot 噪声量级失真
- **WHEN** 一个标记为 mild pilot estimation 的 run 的 `noise_power_signal_ratio` 明显高于其配置 SNR 对应范围
- **THEN** 分析输出 MUST 将该 run 标记为 `invalid_due_to_pilot_noise_scale` 或等价原因
- **AND** 该 run MUST 不参与 slow-high-ceiling 候选排序

#### Scenario: C1 或 C2 单变量异常
- **WHEN** A0 clone 正常学习但 C1 view gate warmup only 或 C2 no internal GRU only 掉到接近随机水平
- **THEN** 分析输出 MUST 将问题归因到对应 encoder 单变量路径
- **AND** 系统 MUST 阻止把 B/D hardening 组合结果解释为 hardening 强度问题

#### Scenario: 旧 sweep 缺少必需 diagnostics
- **WHEN** 分析脚本处理旧 full sweep 目录且该目录缺少 A0 parity、pilot noise ratio 或 debug decision artifacts
- **THEN** 分析输出 MUST 明确标记该 sweep 需要重跑或人工确认
- **AND** 默认候选排序 MUST 排除这些 invalid/pending run

### Requirement: CSI hardening sweep rerun workflow
项目 MUST 提供修复后的 CSI-only A/B/C/D sweep 运行入口或命令说明。该 workflow MUST 先运行短 debug gate，再运行完整 CSI-only sweep，并在输出中记录所使用的配置版本、pilot estimation 模式、noise ratio diagnostics 和旧结果隔离状态。

#### Scenario: 生成修复后的 A1 配置
- **WHEN** 开发者生成或加载修复后的 A1 mild pilot estimation 配置
- **THEN** 配置 MUST 使用 estimation-SNR 模式
- **AND** resolved config MUST 记录固定 SNR 或训练 SNR 采样区间

#### Scenario: 生成修复后的 B/C/D 配置
- **WHEN** 开发者生成或加载修复后的 B、C 或 D 组配置
- **THEN** 每个配置 MUST 显式关闭 pilot estimation noise
- **AND** 每个配置 MUST 保留自身声明的 hardening 或 encoder 变量

#### Scenario: 重跑前执行 debug gate
- **WHEN** 开发者请求完整 CSI hardening sweep
- **THEN** workflow MUST 先确认 A0 original、A0 clone、pilot disabled、C1 only 和 C2 only 的 debug gate 通过
- **AND** 如果 gate 未通过，workflow MUST 停止或将完整 sweep 输出标记为 pending-debug

#### Scenario: 输出新旧结果隔离状态
- **WHEN** 修复后的 sweep analysis 完成
- **THEN** summary artifact MUST 记录当前 sweep 是否基于修复后的 pilot scaling 配置
- **AND** 如果同一项目中存在旧 invalid sweep，summary artifact MUST 不把旧 sweep 的候选结果混入当前 ranking
