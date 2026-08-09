## ADDED Requirements

### Requirement: sensing posterior 必须提供可复算的环形统计

系统 MUST 从归一化的 64 类 sensing posterior 计算并返回 `map_beam`、circular mean、resultant length、`circular_variance`、以 MAP 为中心的 `beam_variance`、`beam_spread`、normalized entropy 和稳定排序的 Top-L。计算 MUST 使用审计 topology 的 modulo-64 circular distance，在 FP32 中执行；输入含 NaN、负值、错误类别数或非单位行和时 MUST 失败。resultant 接近零时 circular mean MUST 确定性退回 MAP。

#### Scenario: 统计处理环形边界

- **WHEN** posterior 质量集中在 beam 63 和 beam 0
- **THEN** circular mean、beam variance 和 spread MUST 将 63/0 视为 phase-cycle 相邻，不得按线性距离分开 63 格

### Requirement: TBCP-7 必须使用 train-only topology likelihood 闭环更新 belief

`TBCP-7` MUST 是固定 K=7 的主 probing policy。它 MUST 以冻结 sensing posterior 为原始 prior，以第一束 measurement 为 reference，并使用 train-only artifact 的相对 log-gain joint Gaussian likelihood 更新 64-beam posterior。联合 covariance MUST 包含共享 reference 的相关项；每轮 MUST 从原始 prior 与当前全部 measurement 重新计算，MUST NOT 将相关 measurement 当成独立证据重复相乘或使用 validation-tuned likelihood temperature。

#### Scenario: 得到前两束 measurement

- **WHEN** policy 已请求两个不同 beam 且返回严格正的 finite power
- **THEN** 系统 MUST 对 64 个 candidate-best-beam hypothesis 计算 normalized finite posterior
- **AND** posterior 更新只能读取这两个 requested measurement、原始 sensing prior 和已验证 train artifact

#### Scenario: measurement 或 covariance 非法

- **WHEN** requested power 非正/非 finite、artifact shape/fingerprint 漂移或 joint covariance 在固定 jitter 后仍不能 Cholesky 分解
- **THEN** policy MUST 失败关闭，不得退回静态策略或读取未请求 power

### Requirement: 下一束必须最大化 posterior expected terminal gain

TBCP-7 的第一束 MUST 为 sensing MAP。第二束以及后续每束 MUST 在未 probe beam 中最大化当前 posterior 下、加入该 candidate 后的期望最终 normalized gain；utility MUST 只使用 train-only `gain_kernel` 与已选择 indices。utility 完全相同 MUST 选择较小 beam index。获得至少两束 measurement 后，第三至第七束 MUST 使用更新 posterior，因此候选序列 MAY 随 requested feedback 改变。

#### Scenario: feedback 改变后续扫描位置

- **WHEN** 两个样本具有相同 sensing prior、但前两束相对 measurement 支持不同 best-beam hypothesis
- **THEN** 其更新 posterior MUST 不同
- **AND** 后续 candidate MAY 不同，同时两条路径均不得读取 GT 或完整 power

#### Scenario: 完成固定预算 probing

- **WHEN** policy 已请求七个唯一的合法 beam
- **THEN** policy MUST 停止并从七个返回 measurement 中选择 power 最大者作为 final beam
- **AND** ledger MUST 保存有序 indices、measurement、posterior trace、final beam、coverage 与 normalized gain

### Requirement: 批量反馈诊断必须保持固定预算与信息边界

系统 MUST 支持预注册的 `Batch-TBCP-2+2+3`、`Batch-TBCP-2+5` 与 `Batch-TBCP-3+4` validation-only diagnostics。每个策略 MUST 请求恰好 7 个唯一 beam；同一 batch 内的后续 candidate MUST 不读取该 batch 的 measurement，只有整个 batch 返回后才允许一次联合 posterior update。批量策略 MUST 使用与 `TBCP-7` 相同的 sensing prior、train-only likelihood、covariance mode、simulator、final measured-argmax 和 validation identity。

#### Scenario: 批量策略减少反馈屏障

- **WHEN** 执行 `Batch-TBCP-2+2+3`
- **THEN** ledger MUST 记录 7 个 requested measurements、3 个 measurement rounds 和 2 个跨批 posterior updates
- **AND** candidate policy MUST 只接收当前 batch 已经返回的 requested measurements

#### Scenario: 批量结果与串行结果成对比较

- **WHEN** 对同一 checkpoint、sample identity/order、topology likelihood 和 clean private power 运行 batch 与 sequential 策略
- **THEN** report MUST 输出 paired Top-1、normalized-gain delta 及 per-pattern 结果
- **AND** 结果 MUST 标记 `claim_ineligible=true`、`outer_test_accessed=false`，不得把 measurement rounds 解释为真实硬件毫秒

#### Scenario: 平衡两批反馈

- **WHEN** 执行 `Batch-TBCP-3+4`
- **THEN** ledger MUST 记录 7 个 requested measurements、2 个 measurement rounds 和 1 个跨批 posterior update
- **AND** 该结果 MUST 与 `Batch-TBCP-2+2+3`、`Batch-TBCP-2+5` 和严格串行 `TBCP-7` 成对报告，不得据此重新选择主策略

### Requirement: 两个创新点必须通过三 seed 嵌套消融分离

系统 MUST 在 topology supervision `off/on` 的现有三 seed validation-best checkpoints 上比较 `Direct Prediction`、`Posterior Top-7` 与 full-covariance `TBCP-7`。两个 topology 条件 MUST 使用同一 `mmw_id_stratified_block_v1` protocol、完整 15 个 CSI-off sensing masks、相同 validation identity/order、同一个 train-only likelihood 和相同 K=7 simulator。系统 MUST 报告 topology 主效应、`TBCP-7 - Posterior Top-7` matched-budget 增益、端到端增益和 difference-in-differences 交互。

#### Scenario: 运行嵌套消融

- **WHEN** topology supervision `off/on` 的 seed 1/2/3 完成 validation replay
- **THEN** 每个单元格 MUST 汇总全部三 seed mean/std、逐 mask 与 Full/drop-1/drop-2/Single 分组
- **AND** topology `off` MUST 只使用已注册 nested-ablation checkpoint，不得在本轮重新训练、重新选择 seed 或修改其他训练预算

#### Scenario: 匹配预算隔离第二个创新点

- **WHEN** 计算第二个创新点的闭环贡献
- **THEN** 主比较 MUST 为相同 topology 条件内 `TBCP-7 - Posterior Top-7`
- **AND** `TBCP-7 - Direct Prediction` 只能作为端到端 pipeline 增益，不得冒充同预算 acquisition 消融

### Requirement: 静态和一轮反馈策略必须作为同预算基线

`Posterior-Top7`、`Local-7`、`Adaptive-Local-7`、`Uniform-7` 与 `Posterior5+Hill2` MUST 保留为 K=7 baseline/ablation，不得在样本级根据 validation 结果切换。`Posterior5+Hill2` MUST 先请求 sensing posterior Top-5，再只根据这五个 requested measurement 的最强 beam 补两个尚未请求的最近 phase-cycle neighbor。Oracle/Full-64 MUST 单独标记 claim-ineligible。

#### Scenario: baseline 构造候选

- **WHEN** 任一非 oracle baseline 运行
- **THEN** 它 MUST 不接收 GT、channel、CSI、完整 power 或未请求 measurement
- **AND** 它 MUST 与 TBCP-7 使用相同 K、sample identity、simulator、final selection 和 normalized-gain 实现

### Requirement: probing diagnostic 必须覆盖全部四-sensing 非空 mask

诊断 MUST 绑定 validation-best checkpoint、其 unbounded 31-mask evidence、唯一 MMW protocol 与正式 topology，并选择 CSI availability 严格为 false、image/radar/gps/lidar availability 任意非空组合的全部 15 个 mask。每个 mask MUST 包含相同且完整的 validation sample identity/order。输出 MUST 包含逐 pattern、Full、drop-1、drop-2、Single macro/worst 以及全部 seed 汇总。

#### Scenario: evidence mask 或 identity 漂移

- **WHEN** 不足/多于 15 个合法 sensing mask、CSI 被开放、任一 mask 样本不完整或 order 不一致
- **THEN** diagnostic MUST 在加载 radio power 前失败关闭

#### Scenario: 运行三 seed replay

- **WHEN** topology-on seed 1/2/3 使用同一 protocol、topology、artifact 和 K=7 运行 validation replay
- **THEN** 报告 MUST 汇总全部 seed mean/std 与 paired delta，不得选择最有利 seed
- **AND** 必须记录 `claim_ineligible=true`、`outer_test_accessed=false` 与 `model_trained_or_updated=false`

### Requirement: finite probing 必须隔离 radio ground truth

radio simulator MAY 私有缓存 evaluation 样本的完整 64-beam power，但 public `probe(sample_id, indices)` MUST 只返回显式请求的 measurement。candidate policy 不得接收 simulator 私有 cache、GT、metric denominator 或完整 vector；`normalized_gain` 只能在 candidate sequence 完成后由 evaluator 计算。

#### Scenario: validation label 与 radio ground truth 漂移

- **WHEN** Full-64 argmax 不等于 GT，或无噪声 probing 的 `final_beam==GT` 与 `GT in candidates` 不一致
- **THEN** diagnostic MUST 失败并报告 label/power/tie 漂移，不得静默汇总

### Requirement: robustness sensitivity 必须使用预注册 matched synthetic measurement error

系统 MUST 提供一个与无噪声主诊断分离的 bounded robustness sensitivity。它 MUST 使用固定 `sigma_db={0,3,6}`、固定 noise seed、固定 replica 集与 stable-id hash subset；requested power MUST 按独立 per-beam dB error 扰动，且同一 `(sample_id, beam, replica)` 的标准正态 draw MUST 跨 policy、mask、batch size、checkpoint seed 与 sigma 复用。噪声参数 MUST 同时进入 TBCP joint likelihood；动态 TBCP 与 Posterior5+Hill2 MUST 真实重放 feedback-dependent candidate sequence，不得从无噪声 trace 推导。

#### Scenario: matched-noise replay

- **WHEN** 两个方法在同一 replica 请求相同 sample 的相同 beam
- **THEN** simulator MUST 返回相同 noisy measurement，与请求顺序、batch size、method 和 missing pattern 无关
- **AND** candidate policy 仍只能看到 requested noisy values，clean power 与 metric denominator 必须保持私有

#### Scenario: noise 造成已覆盖但选错

- **WHEN** K=7 candidates 包含 GT，但 noisy measurement argmax 选择其他 beam
- **THEN** evaluator MUST 分别记录 coverage、Top-1 与 clean normalized gain，不得套用无噪声 `correct == covered` 断言

#### Scenario: sensitivity provenance 或网格漂移

- **WHEN** sigma、replica、noise key version、hash subset、checkpoint/protocol/topology/artifact binding 或 test seal 与预注册配置不一致
- **THEN** sensitivity runner/summary MUST 失败关闭，不得选择性省略不利条件或把 bounded 结果替代完整 validation 主表

### Requirement: robustness 报告必须区分 measurement error、communication SNR 与 latency proxy

系统 MUST 在 clean selected-beam gain 上报告 oracle-beam reference SNR `{0,10,20}` dB 的 spectral-efficiency ratio；该 SNR MUST 明确标记为 communication metric，不得称为 probing measurement SNR。系统 MAY 报告 normalized feedback-overhead 曲线和 TBCP 相对强基线的 break-even，但 MUST 记录各方法 measurement slots/controller updates，并明确没有实测毫秒 latency 或 hardware switching model。

#### Scenario: 生成 feedback-overhead 报告

- **WHEN** sensitivity 汇总 TBCP-7、Posterior5+Hill2 与静态 K=7 baseline
- **THEN** 所有方法 MUST 共享七个 measurement slots，controller updates MUST 分别按 `5/1/0` 计数
- **AND** 报告 MUST 使用固定 normalized overhead grid，不得根据 validation 结果挑选更有利的 payload duration 或 latency 参数

### Requirement: covariance 消融必须只移除 train-estimated cross-offset covariance

系统 MUST 在无噪声完整 validation 上提供 `TBCP-7 full covariance` 与 `TBCP-7 diagonal covariance` 的三 seed paired ablation。diagonal control MUST 对同一个 train-only artifact 使用 `diag(diag(C))` 后再构造 relative-measurement joint covariance；共同 reference measurement 诱导的相关项 MUST 保留。两个方法 MUST 共用 sensing prior、likelihood mean、gain kernel、K=7、simulator、最终 measured-argmax、sample identity/order 和其余全部数值设置。

#### Scenario: 运行 diagonal covariance control

- **WHEN** evaluator 启用预注册 covariance ablation
- **THEN** full/diagonal 两条闭环路径 MUST 各自仅根据其 requested measurement 更新和选择后续 beam
- **AND** ledger/report MUST 记录 covariance mode，并输出逐 mask、分组和三 seed `full-diagonal` paired delta

#### Scenario: covariance control 改变其他变量

- **WHEN** diagonal control 使用重新拟合 artifact、删除共同 reference 相关项、修改 mean/gain kernel/jitter/K/final selection 或读取未请求 power
- **THEN** evaluator MUST 失败关闭或测试 MUST 拒绝该实现
- **AND** 该结果不得进入 covariance ablation 汇总

### Requirement: 开环 topology control 必须只移除 feedback posterior update

系统 MUST 提供 `Topology Open-loop Gain-K` 控制。它 MUST 使用与 TBCP-K 相同的原始 sensing posterior、train-only `gain_kernel`、MAP first probe、expected-terminal-gain utility、lower-index tie-break、simulator、K 和 final measured-argmax，但 MUST 在读取任何 requested measurement 前固定全部候选。它 MUST NOT 使用 likelihood mean/covariance、GT、完整 validation power 或 feedback 修改候选。

#### Scenario: 开环与闭环使用相同起点

- **WHEN** TBCP-K 与开环控制接收相同 sensing posterior、artifact 和 K
- **THEN** 两者第一束 MUST 都是 sensing MAP，第二束 MUST 都由相同原始 prior 和 gain kernel 选择
- **AND** 只有 TBCP-K 的第三束及之后 MAY 因前两束 requested measurement 的 joint posterior update 改变

#### Scenario: 开环候选误读 measurement

- **WHEN** 相同 prior 在不同 radio power callback 下运行开环控制
- **THEN** 候选序列 MUST 完全相同
- **AND** 只有候选固定后的 measured-argmax final beam MAY 随返回 power 改变

### Requirement: 预算敏感性必须预注册且不得替换主 K=7

系统 MUST 在无噪声完整 validation 上运行固定 `K={3,5,7,9}` 的 `TBCP-K`、`Topology Open-loop Gain-K` 与 `Posterior Top-K`。每个 K 下三种方法 MUST 共用相同 posterior、sample identity/order、requested-measurement API、final selection 和 metric。K=7 MUST 保持主设定；其他预算 MUST 标记为 sensitivity，MUST NOT 根据 validation 最优点重新选择主 K 或修改 calibration/policy。

#### Scenario: 生成三 seed 预算曲线

- **WHEN** topology-on seed 1/2/3 完成全部 15 个四-sensing mask 的防守 replay
- **THEN** summary MUST 对每个预注册 K 报告三 seed mean/std、逐 mask 与 macro/worst 结果
- **AND** MUST 保留全部四个预算，包括不利点，并记录 `claim_ineligible=true`、`outer_test_accessed=false` 与 `model_trained_or_updated=false`

#### Scenario: K=7 回归一致性

- **WHEN** 启用预算曲线
- **THEN** 曲线中的 TBCP-7 与 Posterior Top-7 MUST 与同一 run 的主 TBCP-7 和既有 baseline 使用同一候选、measurement 与指标
- **AND** evaluator MUST 不为 K=7 建立第二套策略实现
