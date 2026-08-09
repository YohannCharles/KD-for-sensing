## Context

冻结的 PCPF-T 对每个 missing pattern 输出 `p_sense(b), b in [0,63]`。现有静态 K=7 策略一次性选择候选 beam，后续真实 RF measurement 只用于七选一，不能反过来修正 sensing posterior。MMW 的 serving beam ground truth 同时提供 64 个 ULA-DFT power；这些 power 在 train role 可用于估计 codebook phase-cycle 上“真实最优 beam 周围的相对增益形状”，但 validation/test 的完整向量必须继续由 radio simulator 私有持有。

本 change 将这两部分组合为 TBCP-7：sensing posterior 给出样本级先验，train-only topology artifact 给出与场景无关的 observation model，有限 RF measurement 给出闭环证据。该方法不修改 sensing 网络，也不把 CSI/channel 或完整 validation gain 暴露给 candidate policy。

## Goals / Non-Goals

**Goals:**

- 用唯一 train split 拟合可审计、可复用的 64-beam 相对 log-gain likelihood 和 normalized-gain kernel。
- 固定 K=7，在每次 feedback 后对 64 个 beam hypothesis 做精确联合 likelihood 更新。
- 用 posterior expected terminal normalized gain 选择下一次 probe，而不是用手工不确定性阈值切换间隔。
- 保持 simulator 的 requested-measurement API、最终七选一规则和 test 封存。
- 在同一三 seed、同一 validation identity、全部 15 个四-sensing mask 上比较强基线。

**Non-Goals:**

- 不新增均值/方差网络头、Gaussian latent、Router、跨模态 attention、loss 或训练 stage。
- 不把 train beam-power artifact 放入模型 checkpoint，也不让 validation/test 更新它。
- 不声称当前数据提供真实在线 CSI 噪声或时延；默认 replay 使用数据原生的确定性 beam power。
- 不在 validation 上挑 likelihood temperature、jitter、K、初始 probe 数或 policy switching threshold。

## Decisions

### 1. Train-only artifact 对齐 ULA-DFT phase cycle

对每个 train 样本读取 64 元非负 power `P` 和官方 label `b*`，先验证 `argmax(P)==b*`。定义

```text
r_delta = P[(b* + delta) mod 64] / P[b*]
h_delta = 10 log10(r_delta), delta=0..63
```

artifact 保存：

- `mean_db[64] = E[h]`；
- `covariance_db2[64,64] = Cov[h]`；
- `gain_kernel[64] = E[r]`；
- schema/policy/topology identity、fit split、sample count/hash、protocol manifest/hash、data/window hash、source CSV hashes、train power 内容 hash 与 artifact SHA256。

训练 power 必须严格为正；label/power 漂移、样本 count/hash、protocol/topology 或 artifact digest 不一致时失败关闭。artifact 是 probing calibration，不是 PCPF-T 输入或训练监督。

### 2. 相对 measurement 使用联合高斯 likelihood

已 probe beam 为 `S=(s0,...,st)`，返回 power 为 `Y`。以第一束为 reference，构造

```text
z_j = 10 log10(Y_j / Y_0), j=1..t.
```

在 hypothesis `b` 下，令 `o_j=(s_j-b) mod 64`。联合 likelihood 的均值与协方差为

```text
m_j = mean_db[o_j] - mean_db[o_0]
V_jk = C[o_j,o_k] - C[o_j,o_0] - C[o_0,o_k] + C[o_0,o_0].
```

若显式配置独立 per-beam measurement error `sigma_db`，则 `V` 的对角增加 `2*sigma_db^2`、非对角增加 `sigma_db^2`；默认 `sigma_db=0`。实现只增加固定数值 Cholesky jitter，并写入 policy version，不使用经验 likelihood temperature。每轮必须从原始 sensing prior 加当前完整 joint log-likelihood 重新归一化，不能累计重复使用旧 likelihood。

### 3. 固定 K=7 的闭环 expected-gain decision

TBCP-7 的状态只包含原始 sensing prior、当前 posterior、已请求 indices/measurements 与 train-only artifact：

1. `s0=argmax(p_sense)`；
2. 只有一个 measurement 时尚无相对观测，按 prior expected terminal gain 选择 `s1`；
3. 得到至少两个 measurement 后更新 posterior；
4. 对每个未 probe candidate `s` 计算
   `sum_b q(b) * max(max_{u in S} gain_kernel[(u-b) mod 64], gain_kernel[(s-b) mod 64])`；
5. 取 utility 最大的 candidate，完全并列时取较小 beam index，重复直到七个唯一 beam。

最终 beam 只能是七个实测 power 的 argmax。ledger 保存有序 probe indices、requested measurements、posterior MAP/entropy trace 和 final beam，便于审计闭环行为。

### 4. 强基线与相同预算

- `Posterior-Top7`：静态 posterior 最高七束；
- `Local-7`：MAP 周围连续七束；
- `Adaptive-Local-7`：现有 posterior-mass spacing template；
- `Uniform-7`：与样本无关的全局均匀网格；
- `Posterior5+Hill2`：先 probe posterior Top-5，再围绕五束中实测最强者补两个尚未 probe 的最近邻；
- `Full-64` 与 `Oracle-Local7` 只作 claim-ineligible upper bound。

所有非 oracle 方法共享相同 prior、K、simulator、final selection 与 metric。validation 结果不能触发 per-sample method switching。

### 5. Evidence scope 与汇总

production evidence 仍必须绑定 validation-best checkpoint、experiment seed、31-mask cache、formal topology 和唯一 protocol。新诊断选择 `available[:4]` 任意非空且 `available[csi]==false` 的 15 个 mask；每个 mask 必须覆盖相同的完整 validation sample identity/order。报告输出逐 pattern、Full、drop-1、drop-2、Single macro/worst，以及三 seed mean/std 和 paired delta。

### 6. Provenance 与边界

likelihood artifact 在加载和运行时同时核验：`fit_split=train`、train count/hash、当前 train power 内容 hash、protocol fingerprint、manifest/data/window hash、topology id/descriptor/audit hash、array shape/finite/symmetry 与文件 digest。validation power index 必须从同一 manifest 的 validation domain/CSV hash 构建，不得信任可漂移的 config domain 路径。validation simulator 继续只暴露 `probe(sample_id, requested_indices)`；policy 不能接收 GT、channel、CSI、完整 power 或 metric denominator。所有结果写入 ignored output，固定 `claim_ineligible=true`、`outer_test_accessed=false`、`model_trained_or_updated=false`。

### 7. Synthetic measurement-error 与 feedback-overhead sensitivity

当前数据没有真实 RF measurement SNR 或硬件 timing，因此 robustness 只使用预注册的 synthetic log-power error，不把它改写成真实 SNR。对 requested raw power `P>0` 定义

```text
Y_obs = P * 10^(sigma_db * z / 10), sigma_db in {0, 3, 6}.
```

`z` 由固定 noise seed `20260809`、replica、stable sample id 和 beam index 通过 SHA256 + Box-Muller 确定性生成；key 不含 policy、mask、batch 或调用顺序。不同方法、missing mask、checkpoint seed 与 sigma 共享同一标准正态 draw。`sigma=0` 只运行 replica 0；`sigma in {3,6}` 运行 replica `{0,1,2}`。TBCP belief update 必须接收同一个 `sigma_db`，Posterior5+Hill2 必须根据 noisy Top-5 winner 重新生成邻居，不能后处理无噪声 trace。最终 Top-1 与 normalized gain 仍按 clean private power 评价，噪声下 `GT covered` 与最终正确必须分开记录。

robustness replay 对每个 mask 选择 validation stable-id SHA256 排名最低的 512 个样本，三个 checkpoint seed 使用相同 identity/order。该 bounded subset、noise grid 和所有 replica 在运行前固定，不允许按 validation 数字增删条件。

通信指标复用 oracle-beam reference SNR `{0,10,20}` dB：

```text
rate_ratio = log2(1 + SNR_ref * normalized_gain) / log2(1 + SNR_ref).
```

它是选 beam 后的通信 rate proxy，不是 probing measurement SNR。硬件 latency 未知时只报告 normalized feedback cost：所有非 oracle 方法共享七个 measurement slots，额外 controller updates 为 static `0`、Posterior5+Hill2 `1`、TBCP-7 `5`；对每次 update 的 payload fraction `rho in {0,0.005,0.01,0.02,0.05}` 报告 `rate_ratio * max(0, 1-F*rho)` 和相对强基线的 break-even `rho`。不得把该比例换算成毫秒或真实 beam-switch latency。

### 8. Full/diagonal covariance 消融

无噪声主设定增加一个且仅一个 likelihood covariance 消融。`full` 使用 train artifact 原始 `C[64,64]`；`diagonal` 先固定替换为 `diag(diag(C))`，再使用 Decision 2 的同一 relative-measurement 变换。因此 diagonal control 仍保留共同 reference beam 的方差对所有相对观测诱导的非对角相关项，只移除不同 topology offset 的 train-estimated cross-covariance。

两个方法 MUST 共用同一 sensing prior、`mean_db`、`gain_kernel`、artifact、K=7、requested-measurement API、无噪声 power、最终 measured-argmax、validation identity/order 和 checkpoint seed。不得重新拟合 diagonal artifact、调节 jitter/temperature 或改变其他策略决策。三 seed 汇总 MUST 报告 `TBCP-7 full - TBCP-7 diagonal` 的逐 pattern paired delta；该消融仍为 validation-only、claim-ineligible，不能据结果反向更换主方法。

### 9. 开环控制与预算曲线

`Topology Open-loop Gain-K` 使用与 TBCP-K 相同的冻结 sensing prior、train-only `gain_kernel`、第一个 MAP beam、expected-terminal-gain utility、tie-break、simulator 和 measured-argmax。它在任何 RF measurement 返回前，用未更新的原始 sensing prior 一次性贪心构造全部 K 个候选；候选集合固定后才请求这些 beam。它不得使用 `mean_db`、covariance、GT、完整 validation power 或任何 feedback，因此 TBCP-K 与该控制的差值只隔离闭环 joint posterior update 对后续 acquisition 的贡献。

预算敏感性固定为 `K={3,5,7,9}`，只比较 `TBCP-K`、`Topology Open-loop Gain-K` 和 `Posterior Top-K`。三者对每个 K 共用相同 posterior、sample identity/order、无噪声 requested-measurement API、final measured-argmax 和指标。TBCP-K 在前两束之后进行 `K-2` 次 feedback-dependent acquisition；开环与 Posterior Top-K 的 controller update 数为 0。K=7 是在本曲线前已经冻结的主配置，其他 K 仅用于预算趋势诊断，不得根据 validation 最优点替换主 K、修改 likelihood 或选择性删掉不利预算。

### 10. 批量反馈诊断

为量化反馈时延而不改变 TBCP-7 精度上限，增加三个固定的无噪声 batch schedule：`(2,2,3)`、`(2,5)` 和平衡的 `(3,4)`。每个 schedule 的总测量数严格为 7；第一个 batch 的 beam 在任何 measurement 返回前按原始 sensing prior 选择，后续 batch 在前一批完整 measurement 返回后更新一次 posterior。一个 batch 内的候选按当前 posterior 和同一 `gain_kernel` 逐个加入集合，但不得读取该 batch 内尚未返回的 measurement，因此不伪造串行反馈。

批量策略使用现有 expected-terminal-gain utility，不新增 validation 拟合的 fantasy、温度或阈值。`Batch-TBCP-2+2+3` 有 3 个 measurement rounds/2 个跨批 posterior updates；`Batch-TBCP-2+5` 与 `Batch-TBCP-3+4` 均有 2 个 rounds/1 个跨批 update。最终 beam 仍只能从 7 个 requested measurements 的最大 clean power 中选择。报告必须同时记录 total measurement slots、measurement rounds、feedback updates 和与 `TBCP-7` 的 paired Top-1/normalized-gain delta。

该诊断不宣称批量策略等价于串行最优，也不把 round 数换算成真实毫秒；没有多 RF 链或可分离正交多波束观测时，round reduction 仅表示控制器反馈屏障减少，不能声称物理测量已经并行。

### 11. 两个创新点的嵌套消融

使用已有 topology supervision `off/on` 三 seed validation-best checkpoint 构造固定 2×3 表。`off` 同时关闭注册的环形 unimodal soft-label supervision 与 fused/modality prototype alignment；`on` 使用预注册 `unimodal_soft_weight=0.5`、`lambda_proto=0.2`、`lambda_modality_proto=0.1`。其余模型结构、稀疏 CSI、训练预算、数据协议、missing-mask evidence 与 seed 保持现有 nested-ablation contract，不为本轮重新训练或选择 checkpoint。

三个推理列固定为：不做 RF probing 的 `Direct Prediction`、静态同预算 `Posterior Top-7`、以及 full-covariance `TBCP-7`。所有 K=7 方法复用相同 train-only topology likelihood、无噪声 requested-measurement simulator、最终 measured-argmax 和完整 15-mask validation identity。

报告同时给出：`on-off` 的 topology 主效应；每个 topology 条件内 `TBCP-7 - Posterior Top-7` 的 matched-K closed-loop 增益；完整系统 `on/TBCP-7 - off/Direct` 的端到端增益；以及交互项 `(on,TBCP-on,Posterior) - (off,TBCP-off,Posterior)`。不根据 validation 结果改 loss 权重、K、likelihood 或主策略，结果保持 claim-ineligible。

## Risks / Trade-offs

- **[Gaussian likelihood 是近似]** -> 明确报告为 train-calibrated relative-gain model，并提供 posterior/static/hill-climb baselines；不把 validation 调温度后的数字用于 claim。
- **[共享 reference 造成相关误差]** -> 使用完整 covariance difference 和联合 likelihood，不做逐 measurement 独立相乘或经验 `1/t` temper。
- **[协方差接近奇异]** -> 只使用预注册数值 jitter；Cholesky 仍失败则报错，不自动放宽。
- **[train full power 被误认为额外模型监督]** -> artifact owner 与 PCPF-T 分离，checkpoint/state dict 不含该状态，文档和报告明确只用于 RF observation model calibration。
- **[validation 完整 power 泄漏]** -> 仅 simulator 私有读取；候选 API 和测试锁定 requested measurement 边界。
- **[离线 replay 高估真实在线性能]** -> 默认只主张 deterministic dataset feasibility；真实噪声、切换延迟和硬件开销留给独立实验。
- **[synthetic dB error 不是实测 RF noise]** -> 固定完整 stress grid、共享 common random numbers，并把结果标记为 bounded claim-ineligible sensitivity；不报告虚构的 hardware SNR。
- **[feedback latency 缺少绝对时钟]** -> 只报告 normalized overhead 曲线和 break-even，不填毫秒、不假设 codebook index distance 等于物理切换时间。

## Migration Plan

1. 更新本 change 的 clean-data 与 probing 契约，保留旧 K=7 方法作为 baselines。
2. 实现 train-only artifact fit/load 和 synthetic round-trip/provenance tests。
3. 实现 joint posterior update、expected-gain selection 与 requested-measurement tests。
4. 扩展现有 `probe-diagnostic` 以复用 31-mask evidence 并输出全部 15 个四-sensing mask。
5. 在新 ignored 目录重放 topology-on seed 1/2/3 validation；不覆盖旧 Posterior-Top7 结果、不访问 test。
6. 在固定 hash subset 上重放三 seed 的 synthetic measurement-error grid，并报告 communication reference-SNR 与 normalized feedback-overhead sensitivity。
7. 在完整无噪声 validation 上并排重放 full/diagonal covariance，按全部三 seed 汇总 paired delta。
8. 在同一完整无噪声 validation 上并排重放开环 topology control 与预注册 K 曲线，按全部三 seed 和 15 mask 汇总；不据曲线更换 K=7 主设定。

## Open Questions

- 当前 MMW power 没有已绑定的真实 RF measurement noise/SNR；`sigma_db` 只能作为明确标注的敏感性分析，不能替代硬件实验。
- validation replay 仍是 claim-ineligible 方法筛选；论文正式数字需要预注册后再执行独立最终评估。
