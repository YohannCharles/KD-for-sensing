## Why

现有 PCPF-T 已能从缺失模态下的 sensing 输入给出 64-beam posterior，但静态 `Posterior-Top7` 只在 RF probing 前使用一次 posterior，不能利用前几个真实 measurement 修正后续扫描位置。快速 validation 筛选表明，训练集拟合的 ULA-DFT 相对增益结构与逐次 RF feedback 可以显著缩小 sensing 先验的定位误差，因此主方向应从静态候选集合升级为固定 K=7 的闭环 belief probing，而不是继续增加融合网络或高斯预测头。

## What Changes

- 将主方法改为 `TBCP-7`（Topology-aware Bayesian Closed-loop Probing）：以冻结 sensing posterior 为初始 belief，在每次已请求 measurement 后更新 64-beam belief，并按期望最终 normalized gain 选择下一个 beam。
- 新增独立的 train-only topology likelihood artifact。它只从绑定 protocol 的 train role 读取未来 64-beam power 与其 argmax label，拟合相对 log-gain 均值/协方差和线性 normalized-gain kernel；它不进入 PCPF-T forward、loss、optimizer、checkpoint 或 validation 拟合。
- 使用相对 dB measurement 的联合高斯 likelihood，显式处理共享 reference 带来的协方差；每轮从原始 sensing prior 与当前全部 measurement 重新计算 posterior，避免重复计数。
- 固定总预算 K=7。第一个 probe 为 sensing MAP，第二个 probe 在没有相对 measurement 前按期望终端增益选择，随后五个 probe 由更新后的 belief 闭环选择；最终 beam 只能从七个实测 beam 中取最大 power。
- 将 `Posterior-Top7`、`Local-7`、`Adaptive-Local-7`、`Uniform-7` 与 `Posterior5+Hill2` 保留为相同 K=7 的 baselines/ablations，不再将 Posterior-Top7 作为主方法。
- 正式 development diagnostic 覆盖五模态 evidence 中 CSI 严格缺失、四个 sensing 模态任意非空组合的全部 15 个 mask，并输出 Full/drop-1/drop-2/Single 分组和逐 pattern 结果。
- 增加独立的 synthetic robustness sensitivity：在 requested power 上施加预注册的 matched log-power error，报告 communication reference-SNR rate 与 normalized feedback-latency break-even；该 bounded replay 不替代无噪声完整 validation 主结果。
- 增加无噪声 covariance ablation：在同一 train-only artifact 上仅移除不同 topology offset 之间的协方差，同时保留共同 reference measurement 诱导的相关项，以隔离完整 covariance 的贡献。
- 增加两个无噪声防守实验：用 `Topology Open-loop Gain-K` 隔离闭环 posterior update 的贡献，并在预注册 `K={3,5,7,9}` 上报告 TBCP-K、开环控制与 Posterior-TopK 的预算曲线；K=7 继续作为冻结主设定，不根据 validation 曲线改选预算。
- 增加无噪声批量反馈诊断：在相同 K=7 下比较严格串行 `TBCP-7`、`Batch-TBCP-2+2+3`、`Batch-TBCP-2+5` 和平衡的 `Batch-TBCP-3+4`。批内候选只使用当前 posterior 与 train-only gain kernel 贪心选择，批次测量完成后才进行一次联合 posterior 更新；该实验只评估反馈轮次/精度折中，不改变主模型或正式 K=7 claim。
- 增加两个创新点的三 seed 嵌套消融：比较 topology supervision `off/on` 下的 `Direct Prediction`、`Posterior Top-7` 与 `TBCP-7`。报告 topology 主效应、同预算 closed-loop 增益和 difference-in-differences 交互，不重新训练、不访问 test。
- 本 change 不训练或修改模型，不访问 test，不从 validation 拟合 likelihood、噪声、阈值或策略参数；所有 replay 继续标记 `claim_ineligible=true`。

## Capabilities

### New Capabilities

- `sensing-guided-local-beam-probing`: 扩展为 train-only topology likelihood、TBCP-7 闭环 belief update、固定预算 RF query 隔离、强基线与 15-mask validation-only 诊断契约。

### Modified Capabilities

- `pcpf-temporal-risk-fusion`: PCPF-T 的 categorical posterior 仅作为 probing prior；模型结构、参数和 checkpoint 契约保持不变。
- `clean-data-integrity`: 明确允许 train role 的完整 beam-power 只用于独立 probing calibration artifact，同时禁止其进入 sensing 模型或任何 validation/test 拟合。

## Impact

- 闭环策略与校准：`src/kd_sensing/eval/beam_probe_diagnostic.py` 及一个窄的 topology likelihood owner。
- 诊断入口：`tools/eval_pcpf.py probe-diagnostic`，继续复用 checkpoint-bound prediction evidence，不重新运行模型。
- 测试：topology likelihood round-trip/provenance、联合 posterior update、requested-measurement 隔离、15-mask evidence、matched-noise 与 robustness 报告契约。
- 产物：likelihood artifact、逐样本 trace 和报告只写入 ignored `outputs/`；源码与 tracked config 不依赖本地产物。
