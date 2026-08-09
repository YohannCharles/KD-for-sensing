## 1. Posterior Statistics Owner

- [x] 1.1 新增一个模型与 evaluator 共用的纯 beam-posterior helper，校验 `[B,64]` probability、计算 MAP/circular mean/resultant/variance/spread/entropy，并固定零 resultant 回退和 beam-index tie-break。
- [x] 1.2 为 helper 增加 modulo-64 边界、双峰、均匀分布、非法输入和 Top-L shape 测试。

## 2. K=7 Candidate Policies

- [x] 2.1 在现有 probing owner 中实现固定 Local-7、spacing templates `{1,2,4,8}` 和 posterior-mass Adaptive-Local-7，保证 core、唯一性、wrap 和最窄 tie-break。
- [x] 2.2 实现 Posterior-Top7，并保持 Uniform-7/Oracle-Local7 的 oracle 隔离；增加 candidate policy 不可读取 GT、channel、CSI/full gain 的 API 回归测试。

## 3. PCPF-T Integration

- [x] 3.1 在 PCPF-T forward 从 `fused_probability` 添加 detached posterior statistics 字段，不新增 parameter/buffer/loss/optimizer state。
- [x] 3.2 用 synthetic four-modality missing masks 和旧 validation-best checkpoint 做 strict-load、forward shape、state-dict/gradient-set 回归。

## 4. Diagnostic and Provenance

- [x] 4.1 扩展本地 `probe-diagnostic` 以 K=7 运行 Local7、Adaptive-Local7、Posterior-Top7、Uniform7，并把 spacing、spread、entropy 和 candidate indices 写入 sample ledger/report。
- [x] 4.2 保持 simulator 只返回 requested measurements，增加 noiseless `final_beam == gt_covered`、label drift、test sealed、checkpoint/evidence/topology/protocol lineage 校验。
- [x] 4.3 为新策略写入严格 policy version、K、spacing library 和 `claim_ineligible/outer_test_accessed` provenance；禁止覆盖已有诊断目录。

## 5. Verification and Replay

- [x] 5.1 运行 `conda run -n kd_mm_beam pytest tests/test_beam_posterior.py tests/test_beam_probe_diagnostic.py -q` 及相关 PCPF focused tests。
- [x] 5.2 运行 `openspec validate add-uncertainty-adaptive-local-probing --strict`、`openspec validate --all --strict`、compile/config/CLI 边界验证。
- [x] 5.3 只读复用已绑定 validation evidence，在新 ignored 输出目录完成四个 sensing-only pattern 的 K=7 replay；记录结果为 claim-ineligible，不启动训练、不访问 test。
- [x] 5.4 使用完全相同的协议、拓扑、K 和 policy 对 topology-on seed 1/2/3 做完整 validation replay，并按全部 seed 汇总而非选择最优 seed。
- [x] 5.5 根据三 seed replay 和用户决策将 `Posterior-Top7` 固定为主 probing policy，将 fixed/adaptive Local-7 保留为 matched ablations；不把 validation 数字提升为正式 claim。

## 6. TBCP-7 Contract and Calibration

- [x] 6.1 将 active change 从静态 Posterior-Top7 更新为 TBCP-7，并补充 train-only beam-power calibration、联合 likelihood、15-mask 与强基线契约。
- [x] 6.2 实现 train-only topology likelihood fit/save/load，严格验证 protocol train identity、topology、label/power argmax、array integrity 与 artifact digest。
- [x] 6.3 为 calibration artifact 增加 synthetic round-trip、非 train role、sample hash/count、label drift、corrupt artifact 与 covariance 边界测试。

## 7. Closed-loop Policy and Diagnostic

- [x] 7.1 实现 exact relative-dB joint posterior update、expected-terminal-gain next-beam selection 和固定 K=7 TBCP trace；不使用经验 likelihood temper。
- [x] 7.2 实现 `Posterior5+Hill2` 强基线，并用 API/签名测试锁定所有非 oracle policy 只读取 requested measurement。
- [x] 7.3 扩展 checkpoint-bound evidence loader 与 summary，使其严格覆盖 CSI 缺失的全部 15 个四-sensing mask及 Full/drop-1/drop-2/Single 分组。
- [x] 7.4 将 TBCP-7、artifact provenance、ordered measurements、posterior trace 和 matched baselines 接入现有 `probe-diagnostic`，保持非覆盖输出与 test sealed。

## 8. Verification and Replay

- [x] 8.1 运行 probing focused tests、CLI/config/compile 边界、OpenSpec strict validation，并复核旧 checkpoint/state dict 未改变。
- [x] 8.2 复用相同 train-only artifact，在 topology-on seed 1/2/3 的完整 validation 15-mask evidence 上运行 TBCP-7 replay；按全部 seed 汇总，不访问 test。
- [x] 8.3 对比 Posterior-Top7、Local-7、Adaptive-Local-7 与 Posterior5+Hill2，记录 claim-ineligible 结论和真实 RF noise/latency 的剩余限制。

## 9. Robustness Sensitivity

- [x] 9.1 在 proposal/design/spec 中预注册 matched log-power error、hash subset、communication reference-SNR 与 normalized feedback-overhead 边界。
- [x] 9.2 实现稳定 sample/beam keyed dB noise、noise-aware TBCP/Hill2 replay、clean gain/rate 与 coverage-selection 分离指标。
- [x] 9.3 新增独立 sensitivity runner、三 seed summary 和本地 evaluator action，严格绑定 fixed grid/protocol/topology/artifact/test seal。
- [x] 9.4 增加 deterministic common-noise、sigma=0 等价、noise-aware branching、非法网格与三 seed summary focused tests。
- [x] 9.5 在三个 topology-on checkpoint 上完成每 mask 512 个 hash-selected validation 样本的 `{0,3,6}` dB sensitivity replay，不访问 test。
- [x] 9.6 汇总三 seed/replica robustness、reference-SNR rate 与 feedback-overhead break-even，运行 OpenSpec、focused、CLI、compile 和全量验证。

## 10. Covariance Ablation

- [x] 10.1 在 proposal/design/spec 中预注册无噪声 full/diagonal covariance paired ablation，并固定共同 reference 相关项不被删除。
- [x] 10.2 实现严格 covariance mode、同 evaluator 双路径、paired summary 与 synthetic/manual covariance tests，保持默认 full 行为不变。
- [x] 10.3 在 topology-on seed 1/2/3 的完整 15-mask validation 上运行消融并汇总，完成 OpenSpec、focused、CLI、compile 和全量验证。

## 11. Defense Experiments

- [x] 11.1 在 proposal/design/spec 中预注册开环 topology expected-gain 控制和 `K={3,5,7,9}` 无噪声预算曲线，冻结 K=7 主设定。
- [x] 11.2 泛化 TBCP batch runner 到固定合法 K，实现不读取 measurement 的 open-loop topology policy，并保持 K=7 单一实现路径。
- [x] 11.3 将防守实验接入现有 validation-only diagnostic/three-seed summary，记录完整 15-mask、protocol/topology/artifact/test-seal provenance。
- [x] 11.4 增加 open-loop feedback 隔离、K=7 回归一致性、非法预算和 summary binding focused tests。
- [x] 11.5 在 topology-on seed 1/2/3 的完整无噪声 validation 上运行防守 replay，汇总全部预算和 paired delta，不访问 test、不训练模型。
- [x] 11.6 完成 OpenSpec、focused、CLI、compile 与相关全量回归，并记录防守实验结论和限制。

## 12. Batch Feedback Diagnostic

- [x] 12.1 在 OpenSpec 中冻结 `Batch-TBCP-2+2+3`、`Batch-TBCP-2+5` 的 schedule、信息边界、round/update 语义和 claim boundary。
- [x] 12.2 在 topology likelihood owner 增加不改变默认串行行为的 batch-scheduled TBCP runner，批内只用当前 posterior/gain kernel 选 beam，批后用完整相对 measurement 联合更新。
- [x] 12.3 将两个 batch 策略接入 validation-only `probe-diagnostic`，记录 requested indices/measurements、measurement rounds、feedback updates 和 paired metrics；新增非法 schedule 与 requested-only 测试。
- [x] 12.4 在 topology-on seed 1/2/3 的完整 15-mask noiseless validation evidence 上运行 batch 与 sequential 对照，不访问 test、不训练模型，并生成三 seed汇总。
- [x] 12.5 运行 probing focused tests、CLI/config/compile 与 OpenSpec strict validation，记录批量近似的精度/反馈轮次限制。

## 13. Balanced Batch Schedule Diagnostic

- [x] 13.1 在 OpenSpec 中预注册 `Batch-TBCP-3+4`，固定其两轮/一次跨批更新语义，并声明不用于事后选择主策略。
- [x] 13.2 将 `(3,4)` 注册到 batch runner、diagnostic ledger/report、CLI help 与三 seed binding，保持旧 schedule 行为不变。
- [x] 13.3 增加 `(3,4)` 的 runner、requested-only 和 summary binding 回归测试。
- [x] 13.4 在 topology-on seed 1/2/3 的完整 15-mask noiseless validation evidence 上重放 `(3,4)`，并与现有 batch/sequential 结果汇总；不访问 test、不训练模型。
- [x] 13.5 运行 focused/full/OpenSpec/compile 验证，记录 3+4 的精度与反馈轮次折中。

## 14. Nested Innovation Ablation

- [x] 14.1 在 OpenSpec 中冻结 topology `off/on` × `Direct/Posterior Top-7/TBCP-7` 的 2×3 表、matched-budget 对比和交互项定义。
- [x] 14.2 核验 no-topology seed 1/2/3 checkpoint/evidence 与 topology-on 在协议、样本顺序、训练预算和 test seal 上可比。
- [x] 14.3 在 no-topology 三 seed 的完整 15-mask noiseless validation evidence 上运行同一 TBCP diagnostic，并生成严格三 seed summary。
- [x] 14.4 将 topology `off/on` 两份 sealed summary 汇总为嵌套消融表，报告主效应、closed-loop 增益、端到端增益与交互。
- [x] 14.5 运行 OpenSpec、focused/compile 与产物完整性验证，记录消融限制与下一步决策。
