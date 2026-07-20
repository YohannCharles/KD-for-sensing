## 1. P0：严格协议、split 与 fixed-mask 基础

- [x] 1.1 新增只读 MMW evidence split helper：从 strict-v2 train CSV 以一次 group-safe assignment 生成 inner-train/inner-validation/outer-evidence，并输出全角色 leakage/identity audit。
- [x] 1.2 新增 MMW TWC protocol preparer，生成内容寻址 split manifest、15 域 absolute CSV 路径和 immutable fixed-mask cache；已有内容不一致时 fail closed。
- [x] 1.3 为 split、cache、source hash、角色隔离和 cache idempotence 增加单元测试，并使用 `conda run -n kd_mm_beam pytest` 运行。

## 2. P1：物理 codebook topology 审计

- [x] 2.1 新增 MMW codebook topology audit，读取 15 域 metadata/RSU transform，重建 64×64 ULA-DFT local response、label 邻接、0/63 endpoint 关系与 descriptor SHA256。
- [x] 2.2 输出审计 JSON/CSV/figure-ready data，并在 metadata 不一致时产生 unverified 结论而非物理 claim。
- [x] 2.3 为 ULA codebook、yaw-only explanation、endpoint 判据和 metadata-failure path 增加测试，并使用 `conda run -n kd_mm_beam pytest` 运行。

## 3. P2：严格训练、拓扑配置与 provenance

- [x] 3.1 扩展 BPA 配置/loss 为显式 topology descriptor，支持 linear、cyclic、deterministic permutation 和受 audit 约束的 physical descriptor；保留 BPA-disabled control。
- [x] 3.2 新增 strict evidence launcher，生成 H4+RouterNoPattern T2、matched S1、AMBER-Full、RMBP-MM 和预注册 ablation configs，固定 seed `(1,2,3,4,5)`、batch 64、40 epoch、last checkpoint 与 training-mask seed algorithm。
- [x] 3.3 将 evidence protocol、split/cache/topology/provenance 写入 config、checkpoint metadata、evaluation validation 和 summary fail-closed identity checks。
- [x] 3.4 为 strict launcher、topology descriptor、checkpoint provenance 与 H4/RouterNoPattern matching 增加测试，并使用 `conda run -n kd_mm_beam pytest` 运行。

## 4. P3：fixed-mask outer evaluator、统计与论文证据导出

- [x] 4.1 扩展 fixed-mask evaluator，消费 strict cache 的 whole、temporal 与 joint masks，且只评估 outer-evidence split 的 `last.pth`。
- [x] 4.2 新增 strict summary：15-domain/weather/scene/worst-domain、main cells、temporal AUC、paired domain×seed bootstrap CI、coverage 和 baseline-fidelity export。
- [x] 4.3 增加 evaluator/summary 的 identity、all-seed completeness、cache consistency、paired statistics 和 partial-refusal 测试，并使用 `conda run -n kd_mm_beam pytest` 运行。

## 5. P4：完整消融矩阵与 nightly queue

- [x] 5.1 修复可恢复的默认 GPU4--7 nightly queue：用户显式授权 `--allow-gpu0-3` 时可扩展至 GPU0--7，同卡单个本 manifest job；单条失败不得中断已启动 job，orphan 仅可通过 strict `--auto-resume` 续跑，failed evaluation 仅可显式 retry。
- [x] 5.2 冻结并生成主比较 6 methods × 5 seeds，以及 BPA/topology/head/router/temporal/mask/CMA matched-control ablation manifests；每条记录 allowlist 与 matched control。
- [x] 5.3 使用 `conda run -n kd_mm_beam` 完成 protocol preflight、config dry-run、每类方法 one-step smoke 和共同 batch 64 实测验证。
- [ ] 5.4 修复 WholeOnly no-op：新增 480-entry whole-modality-only 平衡 panel、独立 provenance 与回归测试，归档无效 seed1/partial seed2 后重跑 seed1。

## 6. P5：执行、验收与报告

- [x] 6.1 在默认 GPU4--7 启动严格主比较 nightly queue；在用户明确授权后可扩展至 GPU0--7，且不触碰其他 manifest 的进程。
- [ ] 6.2 在主比较完成后以同一 queue 运行完整 matched-control ablation，并只从完整 evidence cell 生成统计。
- [x] 6.3 运行 `conda run -n kd_mm_beam make verify-quick`、`conda run -n kd_mm_beam python scripts/verify_compile.py`、`openspec validate prepare-twc-rigorous-evidence --strict` 与 `openspec validate --all --strict`，确认新产物仅在 ignored output 边界。
- [x] 6.4 输出 P0--P5 protocol/report/plot manifest，明确已完成、排队、失败和不可主张的结论。

## 7. P6：精确 modality-frame temporal stress extension

- [x] 7.1 生成版本化 `mmw_twc_temporal_token_stress_v3` cache/manifest：20--90% 各 100 个随机化、集合级 cell/modality/frame 精确平衡 masks，95% 为 20 个唯一 single-cell masks；允许单 mask 模态组成偏斜并记录分布审计。
- [x] 7.2 扩展 strict evaluator 与 summary，使其记录 evaluation-extension、token 数、随机平衡策略和完整可复现实验参数；主曲线只聚合至 90%，95% 作为 single-cell extreme stress 独立导出。
- [x] 7.3 为 exact cardinality、randomized balanced coverage、parent-protocol binding、identity refusal 和 90/95% summary 增加测试。
- [ ] 7.4 在最新四方法×五 seeds 的正式 `last.pth` 全部生成后，运行完整 v3 stress evaluation 并导出报告。

## 8. P7：四方法公平训练与 Pattern-weighted CE 退役

- [x] 8.1 新增互斥的公平外部训练 mask schedule：Clean 20%，Drop1/2/3 各 10%，TokenDrop20/40/60/80/90 各 10%；按 seed/epoch/step 可复现并输出 per-sample condition id 与集合级模态/cell 审计。
- [x] 8.2 让 T2、S1 只消费外部 availability，删除第二套 `p_missing` 采样，并保持两者使用普通 Beam CE。
- [x] 8.3 将 strict launcher、evaluator、summary 和 provenance 收敛为 T2、S1、AMBER-Full、RMBP-MM 四方法×五 seeds。
- [x] 8.4 补充配置、mask公平性、四方法完整性和负路径测试；生成清理 manifest，结束旧暂停队列并删除 `outputs/` 中被替代的旧实验结果，保留数据派生 cache、cleanup manifest 与 codebook audit。
- [x] 8.5 归档 Pattern-weighted CE 的 seed1 开发筛选证据，删除其 checkpoint、待跑任务、配置/runtime/launcher/summary 分支与测试，并记录退役原因。

## 9. TWC 比较面与通信证据扩展

- [x] 9.1 新增 MaskTrain-CLS 与 AMR-Net-4M-Adapted 的 modular representation core、canonical configs、auxiliary loss/fidelity metadata 和 synthetic/focused tests；扩展 MMW strict launcher 为六方法五 seeds。
- [x] 9.2 新增 `deepsense6g_twc_secondary_v1`：将 Scene31--34 合并为一个完整数据集，运行五方法×三 seeds=15 个训练任务，并提供 pooled fixed-mask evaluator、跨 seed summary、provenance 和 tests；协议冻结剔除无效 `future_beam1` 后的派生 CSV 与审计哈希。
- [x] 9.3 扩展 MMW/DeepSense evaluator：读取 future beam-power vector，输出 normalized gain、gain loss dB、SNR 0/10/20 dB spectral-efficiency ratio/rate loss，并增加 exact-value tests。
- [x] 9.4 新增 BPA/router 定量机制分析，输出 physical error CDF、far-error、clean-to-missing drift、prototype neighbor margin 和 router-oracle alignment；PCA/t-SNE 只保留为辅助。
- [x] 9.5 新增固定推理期 corruption cache/evaluator，覆盖 GPS 噪声、image 遮挡/模糊、Radar 噪声和 LiDAR 稀疏化；不得改变 checkpoint 或训练 recipe；默认不执行，只有显式 `--run-reliability-stress` 才生成/运行压力测试 shards。
- [x] 9.6 新增统一 complexity profiler 与 paper table，记录 params、可用时的 MACs、batch1/batch64 latency、throughput、peak memory、硬件和 AMP policy。
- [x] 9.7 新增 manifest-driven `outputs/twc_experiment_status.md` 生成器，记录已完成、运行、排队、失败、缺失和 claim eligibility；任务类型/范围 MUST 唯一显示，corruption severity 等后处理维度不得折叠成重复行。

## 10. 扩展实验执行

- [ ] 10.1 先完成 MMW 当前方法与 WholeOnly 的 seed1 gate、固定 outer evaluation 和初步分析；在用户确认前暂停 seed2--5，确认后再按同一 immutable manifest 补齐。
- [ ] 10.2 MMW gate 通过后，在 Scene31--34 合并数据集上运行 DeepSense6G 五方法三 seeds，并执行 pooled fixed-mask、wireless、mechanism 和 complexity evaluation；corruption/reliability stress 仅在显式 `--run-reliability-stress` 后追加。
- [ ] 10.3 完成扩展协议验证、完整性审计和论文表图；任何未完成 cell 保持 pending，不进入 claim。

## 11. Tie-aware Router oracle 开发筛选

- [x] 11.1 扩展 Router oracle config/loss，支持 hard-first control、confidence tie-break、uniform/confidence exact-tie soft target、distance-soft 与 distance+confidence soft target，并输出 tie/entropy diagnostics。
- [x] 11.2 增加 focused tests，验证并列时无固定 Image 偏置、缺失模态 target mass 为零、soft loss/gradient 有限以及旧 hard-first control可复现。
- [x] 11.3 新增 `mmw_tie_aware_router_screen_v1` inner-only launcher/manifest，固定 seed1、40 epoch、batch64、H4、RouterNoPattern 和相同训练 mask；八候选各绑定 GPU0--7，outer CSV 不得进入 config。
- [x] 11.4 完成八候选训练、inner fixed-mask evaluation 与 ADBA-first/Top-1-secondary 汇总；用户审阅前不修改 canonical T2，也不把筛选结果升级为 outer claim。

## 12. Oracle Gap 可靠性机制压力测试

- [x] 12.1 冻结 `mmw_router_oracle_gap_v1`：同一 `SoftConfidenceTie` seed1 inner checkpoint、13 个 clean/corruption conditions、物理单位 severity、通信效用 Oracle、trace schema、主辅指标与 claim-ineligible 边界。
- [x] 12.2 将 corruption runtime 收敛为物理 GPS XY 噪声、目标 SNR Radar AWGN、空间一致 LiDAR cell dropout 和 seeded image occlusion，并增加确定性、单位变换、availability 不变和 severity 检查。
- [x] 12.3 实现同 checkpoint Uniform/Learned/Oracle evaluator、GPU0--7 condition-shard launcher、compressed trace、完整 provenance 与汇总图表。
- [x] 12.4 完成 15-domain×13-condition 评估，报告 ADBA、normalized gain、rate ratio、gap closure、Router oracle regret 和受损模态权重单调性，保留负结果。

## 13. Router utility alignment 与 paired monotonic seed1 筛选

- [x] 13.1 冻结 `mmw_router_utility_screen_v1` 的八候选、training-only power-vector contract、online paired corruption、quality-gated monotonic 公式、独立 train/eval seed 和 inner-only 边界。
- [x] 13.2 实现可选 future beam-power/scaler batch 字段、beam-power utility Router target、detached paired Router forward 与 monotonic loss，并增加 focused tests。
- [x] 13.3 实现八卡 seed1 launcher、每候选固定 Oracle Gap 后评估、跨候选 summary 与完整 provenance；seed2--5 默认不得生成。
- [x] 13.4 通过 config/focused/compile/OpenSpec 与 batch64 smoke 后，在 GPU0--7 启动八候选 overnight queue；完成后分析但不自动晋升 canonical T2。

## 14. Expected-utility Router 修复筛选

- [x] 14.1 冻结 v1 零激活与 v2 近均匀 target 负结果、continuous expected-utility/entropy gate、八个 seed1 候选和 inner-only 边界。
- [x] 14.2 实现 expected beam-power utility、main/paired target 解耦、连续 monotonic gate 与 diagnostics，并增加 focused tests。
- [x] 14.3 实现 trace preflight、v2 八卡 launcher、隔离式 Oracle Gap worker 与 summary/provenance。
- [ ] 14.4 通过 OpenSpec/config/focused/compile、12-cell preflight 和 batch64 smoke 后，在 GPU0--7 启动八候选并完成固定评估。

## 15. P8：Joint Drop+Corrupt Router 诊断

- [x] 15.1 冻结 `mmw_router_joint_stress_v1`：CurrentControl seed1 inner checkpoint、Joint20/40/60/80、每级 20 个三状态精确平衡 masks、固定 S2 corruption、同 checkpoint Learned/Uniform/Oracle 和 claim-ineligible 边界。
- [x] 15.2 实现内容寻址三状态 cache、token-selective corruption 与 unavailable-aware Uniform/Oracle，并增加 cardinality、cell/modality/frame balance、确定性、selector 和 branch masking focused tests。
- [x] 15.3 实现 condition evaluator、GPU0--7 manifest launcher、compressed trace 与 paired summary；完成 one-domain/one-batch smoke 和 OpenSpec/focused validation。
- [x] 15.4 在 GPU0--7 完成 CurrentControl seed1 的 15-domain×81-condition 固定评估，输出 ADBA、normalized gain、rate ratio、paired CI/win rate、gap closure、regret 和权重响应。
- [x] 15.5 按 Joint40/60/80 的预注册结果冻结结构决策：当前 Router 通过则停止扩结构；未通过则只记录 R1/R2 独立 change 的失败模式与最小筛选范围，不在本任务自动修改 canonical T2。
- [x] 15.6 对已保存 trace 运行独立 static-prior falsification，比较 Dynamic、GlobalCleanPrior、PerSampleClean 与 Uniform，输出 paired-domain CI 和 claim 边界；不得改变 15.5 的预注册 Gate。

## 16. P9：DeepSense6G Router falsification

- [x] 16.1 冻结 `deepsense6g_router_falsification_v1` 的 epoch20 T2 seed1 checkpoint、Scene31--34 pooled test、65 个 whole/token fixed masks、四分支与 claim-ineligible provenance。
- [x] 16.2 实现按 Scene31--34 分片的同 checkpoint trace evaluator、GlobalCleanPrior 离线融合、pooled/day/night summary 和聚焦测试。
- [x] 16.3 仅在 GPU4--7 完成四 scene 固定评估，汇总 Learned/Uniform/GlobalCleanPrior/Oracle 的 ADBA、normalized gain、rate ratio、Router regret、权重与 Oracle modality frequency，并据此决定是否恢复正式 40 epoch 队列。
