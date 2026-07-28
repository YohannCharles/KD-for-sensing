## 1. 模型与原型状态

- [x] 1.1 在 `BeamPrototypeBank` 与 U0 输出中实现只读、无标签的 prototype state 导出，并覆盖其数值不改变 logits 的测试。
- [x] 1.2 实现冻结 U0 包装器、mask bias 与共享低秩 `MissingDecisionAdapter`，确保 Full 显式旁路、B 零初始化和 prototype state detach。
- [x] 1.3 使用 `conda run -n kd_mm_beam pytest` 添加并运行 Adapter 的冻结、掩码顺序、零初始化、Full 等价和 A7 置换隔离单元测试。

## 2. 受审计的 Stage A 工作流

- [ ] 2.1 实现 protocol/checkpoint fail-closed 审计、U0 结构审计、确定性 20-epoch mask schedule、split 内 A7 置换和 train-only condition normalizer。
- [ ] 2.2 实现只训练 Adapter 的 AdamW/cosine/warmup 训练、15-mask 逐样本 NPZ 评估、Full 等价审计、聚合/Retention/SPA/行为诊断与运行 manifest。
- [ ] 2.3 使用 `conda run -n kd_mm_beam pytest` 覆盖 clean protocol fail-closed、validation 状态不变、mask schedule 一致及内部指标检查。

## 3. 独立分析与启动表面

- [ ] 3.1 新增 `tools/analyze_prototype_decision_adapter.py`，从逐样本产物独立 NumPy 重算指标、bootstrap、精确 McNemar 和 Holm 校正。
- [ ] 3.2 新增 GPU 0--7 Stage A 启动脚本及仅生成不执行的 Stage B 多 seed 脚本，记录 GPU/PID/命令/日志/状态且不终止无关进程。
- [ ] 3.3 更新架构边界测试以显式允许受控实验脚本，确保仍无新增 public CLI 或 canonical recipe。

## 4. 验证与 Seed 1 筛选

- [ ] 4.1 使用 `conda run -n kd_mm_beam` 运行 OpenSpec、模型/协议/架构测试和编译检查。
- [ ] 4.2 在启动前验证 checkpoint SHA256、clean protocol/audit、GPU 状态与 Stage A 配置；不满足任一条件时 fail-closed。
- [ ] 4.3 在 GPU 0--7 执行 Seed 1 Stage A，运行独立重算与统计，并根据预注册门槛生成阶段性结论及 Stage B 建议。

## 5. Full-pool capacity protocol

- [x] 5.1 实现 46,860 候选窗口发现、trajectory 可靠性判定、共享时间轴 80/20 macro split、真实边界 purge 和 588 历史身份的 fail-closed 恢复。
- [x] 5.2 复用 Radar/BS-GPS 确定性增强，输出 protocol/cache manifest、每 domain/Beam 支持量并在增强后完成资源级零交叉审计。
- [x] 5.3 扩展受控实验 loader/config 审计以接受 `mmw_full_pool_development_v1`，不放宽既有 clean-inner 或 public CLI 边界。

## 6. Full-data 两阶段运行

- [x] 6.1 实现测时、动态 epoch 计算和唯一 Full-data U0 的 GPU4 Stage 1，从头训练并发布带 SHA256 的 `last.pth`。
- [x] 6.2 新增 Full-pool 编排脚本，在 checkpoint 哈希通过后按当前实验范围并行运行 Adapter，并每 600 秒记录运行状态。
- [x] 6.3 保存四个实验的 15-mask 逐样本结果、Full 等价证据和内部指标，使用独立脚本以 `1e-7` 容差重算聚合、Retention、SPA、MAE、Within-3 与 ADBA。
- [x] 6.4 运行 OpenSpec、focused tests、架构/CLI/compile 验证，完成 Full-data 与旧 3,600 结果的描述性对照和证据结论。
- [x] 6.5 修正测时 probe 被 resume 为正式 U0、reference clean U0 profile 漂移和 prototype 塌缩未 fail-closed 的问题，并添加聚焦测试。
- [ ] 6.6 保留无效 FP16 run 证据，按修正协议从头重跑唯一 U0 和 GPU1--7 上的 A0--A7，重新生成独立指标与最终结论。
- [x] 6.7 将 Full-pool 作业从不可靠 CUDA ordinal 改为物理 GPU UUID 绑定，拒绝 GPU0，并以聚焦测试和运行 manifest 验证 GPU1--7 映射。
- [x] 6.8 并行构建 pooled domain 与 train-only GPS moments，固定顺序归并并验证与单线程统计一致。
- [x] 6.9 将 Stage 2 扩展为 GPU1 上 A0→A7 队列及 GPU2--7 上 A1--A6，并验证八项独立返回码和 GPU0 零使用。
- [x] 6.10 将 prepared CSV 的重复资源检查改为逐单元向量化验证、按真实路径去重及总并发不超过 90 的 fail-closed 并行校验，并覆盖缺失资源、Radar `_DA` 和非法 Beam 回归测试。
- [x] 6.11 将 Full-pool split audit 从逐行全列扫描改为保持同一身份口径的向量化资源集合与 `itertuples` 强哈希/帧展开，并用手工期望集合验证零交叉输入不变。
- [x] 6.12 严格复用现有 Image/LiDAR 帧缓存，生成 protocol-bound GPS 坐标与 scaler artifact，完成数值/覆盖/吞吐审计，并仅在 1.5x 吞吐门槛失败时构建唯一帧 LMDB。
- [ ] 6.13 为 U0 与 A1--A7 实现预注册训练损失早停，记录实际 epoch/steps/stop reason，按用户授权在物理 GPU0/4/6 启动 Stage 2，并在 GPU7 空闲后无重复地迁移尚未启动的 A7、在 A5 完成释放 GPU6 后迁移尚未启动的 A6。
- [x] 6.14 实现固定 `lambda=0.5`、`sigma=2.0` 的 circular ADBA-surrogate loss profile、B1/B4/B6/B7 四 GPU 编排、独立 ADBA 重算与聚焦测试，且保持原 CE profile 数值不变。
- [ ] 6.15 在物理 GPU0/4/6/7 并行运行 B1/B4/B6/B7，完成 15-mask、Full 等价、B6-B7 对照和最终 ADBA 导向分析。
- [x] 6.16 实现 Global/Lookup/Factorized bias、允许 mask schedule、确定性分层 fold、weight-space probe、条件四卡编排与独立重算，并以聚焦测试验证零初始化、Full 旁路和 held-out 零曝光。
- [x] 6.17 复用 B1 并运行单 seed Global/Lookup；若 MLP All-14 ADBA 严格高于 Lookup，则运行 fold 0 的 MLP/Factorized unseen pilot，输出阶段门槛和 held-out 结果。
- [x] 6.18 实现半径为 3 的 `circular_transport` 概率算子、Full 显式恒等、局部核审计、all-seen Factorized 对照和独立指标重算，并以聚焦测试验证概率质量、回绕和组合顺序。
- [x] 6.19 在物理 GPU0/4 上按相同 8-epoch ADBA-surrogate 协议并行运行 Circular Transport 与 all-seen Factorized Bias，输出 A0/B1/Factorized/Transport 的阶段性比较。

## 7. Full-Pool BT-SCL

- [x] 7.1 实现 Full-pool protocol/input/topology 强制审计、训练期 normalization manifest 与均衡 nested-subset schedule。
- [x] 7.2 实现专用轻量四模态 token-Fusion-Prototype 模型、R0--R5 预注册损失与 R5 固定 round-robin 日程；不得接入 Router、attention 或 Adapter。
- [x] 7.3 实现训练、统一选择、15-pattern 评测、拓扑/粗细一致性/证据/梯度/分域诊断和独立汇总工具。
- [x] 7.4 添加并运行模型、数据隔离、topology、schedule、auxiliary 与 smoke-test 覆盖。
- [x] 7.5 以物理 GPU0--5 启动单 seed R0--R5，记录 PID、状态与日志；不启动 multi-seed、outer test 或下一轮。
- [x] 7.6 实现 R6 标签锚定 4/8/16-sector 层次损失、半径 0/3/5 随机占优损失、机制诊断与 R0 只读基线补算，并添加聚焦测试。
- [x] 7.7 保留失稳 R6 轨迹，实施统一 stable profile 并从相同初始化并行训练 R0/R3/R6，完成 15-pattern、机制诊断与创新性判断；不得自动启动 multi-seed 或 outer test。

## 8. Full-Pool Candidate12

- [x] 8.1 实现 Candidate12 Full-pool/input/BPA/topology/signed-angle fail-closed 审计与 27 项 preflight。
- [x] 8.2 实现共享 prototype 模型、公共 BPA、KL/risk assignment、15%--40% 容量调整及 mixed/homogeneous remix 日程。
- [x] 8.3 实现非环形 PAMR shift、局部 offset loss、Dynamic/Zero/Mean/Shuffle/Oracle 替换与机制诊断。
- [x] 8.4 实现统一 warm-up/训练/评测/聚合工具、GPU0--5 启动与 600 秒监控脚本，并运行聚焦测试和 smoke tests。
- [x] 8.5 训练唯一 5-epoch warm-up，缓存 train-only prototype risk，并发布公共 checkpoint SHA256。
- [x] 8.6 从同一 warm-up 并行运行 A0--A5，完成 Full 主结果、assignment/gradient/motion/效率诊断、success gates 与唯一候选结论；不得自动启动 multi-seed 或 outer test。

## 9. Full-Pool BTMA Causal Ablation

- [x] 9.1 新增统一 BTMA assignment 模块，覆盖固定随机均衡、历史 A1 固定比例、KL、topology risk、prototype margin 和 risk+margin；B0/B1 不执行模型难度打分。
- [x] 9.2 新增独立运行器、37,038/9,180 preflight、train-only cache manifest、B5 历史 A2 复现检查与因果归因汇总。
- [x] 9.3 新增 B0--B5 协议测试和 GPU0--5 启动/600 秒监控脚本；不自动运行 multi-seed、outer test 或下一轮。

## 10. BTMA 负结果只读收尾

- [x] 10.1 新增只读收尾工具：从六个 BTMA `best_checkpoint.pt` 按 `evaluate()` 口径重算 pattern=`full` 的逐样本 `anchor_logits`，导出 `validation_predictions.npz`；不重训、不改协议。
- [x] 10.2 实现 `(domain, cav)` 内连续帧块（块长 32、重抽 2,000 次，计算前固定）的成对 temporal block bootstrap，输出各方法对在 Top-1/Top-3/Within-3/MAE/topology risk 上的点差与 95% CI。
- [x] 10.3 实现纯 numpy 的 score correlation：epoch 5 主表（score 与同源 warm-up 单模态环形拓扑误差的每模态 Spearman）、后续 epoch 追踪表、跨 epoch 秩稳定性表。
- [x] 10.4 新增收尾测试覆盖块划分、成对重抽确定性与相关性口径；发布收尾报告并显式声明不得据此重开 BTMA 路线。

## 11. Router 可观测性因果筛选

- [x] 11.1 新增冻结 Full-pool U0 的表征缓存：钩取各 encoder 末层线性变换输入与 `latent_sequence`，并以逐样本等价性测试证明缓存重算的融合 logits 与直接前向一致。
- [x] 11.2 新增 quality 分支与四条嵌套 router 输入路线 Q0--Q3，断言 Q3 与 Q2 参数量严格相等，且置换只作用于投影前特征。
- [x] 11.3 按既有 diagnostic sample manifest 重新实现 45 个腐蚀条件算子，并以固定种子为每个样本预抽唯一条件。
- [x] 11.4 新增运行器与测试，在设定 N 与设定 C 下各运行 Q0--Q3 × 3 个 router seed；不训练 encoder，不启动多 seed 骨干或 outer test。
- [x] 11.5 对处理组执行冻结权重推理期消融（quality embedding 替换为训练集均值），并按预注册门槛输出 success gates 与筛选报告。
## 12. Prototype-Conditioned Sparse Pilot Transition

- [x] 12.1 完成训练入口、dataset/collate/forward/evaluator、四模态时间对齐、prototype/topology、47,100 个 channel shape 与历史 label 生成审计，输出 `docs/prototype_pilot_transition_audit.md`。
- [x] 12.2 实现固定恒模 probe codebook、path-domain sparse pilot simulator、AWGN/dropout、frequency fail-closed 与 hash-aware noiseless cache，并添加 shape、Nt/Nr、数值一致性、SNR、复现和失效测试。
- [x] 12.3 扩展 MMW row contract 只导出最后输入帧 `channel_ref` 和 frame metadata，完成至少 100 样本的 target/future channel 泄漏诊断；不得把 path tensor放入 batch 或 model。
- [x] 12.4 实现 prototype selector/offline lookup 导出、SparsePilotEncoder、局部/全局 PrototypeTransition、reliability fallback 与只把 selected M patterns 交给 encoder 的单元测试。
- [x] 12.5 实现受 Full-pool clean protocol 约束的本地 Stage A workflow、配置开关、C0--C6/SNR/budget/Fix-Harm/latency 指标和独立结果汇总；关闭开关时保持四模态基线。
- [x] 12.6 生成小规模 cache，运行 single-batch forward/backward 与 100--500 sample smoke，保存 resolved config、lookup、diagnostics 和 summary 到 `outputs/sparse_pilot_transition/`。
- [x] 12.7 运行 C0/C2/C3/C4/C5/C6 单 seed 短程诊断；只有 Proto+Pilot 与 learned lookup 门槛通过后才提出长实验建议，不自动启动正式长训练或 outer test。
- [x] 12.8 运行 OpenSpec strict、focused channel/model/data/config tests、`make verify-quick`、CLI config 与 compile 验证，并记录剩余实验风险。

## 13. Dense-to-Sparse Pilot Curriculum

- [x] 13.1 预注册 `32x16 -> 16x16 -> 8x8 -> 4x8` 四阶段、嵌套频点、每阶段 2 epoch、独立产物根和停止解释，并通过 OpenSpec strict。
- [x] 13.2 让 selector、训练与评估路径支持运行时 M/Kp 子预算，保证 encoder 只读取所选嵌套 observation，并添加预算单调、资源计数和禁用路径测试。
- [x] 13.3 构建 16 频点 train-only cache，运行 100/100 单 seed curriculum smoke，输出逐阶段 budget summary、最终 C0--C6/SNR/15-mask/dropout、lookup 与报告；不得访问 outer test。
- [x] 13.4 运行 focused/full tests、OpenSpec、CLI/config、architecture 与 compile 验证，根据 dense 上界和 T4x8 结果判断问题来自 pilot budget 还是 transition，不自动启动正式长训练。

## 14. Matched-Update Pilot Budget Controls

- [x] 14.1 预注册 GPU0 D32x16-only、GPU1 4x16-only、GPU2 T4x8-only、GPU3 curriculum 四路 8-epoch 对照及失败不抢占规则，并通过 OpenSpec strict。
- [x] 14.2 为本地 runner 增加受限 budget-arm/method 选择和共享 codebook 读取，保证 resolved config 记录实际 arm 且原产物不被覆盖，并添加解析测试。
- [x] 14.3 在 GPU0--3 UUID 隔离并行运行四路 100/100 matched-update 诊断，保存各自返回码与只读汇总；不得终止既有进程、访问 outer test 或自动重跑失败任务。
- [x] 14.4 汇总四路 Top-k、Fix/Harm、route/alpha 与资源，运行 focused/full/architecture/config/compile/OpenSpec 验证并给出预算与 transition 的归因结论。

## 15. Scale-Up Sample and Epoch Diagnosis

- [x] 15.1 预注册 2,000/1,000、40 epoch、batch 8、15-domain 均衡无重复子集、四路 GPU0--3 与失败不重跑边界，并通过 OpenSpec strict。
- [x] 15.2 为本地 runner 增加可审计 scale-up epoch、均衡子集、分块 evaluation、逐 epoch loss/route/alpha/梯度历史和 trainable-only checkpoint，添加聚焦测试且保持默认 smoke 行为。
- [x] 15.3 在 GPU0--3 并行运行四路 scale-up，保存独立 config、history、checkpoint、返回码和最终指标；不得访问 outer test 或终止既有进程。
- [x] 15.4 汇总学习曲线和最终 Top-k/Fix-Harm/route/alpha，运行 full/focused/architecture/config/compile/OpenSpec 验证，并判断样本/轮次假设是否成立。

## 16. Sparse CSI Missing-Modality Fallback

- [x] 16.1 预注册严重缺失训练权重、CSI-only 直接监督、availability-aware gate、Single/Worst/All-14 主指标与 Full 不伤害约束，并通过 OpenSpec strict。
- [x] 16.2 实现可审计 mask schedule、CSI-only prototype fallback、gate target/loss、severe-validation history 和聚焦测试，保持 CSI-off 精确回退。
- [x] 16.3 在 GPU0--3 以 2,000/1,000、40 epoch 并行运行四个 pilot budget arm，保存独立产物且不访问 outer test。
- [x] 16.4 汇总严重缺失 CSI-on/off 增益与 Full 约束，完成 full/focused/architecture/config/compile/OpenSpec 验证并给出兜底能力结论。

## 17. Missing-Fallback Progressive Pilot Budgets

- [x] 17.1 预注册 D16x16、S8x16、S16x8、S8x8 四个独立 40-epoch 中间预算、共享 missing-fallback profile 与失败不重跑边界，并通过 OpenSpec strict。
- [x] 17.2 扩展受限 budget-arm 解析与聚焦测试，保持原四路和默认配置不变。
- [x] 17.3 在 GPU0--3 并行运行四个中间预算，保存独立产物且不访问 outer test。
- [x] 17.4 联合既有 512/64/32 RE 结果定位可学习性断点，完成全量验证并给出目标 sparse CSI 兜底结论。

## 18. Sparse-Pilot CSI 信息恢复诊断

- [x] 18.1 审计现有时间对齐、标签、p0/q_local/alpha/preserve、梯度与 2,000/1,000 抽样/类别/U0 错误分布，输出 `outputs/sparse_pilot_recovery/audit.md`；不得访问 outer test。
- [x] 18.2 为显式 recovery diagnostic 增加逐帧校验的五帧历史 channel 引用、从既有 `beam5` 功率 artifact 生成当前标签及聚焦测试，保持默认 dataset contract 不变。
- [x] 18.3 实现固定 4x8 QPSK 的 I0--I5 本地信息诊断、单帧/两层 GRU 历史编码、完整 epoch 日志、early stopping 和可恢复 checkpoint，并添加聚焦测试。
- [x] 18.4 在 GPU0--3 对 seed 1/2/3 运行 2,000/1,000 development 信息诊断；该运行后确认误绑定旧 Full-pool protocol，结果只保留为无效历史证据，不得进入 Stage A1。
- [ ] 18.5 运行 OpenSpec strict、focused data/model/metric tests、architecture/config/compile 与全量 pytest，记录修改文件、命令、进程和剩余风险。

## 19. Trajectory Recovery 协议纠错与三轮改进

- [x] 19.1 将 recovery 精确迁移到 `mmw_trajectory_disjoint_v1` 与最新 M4 checkpoint，复核 37,510/6,365、split/checkpoint hash、Full Top-1 和 test 封存。
- [x] 19.2 实现一次性 32x16 母 cache、嵌套预算读取、15-mask 冻结 M4 表征、完整数据训练日志与聚焦测试，不覆盖旧产物。
- [x] 19.3 Round 1 在 GPU0--3 完成 32x16 I1--I3 信息上界并筛查 I4/I5；I3 达 95.60%，而 I4/I5 三 seed 已观测 Worst 至多 75.37%，据预注册信息判据停止 concat 并转向 CSI-only fallback。
- [x] 19.4 Round 2 从 dense I3 checkpoint 并行微调 16x16、8x16、16x8、8x8、4x16、4x8；seed-1 Top-1 分别为 96.15%、93.95%、95.54%、91.77%、80.46%、76.43%，16x8 未进入距最佳 0.5 个百分点的近优区间，故选择 16x16（256 RE）。
- [x] 19.5 Round 3 在 16x16 上运行三个固定 availability fallback seed：Single Macro/Worst 均值 96.24%，All-14 Macro 89.09%、Worst 25.61%，Full Top-1 86.3315% 且逐样本概率差 0、argmax mismatch 0；focused 38、full 381、architecture/config/compile/lint/OpenSpec 验证均通过，test 未访问且未自动启动 Stage A1。
