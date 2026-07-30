## 1. 边界与配置契约

- [x] 1.1 新增严格 PCPF-T model/loss 配置解析与 stage、维度、非负系数、禁止输入校验
- [x] 1.2 注册独立 `pcpf_temporal_risk_fusion`，确认 U0 构造与 state dict 不包含 PCPF owner
- [x] 1.3 新增 `tools/configs/pcpf/` 的 base、三阶段、A0--A4 和分量消融模板，不修改 canonical MMW recipe

## 2. Temporal Expert 与 Prototype

- [x] 2.1 实现唯一共享的逐模态 Temporal Transformer、真实 `[B,T,4]` padding mask 和全缺失清零
- [x] 2.2 实现四 encoder、projection、唯一 `BeamPrototypeBank` 与 deterministic unimodal probability forward
- [x] 2.3 实现 availability-aware unimodal hard/soft topology loss，并复用现有 prototype alignment

## 3. 概率嵌入与拓扑风险

- [x] 3.1 实现共享 ProbabilityEmbeddingHead、零初始化 identity 行为与训练采样/eval deterministic 路径
- [x] 3.2 实现 FP32 `U_var`、`U_proto`、circular `U_temp`、JS `U_conflict` 和 disabled-component 消融
- [x] 3.3 实现 detached `R_star`、masked Huber、pair ranking、Gaussian KL 与 topology preserve loss
- [x] 3.4 实现只读 train-only normalization 与 `mean_train_risk` stage preparation、buffer 和 provenance

## 4. 解析融合与阶段训练

- [x] 4.1 实现 uniform、static prior、analytic PCPF、direct Router control 和 `cuaf_local_adaptation` 概率融合
- [x] 4.2 实现 FP32 temperature、static capability、risk/tau log-score 与 missing/Single 权重约束
- [x] 4.3 实现三个 stage 的精确冻结、frozen module eval、trainable name/count metadata 和断言
- [x] 4.4 接入共享 trainer 的 opt-in preparation、stage-specific loss 和 validation loss，保持 U0 默认路径不变
- [x] 4.5 扩展 checkpoint model metadata、source-stage fail-closed 校验和 PCPF validation-best stage alias

## 5. 评估与本地工具

- [x] 5.1 实现 Stage 2 gate 的相关性、分位、常数化、天气/domain/mask-group 报告和固定阈值判定
- [x] 5.2 扩展 15-mask 评估输出 weight/risk/calibration/temperature/claim provenance 诊断
- [x] 5.3 实现 `tools/run_pcpf.py` 的 resolve/preflight/synthetic/one-batch smoke 与 `tools/eval_pcpf.py` 的 gate/matrix 入口
- [x] 5.4 对所有 Stage 3 训练请求校验 gate JSON、SHA256、通过状态与 Stage 1 expert fingerprint

## 6. 测试与验证

- [x] 6.1 使用 `conda run -n kd_mm_beam pytest` 覆盖 temporal shape/mask、四项风险、概率融合、stage freezing 和数值稳定性
- [x] 6.2 使用 `conda run -n kd_mm_beam pytest` 覆盖配置拒绝、checkpoint 来源、gate、15-mask 聚合和 U0 隔离
- [x] 6.3 使用 `conda run -n kd_mm_beam` 完成 synthetic 三阶段 forward/backward smoke 并记录 shape、loss、梯度、权重和 NaN/Inf
- [x] 6.4 使用 `conda run -n kd_mm_beam` 完成真实 MMW 单 batch Stage 1/2/3 smoke，不访问 outer test、不启动长训练
- [x] 6.5 运行 `openspec validate add-pcpf-temporal-risk-fusion --strict`、`make verify-quick` 与相关 focused tests，记录既有 active-change 风险

## 7. 主线收敛与历史归档

- [x] 7.1 为三个历史 active change 写入真实 closure note，并保存仓库外快照；不把未完成任务伪装为完成
- [x] 7.2 从当前工作树移除已停止且孤立的 probabilistic prototype owner
- [x] 7.3 保留并验证 U0、AMBER-Full、RMBP-MM、DeepSense6G、clean/trajectory protocol 与全部 cache 边界
- [x] 7.4 更新 current specs、README、agent navigation、maintainer context 和 architecture boundary，使 PCPF-T 成为唯一 active research mainline
- [x] 7.5 成组移除已停止且无保留路线反向依赖的 SMSL R5、CPSU、Router observability/conformal 与 BTMA closure owner，并删除根目录原始 PCPF 提示和汇报 PDF

## 8. Stage 2 数值塌缩修复

- [x] 8.1 将风险 normalization std floor 固定为 `0.01`，同步正式配置与模型默认值
- [x] 8.2 增加退化 `U_var` preparation 后单优化步回归测试，并完成 focused/OpenSpec 验证
- [x] 8.3 补齐 15-mask matrix 的 checkpoint、clean protocol、gate 与 normalization provenance，并增加聚焦回归测试
- [x] 8.4 让 A0--A3 control 在同一次 A4 forward 上应用各自 validation-best 参数，记录 control provenance 并完成 seed1 公平矩阵

## 9. 历史 sparse CSI 数据契约

- [x] 9.1 审计 PCPF-T sparse-CSI、clean-inner/trajectory split、历史 channel 时间顺序、固定 2x2 selection、codebook/checkpoint identity，并记录 `docs/pcpf_sparse_csi/source_audit.md`
- [x] 9.2 实现 clean-inner 样本自身五帧历史 channel 到 `[5,2,2]` complex sparse CSI 的确定性 sidecar/cache，写出 selection/codebook/frequency/history provenance
- [x] 9.3 让 SparsePilotEncoder 在真实 SNR 缺失时显式工作并输出 `snr_available=false`；禁止 AWGN、dropout、corruption 和伪造 SNR
- [x] 9.4 将 sparse-CSI 正式数据绑定迁移到 `mmw_trajectory_disjoint_v1` 的 37,510/6,365 train/validation，复用 train-only GPS scaler并拒绝 outer test
- [x] 9.5 在训练前扫描 trajectory train/validation 补齐固定 2x2 CSI cache，保存协议、样本数、唯一 channel 数与 `outer_test_accessed=false` 清单

## 10. 五模态模型与初始化

- [x] 10.1 为 PCPF-T 增加 `use_sparse_csi` opt-in，默认四模态不创建 CSI owner
- [x] 10.2 实现 CSI temporal encoder/projection、五模态共享 Transformer/prototype/probability/risk/fusion 与完全缺失硬置零
- [x] 10.3 删除四转五 checkpoint 迁移，trajectory sparse-CSI Stage 1 只允许 fresh start
- [x] 10.4 实现确定性等频 31-subset 训练 schedule、逐 epoch mask 计数和相同缺失 mask 的五路传播
- [x] 10.5 增加本地 sparse-CSI Stage 1/2/3、R0--R7 与 gate 配置，不扩大 canonical recipe 或公共 CLI
- [x] 10.6 禁止用与 trajectory validation 重叠的 clean-inner checkpoint 初始化，允许 trajectory sparse-CSI Stage 1 fresh start，并对后续 checkpoint 保持同协议校验

## 11. 五模态评估与机制诊断

- [x] 11.1 将 evaluator 泛化为默认 15/opt-in 31 个非空 mask，并约束 R0--R7 使用同一 split/seed/budget/expert fingerprint
- [x] 11.2 保存可复算样本级表：identity/group、label/prediction、每专家 probability/risk/weight/R_star/confidence/correctness、availability 与 CSI 质量字段
- [x] 11.3 实现 D0 原始 risk、D1 domain+mask 内 risk shuffle、D2 domain+mask 平均 risk、D3 static prior，以及独立的 risk/component/error/weight 诊断和按 group 的 paired bootstrap

## 12. 回归与 smoke

- [x] 12.1 使用 `conda run -n kd_mm_beam pytest` 覆盖固定 CSI selection/复数/SNR 缺失、五模态 shape/mask/risk/weight、31-subset 等频和 fresh-start 约束
- [x] 12.2 覆盖 time-order/split/resource 泄漏拒绝、禁止输入/噪声拒绝、四模态 checkpoint/forward 数值回归与 U0/CSI-TSPC 隔离
- [x] 12.3 完成 synthetic 与真实 MMW 单 batch Stage 1/2/3 smoke，记录 shape、loss、梯度、mask 计数、NaN/Inf 和 GPU peak memory
- [x] 12.4 在 trajectory 37,510/6,365 binding 上完成 fresh-start 五模态真实单 batch Stage 1/2/3 CUDA smoke，并核验同协议 checkpoint provenance

## 13. 主线源码收缩

- [x] 13.1 删除非 PCPF 本地实验 runner、配置、脚本、专用源码、文档和测试
- [x] 13.2 删除 Stage 3B、公式候选筛选和四转五 checkpoint 迁移
- [x] 13.3 删除未启用的 TensorBoard/Matplotlib、参数组 DSL 与冗余配置/目标包装层
- [x] 13.4 更新 current specs、README、维护导航并完成 focused、compile 与 strict OpenSpec 验证

## 14. trajectory 训练吞吐修复

- [x] 14.1 为 train/validation sparse CSI cache scan 增加协议和 SHA256 绑定的 packed 2x2 bundle，并让正式 dataset 严格内存命中
- [x] 14.2 在 resolver 中恢复 RGB/LiDAR/GPS 严格缓存绑定，将 sparse-CSI 默认 batch 调整为 64、workers 调整为 8
- [x] 14.3 增加 packed cache、配置绑定和 fail-closed 回归测试，完成 focused、compile 与 strict OpenSpec 验证
- [x] 14.4 重建 trajectory cache bundle，完成真实 CUDA batch 64 显存/吞吐 smoke，并在新目录 fresh start Stage 1
