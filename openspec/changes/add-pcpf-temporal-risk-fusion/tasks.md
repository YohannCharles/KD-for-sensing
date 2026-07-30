## 1. 边界与配置契约

- [x] 1.1 新增严格 PCPF-T model/loss 配置解析与 stage、维度、非负系数、禁止输入校验
- [x] 1.2 注册独立 `pcpf_temporal_risk_fusion`，确认 U0 构造与 state dict 不包含 PCPF owner
- [x] 1.3 新增 `tools/configs/pcpf/` 的 base、四阶段、A0--A4 和分量消融模板，不修改 canonical MMW recipe

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
- [x] 4.3 实现四个 stage 的精确冻结、frozen module eval、trainable name/count metadata 和断言
- [x] 4.4 接入共享 trainer 的 opt-in preparation、stage-specific loss 和 validation loss，保持 U0 默认路径不变
- [x] 4.5 扩展 checkpoint model metadata、source-stage fail-closed 校验和 PCPF validation-best stage alias

## 5. 评估与本地工具

- [x] 5.1 实现 Stage 2 gate 的相关性、分位、常数化、天气/domain/mask-group 报告和固定阈值判定
- [x] 5.2 扩展 15-mask 评估输出 weight/risk/calibration/temperature/claim provenance 诊断
- [x] 5.3 实现 `tools/run_pcpf.py` 的 resolve/preflight/synthetic/one-batch smoke 与 `tools/eval_pcpf.py` 的 gate/matrix 入口
- [x] 5.4 对所有 Stage 3 训练请求校验 gate JSON、SHA256、通过状态与 Stage 1 expert fingerprint

## 6. 测试与验证

- [x] 6.1 使用 `conda run -n kd_mm_beam pytest` 覆盖 temporal shape/mask、四项风险、概率融合、stage freezing 和数值稳定性
- [x] 6.2 使用 `conda run -n kd_mm_beam pytest` 覆盖配置拒绝、checkpoint 来源、gate、15-mask 聚合和 U0 兼容性
- [x] 6.3 使用 `conda run -n kd_mm_beam` 完成 synthetic 四阶段 forward/backward smoke 并记录 shape、loss、梯度、权重和 NaN/Inf
- [x] 6.4 使用 `conda run -n kd_mm_beam` 完成真实 MMW 单 batch Stage 1/2/3 smoke，不访问 outer test、不启动长训练
- [x] 6.5 运行 `openspec validate add-pcpf-temporal-risk-fusion --strict`、`make verify-quick` 与相关 focused tests，记录既有 active-change 风险

## 7. 主线收敛与历史归档

- [x] 7.1 为三个历史 active change 写入真实 closure note，并保存仓库外快照；不把未完成任务伪装为完成
- [x] 7.2 从当前工作树移除已停止且孤立的 probabilistic prototype owner
- [x] 7.3 保留并验证 U0、AMBER-Full、RMBP-MM、DeepSense6G、MMW、trajectory baseline、CSI/TSPC 与全部 cache 边界
- [x] 7.4 更新 current specs、README、agent navigation、maintainer context 和 architecture boundary，使 PCPF-T 成为唯一 active research mainline
