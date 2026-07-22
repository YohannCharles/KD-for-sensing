## Context

已完成的 feature-fusion quick search 在同一冻结缓存上提供 F1 concat MLP 与 F4 prototype query checkpoint。F4 的 query 来自 prototype，但 decoder 后由共享 `SharedScore(h_k)` 输出 logit，最终 score 不再显式与第 k 个 prototype 对齐。本轮仅在该缓存上进行单 seed、inner train/validation 搜索，且所有输出均为本地、claim-ineligible 证据。

## Goals / Non-Goals

**Goals:**

- 验证 prototype-compatible score 与 query Gram 保持能否改善 F4 的 beam topology 指标。
- 验证冻结 F1 全局输出加受限 prototype-query local 修正，能否在不损害总体指标下改善 Missing LiDAR。
- 用固定 checkpoint 选择、固定训练 mask、统一 inner validation 和可审计诊断公平比较 G0--G5。

**Non-Goals:**

- 不重新计算 TinyViT encoder 特征，不解冻 encoder 或 prototype bank。
- 不使用 channel、CSI、gain、beam power、path、历史 beam、动态 Router、quality/evidence-reconstruction、MoE、样本级 reliability 或样本级 beta。
- 不增加公共 CLI、canonical recipe、registry 或依赖；不运行 outer test、多 seed 或端到端训练。

## Decisions

### 1. 保留已有 cache 和 F1/F4 checkpoint，先 fail-closed 复现

G0/G1 只加载既有 F1/F4 best checkpoint，并使用其 cache manifest、split sample ID、mask 协议和 prototype SHA。新的准备步骤先检查这些身份、禁止字段、split 不重叠及 cache dtype/shape，再用同一内层评测重现旧结果；Top1 的容差为 0.02 pp。重训 G0/G1 会引入无关随机差异，故不作为默认路径。

### 2. G2/G3 以 prototype 同时定义 query 与最终 score

原型 buffer `P` 始终 detached；共享可训练投影产生 `Q=normalize(W_q(normalize(P)))`。既有两层 pre-norm cross-attention 保持不变，但其最后层 attended evidence 经 `LayerNorm` 与 `W_o` 得到 `E=normalize(W_o(LayerNorm(C)))`，并以逐 beam `dot(E_k,Q_k)/tau` 输出 logit。`tau` 以 bounded softplus 参数化。不会把 `Q` 残差加入 `E`，因此零 token 无法从 query 自身产生样本特异预测。

这复用既有 token adapter 和 decoder，避免为 64 个 beam 新增独立头。原 shared scalar score 保留给 G1 复现，不能作为 G2--G5 的最终输出。

### 3. G3 用非对角 query Gram 约束而不修改 prototype

G3 与 G2 架构相同，只追加 `mean((Q Q^T - stopgrad(P_norm P_norm^T))^2)` 的非对角元素。固定的前 2--3 个 train batch 只校准一次 topology、Gram 和 preserve lambda；验证结果不参与回调。选择 Gram 而非手工近/远 pair 采样，是因为 64-beam 全矩阵已完整表达当前 topology，且改动更小。

### 4. G4/G5 冻结 F1 作全局锚点，local branch 只作静态受限修正

G4 的 F1 token adapter、concat MLP、output projection 和 prototype head 从已验证 checkpoint 加载并完全冻结。训练一个独立的 G2 local branch，输出经 beam 维中心化后与 `z_global` 相加：`z_final=z_global+beta*z_local_centered`。`beta=beta_max*sigmoid(raw_beta)` 是单个全局参数，初始化约 0.075，默认 `beta_max=1`。

G5 使用相同结构，增加 G3 Gram 和固定六个 mask group 的平均 CE/topology loss；只对 Full group 加入 teacher 为 detached F1 的 KL preserve loss。它不改变样本、batch order 或 mask 生成，只改变同一步已生成组的固定等权聚合。

### 5. 单一 local/manual runner 管理训练和诊断

新的分析入口从 feature-fusion runner 的 cache loader、mask、topology loss、指标和报告 helper 复用实现。它只接受预定义 G0--G5 config 名称，在 `outputs/topology_anchored_query_search/` 写入 resolved config、status、checkpoint、指标和诊断。GPU launcher 固定 GPU0--5，记录 PID/GPU 状态，单项失败不终止其余项，也不自动重跑或发起任何后续实验。

## Risks / Trade-offs

- [Risk] 旧 checkpoint 或 cache 身份漂移会把比较变成不同输入比较。→ 准备阶段校验 SHA、sample ID、mask、prototype 和禁止字段，失败即停止。
- [Risk] anchored branch 可能忽略 value 或退化为静态偏置。→ 增加 zero-token、token shuffle、masked-value、attention leakage 和 G4/G5 mean/zero/shuffled local 替换测试。
- [Risk] 双分支仅靠容量获益。→ 报告 total、frozen、trainable、active inference 参数和相对 F1 延迟/FLOPs，并以 beta 与 local/global norm 诊断覆盖程度。
- [Risk] G5 改变实际数据分布。→ 固定同一 batch/mask 样本，只对既有组损失按相同权重求均值。
