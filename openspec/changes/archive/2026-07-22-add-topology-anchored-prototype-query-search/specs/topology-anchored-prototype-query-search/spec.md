## ADDED Requirements

### Requirement: G0/G1 必须在同一冻结身份上复现既有 F1/F4

系统 MUST 从 feature-fusion cache manifest 读取 inner train、inner validation、冻结 prototype、F1 best checkpoint 与 F4 best checkpoint，并在任何 G2--G5 训练前校验 checkpoint/prototype SHA、sample ID、split 不重叠、token/mask 协议、shape/dtype 和禁止字段。G0 与 G1 MUST 只加载对应既有 best checkpoint，不得重新选择 checkpoint；统一内层评测的 Top1 与已记录结果差异 MUST 不超过 0.02 pp，否则系统 MUST 停止本轮训练。

#### Scenario: 复现身份不一致
- **WHEN** cache、split、mask、prototype 或 checkpoint 的身份与基准 manifest 不一致
- **THEN** 系统 MUST 写出具体不一致字段并以失败状态结束准备步骤
- **AND** GPU0--5 launcher MUST 不启动 G2--G5

### Requirement: Anchored prototype-query score 必须由 attended evidence 与对应 query 配对

G2--G5 的 prototype bank MUST 是无梯度冻结 buffer。系统 MUST 使用共享投影构建 `Q=normalize(W_q(normalize(P.detach())))`，保留 64 个 query、既有 token adapter、time/modality embedding、两层 pre-norm cross-attention、4 heads、FFN 512、dropout 0.1 和 key padding mask。最终 local score MUST 为 `dot(E_k,Q_k)/tau`，其中 `E=normalize(W_o(LayerNorm(C)))` 且 `C` 来自 attended value；`tau` MUST 保持为有限正值。

系统 MUST NOT 为 beam 创建 64 个独立分类头，MUST NOT 将 query 残差直接加入 evidence，且 MUST NOT 以共享 scalar score 作为 G2--G5 的最终 score。

#### Scenario: Anchored score 的输入与 mask 安全
- **WHEN** 使用 synthetic batch 对 anchored branch 前向和反向传播
- **THEN** `Q` MUST 为 `[64,d_model]`、`E` MUST 为 `[B,64,d_model]`、logits MUST 为 `[B,64]`，prototype MUST 无梯度且 tau 为正
- **AND** zero-token 不得产生样本特异输出，打乱可用 token 必须改变输出，被 mask token 改写不得改变输出，missing attention leakage MUST 在数值容差内为零

### Requirement: G3 必须只以非对角 Gram loss 保持 query topology

G3 MUST 与 G2 保持相同的模块和超参数，只增加 `L_gram=mean((offdiag(Q Q^T)-stopgrad(offdiag(P_norm P_norm^T)))^2)`。对角元素 MUST 不参与该损失，prototype bank MUST 不接收梯度，且 loss MUST 仅通过 query projection 反向传播。`lambda_gram` MUST 只使用固定 2--3 个 train batch 校准一次并保存记录。

#### Scenario: G2/G3 结构公平
- **WHEN** 构建同一随机初始化下的 G2 和 G3
- **THEN** 两者的 state-dict key、参数量、forward 输出 contract 和基础训练损失 MUST 相同
- **AND** G3 的差异 MUST 仅为有限的非对角 Gram loss

### Requirement: Global-local hybrid 必须冻结 F1 并限制静态 local 修正

G4/G5 MUST 从已验证 F1 checkpoint 加载 global branch 并冻结其 token adapter、concat MLP、output projection 和 prototype head。local branch MUST 使用 anchored prototype-query score；系统 MUST 在 beam 维中心化 local logits，并以单个静态 `beta=beta_max*sigmoid(raw_beta)` 生成 `z_global+beta*z_local_centered`。beta MUST 位于 `[0,beta_max]`，初始化在 0.05--0.1，且不得依赖 sample、标签、teacher 错误标记或模态 reliability。

G5 MUST 在 full、missing image、missing lidar、missing radar、missing gps 与原 S3 group 上以固定相同权重聚合主损失，并且只对 Full group 加入 detached F1 的 KL preserve loss；它 MUST 不使用动态 GroupDRO、auxiliary unimodal head、Router 或 sample-specific beta。

#### Scenario: Hybrid 边界与替换诊断
- **WHEN** 设置 beta 为零或替换 local logits
- **THEN** beta 为零时 hybrid logit MUST 逐元素等于 F1，local logits MUST 正确中心化，且 F1 参数保持无梯度
- **AND** 评测 MUST 分别报告 sample local、train-mean local、zero local 和跨 sample shuffled local，不能以静态偏置替代样本级 local evidence

### Requirement: 本地 quick search 必须统一报告、隔离运行并停止

系统 MUST 在同一 inner validation 协议上报告 Full、四个单模态缺失、S3 macro/worst、Missing Avg、Worst、Top3、Top5、Within-3、MAE、weather、sector、exact/1--3/4--5/>5 error distance、query/score/attention topology、模态 missing 与 shuffle dependence、global-local 替换、参数、显存、速度和 success gates。G4/G5 MUST 额外报告 beta、local/global norm、修正强度和双分支 total/frozen/trainable/active inference 参数。

GPU launcher MUST 固定 G0--G5 至 GPU0--5，保存 `nvidia-smi`、PID、resolved config、日志和失败原因；单项失败 MUST NOT 终止其他项。所有结果 MUST 标记 single-seed、inner-only、claim-ineligible，且系统 MUST NOT 自动运行 outer test、多 seed、端到端训练或修改正式 claim。

#### Scenario: 生成受控筛选报告
- **WHEN** launcher 完成或任一任务失败后执行汇总
- **THEN** 系统 MUST 将每个方向标记为 complete 或 failed，并生成可用方向的主表、拓扑表、依赖表、替换表、success gate 与唯一推荐方向
- **AND** 工作流 MUST 在该报告后停止
