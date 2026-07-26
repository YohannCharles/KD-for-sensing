## Why

Clean MMW U0 在 Image、LiDAR 等强模态缺失时仍可能出现决策边界退化，但现有主线没有隔离验证“缺失模式条件的轻量决策残差”是否能在不改变 Full 路径和冻结 U0 的前提下提高保持率。需要以受审计的 Seed 1 筛选实验将固定偏置、mask 低秩、原型条件低秩及打乱原型负对照依次比较。

## What Changes

- 为冻结的 U0 增加只读 Beam prototype 状态导出，复用既有 prototype logits，不更新 prototype 或影响现有 forward 数值。
- 增加只在非 Full mask 激活的低秩决策 Adapter，以及 mask bias、mask-only、prototype-conditioned、uncertainty-conditioned 与 shuffled-prototype 对照实现。
- 增加严格绑定 clean inner protocol 的 Adapter 训练、15-mask 评估、Full 逐样本等价检查、逐样本预测导出和独立指标/配对统计工具。
- 增加 GPU 0--7 Seed 1 Stage A 启动脚本与条件生成但不执行的 Stage B 多 seed 脚本；它们不成为新的公共 CLI 或 canonical recipe。
- 增加一次性 Full-pool capacity verification：从 Town3 的 46,860 个候选窗口构建共享时间轴的 80/20 连续开发划分，保留 588 个历史验证身份排除，训练唯一 Full-data U0，并在用户扩展后完整比较 A0--A7；A1/A4/A6 仍是预注册核心科学问题。
- 增加 GPU 1--7 两阶段编排、动态测时与 epoch 预算、多核 train-only 数据构建、10 分钟运行审计、完整 15-mask 逐样本导出和独立重算；该工作流仍是本地实验工具，不扩展 public CLI 或 canonical recipe。
- 复用已覆盖全部候选帧的只读 ImageNet RGB 与 LiDAR BEV 派生缓存，持久化无标签 GPS 坐标特征和 train-only scaler；只有实测小文件 I/O 仍限制吞吐时才构建按 domain 分片、按唯一帧存储的新 LMDB。
- 按用户最新资源约束将正式 Full-pool Stage 2 运行在物理 GPU0/4/6，并在 GPU7 空闲后将尚未启动的 A7 动态迁移到 GPU7、在 A5 完成释放 GPU6 后将尚未启动的 A6 从 GPU4 队列迁移到 GPU6；U0 与 A1--A7 启用不读取 validation 的预注册训练损失早停，20 epoch 保持为上限而非强制执行量。
- 在 A0--A7 完成且 Top-1 未显示总体 Adapter 收益后，增加不覆盖原结果的 ADBA-surrogate B1/B4/B6/B7 对照；它复用同一 U0、Full-pool protocol、mask schedule 与早停规则，仅将 Adapter 分类损失预注册为 hard CE 与 circular soft-label CE 的等权组合，并以 B7 检验 B6 的 prototype 语义。
- 在 B1 成为 ADBA-surrogate 最强方法后，增加单 seed 的最小 novelty triage：复用 B1，对比 Global Bias、Mask Lookup 与 Mask MLP；只有 MLP 的 All-14 ADBA 超过 Lookup 时，才运行一个确定性分层 unseen-mask pilot，并以可组合的 Factorized Additive Bias 作为未见 mask 对照。
- 在 Mask MLP 显示收益但 Factorized Bias 已解释其大部分增益后，增加一个不产生任意 logits 残差的环形 Beam 分布传输对照：每个缺失模态学习半径为 3 的局部概率核，多个缺失模态按固定模态顺序作环形卷积组合，Full 显式恒等。该方法与 all-seen Factorized Bias 共享 8 epoch、原 mask schedule 和 ADBA-surrogate，并保持独立本地产物根。
- 增加独立的 Full-Pool BT-SCL 本地实验：从头训练统一的轻量 CNN/MLP、时间-模态 token Fusion MLP 与 64-beam prototype bank，在相同初始化、Full-pool 划分和 nested subset chain 下比较 R0--R5。该工作流不调用 U0、Router、attention、Adapter、teacher 或 logits residual 路线。
- 将经 15 个 Town3 domain 回放验证的 ULA-DFT phase-cycle topology manifest 作为 R2--R5 的强制输入，使用它构造 beam distance、连续 sector 和局部邻域；仅有 `circular=true` 的配置不得启动拓扑损失。
- 在 R0--R5 未通过预注册门槛后增加单一 R6 修复性筛选：以真实 Beam 标签锚定 S1/S2/S3 的 4/8/16-sector 监督，并对嵌套 subset 在半径 0/3/5 内的真实标签邻域概率施加随机占优约束；R6 不再使用 Full-teacher KL 或随机初始化反比例 loss 校准，且不新增推理参数。
- 在首条 R6 轨迹复现 epoch-1 后 validation selection loss 持续上升后，保留该失稳轨迹并追加明确标记为 post-hoc 的稳定训练筛选；稳定版从同一初始化并行重跑 R0、固定小权重 R3 与 R6，统一降低学习率、提高 weight decay 并使用相同 validation patience，不将其冒充原预注册结果。
- 增加独立的 Full-Pool Candidate12 快速筛选：在禁止 current U0 sample-specific router 的前提下，复用其四模态 encoder 配置、`BeamPrototypeBank`、BPA 权重与审计 topology，以唯一 5-epoch warm-up 比较 KL Data Remixing、共享 prototype-risk assignment、同模态 batch 重组、非环形角度顺序上的 prototype-anchored motion refinement 及组合路线；六路只研究 Full 四模态主结果，不新增 public CLI 或 canonical recipe。
- 在 BTMA B0--B5 判定不进入 multi-seed 后，增加一次只读负结果收尾：从已保存的六个 BTMA checkpoint 重算逐样本 validation 预测，按 `(domain, cav)` 内连续帧块做成对 temporal block bootstrap，并用既有 warm-up train cache 与 assignment CSV 计算 assignment score 与单模态拓扑误差的相关性及跨 epoch 秩稳定性。该收尾不重训、不调参、不启动 multi-seed 或 outer test，其结果不得用于重开 BTMA 路线。
- 增加冻结 Full-pool U0 上的 Router 可观测性因果筛选：一次性缓存与 mask 无关的 `latent_sequence` 与各 encoder 末层线性变换的输入特征，然后在完全相同的冻结表征上只重训 router，比较仅现有标量、加 prototype-space 状态、加投影前 quality 分支及等参数量置换负对照四条嵌套路线；同时在自然天气与注入分级腐蚀两个退化设定下运行三个 router seed，并对处理组执行冻结权重的推理期消融。该筛选只回答 routing 输入问题，不训练 encoder，不修改 U0 canonical recipe 或 public CLI。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `u0-mainline`: 定义冻结 U0 上仅对缺失条件生效的实验性决策 Adapter 合约，以及 Full 前向严格旁路。
- `clean-data-integrity`: 定义该实验只可访问经过审计的 inner_train 与 inner_validation，且所有可拟合 Adapter 状态仅来自 inner_train。
- `repo-boundaries`: 明确实验运行产物、checkpoint、逐样本预测与统计报告保持在本地产物边界，实验脚本不扩展公共 CLI。

## Impact

影响 `src/kd_sensing/models/` 中 U0 的只读诊断接口和新增 Adapter 模块，新增受协议约束的实验训练/评估模块、Full-pool 数据协议构造器、脚本、配置及针对冻结、掩码、等价性和统计隔离的测试。不会修改 U0 canonical recipe、编码器、融合、prototype、分类器或公共 CLI。
