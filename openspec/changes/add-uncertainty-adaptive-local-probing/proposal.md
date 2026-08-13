## Why

旧 PCPF-T 将四模态 sensing、可选 sparse CSI、evidential risk 与解析动态融合拆成三个训练 stage。当前四模态主线将第二创新点固定为低开销 TBCP-3（2+1）finite RF probing；TBCP-5/7/9 仅作为预注册预算敏感性。继续保留第五模态和 Stage 2/3 会混淆两个创新点、污染配置与 checkpoint 语义，并使四模态消融不可解释。

## What Changes

- 以原生 `image/radar/gps/lidar` 单阶段模型替代旧分阶段预测/融合模型。
- 保留四个 encoder、共享 Temporal Transformer、唯一 Beam Prototype Bank、邻近 beam soft supervision 与 prototype alignment。
- arithmetic-mean、无界/有界四标量 reliability 只保留为诊断对照；新增一个标准 `masked_feature_mlp` backbone 分支，在分类前联合四模态特征与 availability mask，不输出显式模态权重，也不作为论文创新点。
- 四个单模态特征和融合特征必须查询同一个64-beam Prototype Bank：单模态由 availability-normalized hard/neighbor-soft supervision聚类，融合特征由 fused hard 与 topology prototype supervision聚类；不复制第二套原型或重复叠加同形态 modality-prototype loss。
- 将 masked-feature 主线中重复的单模态soft项与融合prototype项收敛为唯一 `joint_topology_weight`：对融合特征和可用单模态特征的同一环形soft目标取等权平均，再只乘一次总权重；旧分项在该主线显式置零。
- 删除 sparse CSI 第五模态、evidential/risk head、static prior、temperature/tau、analytic fusion、Stage 2 gate、Stage 3、跨 stage checkpoint 初始化与续跑。
- 模型 evaluation 原生生成四模态全部 15 个非空 mask evidence，不再从五模态 31-mask cache 过滤 CSI-off rows。
- 保留 train-only topology likelihood、TBCP-3（2+1）、matched baselines、batch schedule、robustness 与 covariance/预算诊断；TBCP-5/7/9 仅用于预算曲线。
- 增加公平 sensing-only 对照：topology soft-only、prototype-only、普通 uniform label smoothing，以及仓库保留的 AMBER-Full-local/RMBP-MM-local；所有方法统一四模态、五帧、15-mask，主表不使用历史 beam index 或当前 beam-power。
- topology `off/on` × seed `1/2/3` 必须在相同当前 protocol 上重新训练六个单阶段模型；旧五模态结果不得进入最终消融。
- 主报告必须同时披露直接 sensing Top-1、Posterior Top-3 覆盖率和 probing 后 selected-beam Top-1，禁止把额外 RF 测量后的结果表述为纯模型 Top-1。
- 增加 DeepSense6G Scene31–34 secondary transfer panel：Prototype-only 使用数据集适配的线性 label-index 邻接，与 AMBER-Full-local/RMBP-MM-local 在固定40 epoch/last checkpoint下比较；该面板不复用 MMW ULA audit 或 TBCP likelihood。
- masked-feature fusion 必须先运行 topology-on seed1 fresh pilot，并与 arithmetic-mean、无界/有界诊断及 sensing-only baselines 使用相同 whole-modality schedule、预算和15-mask/TBCP-3协议；只有pilot有效才进入 topology off/on 三seedmatched panel。
- 不提供旧模型名、配置字段、CLI action 或 checkpoint 的兼容层。

## Capabilities

### New Capabilities

- `four-modal-topology-predictor`: 四模态单阶段 topology-prototype beam posterior predictor。

### Modified Capabilities

- `sensing-guided-local-beam-probing`: 改为消费原生四模态 15-mask evidence。
- `clean-data-integrity`: 删除 sensing 模型的 sparse-CSI 与 stage-fitted risk/fusion 状态，保留 train-only probing likelihood 隔离。

### Removed Capabilities

- `pcpf-temporal-risk-fusion`: 整体退出 active surface。

## Impact

- 模型、loss、registry、trainer/config validation、MMW batch/dataset sidecar。
- 本地单阶段 resolve/preflight/train 工具与 native 15-mask evidence/probing evaluator。
- OpenSpec、README、agent navigation、focused tests 与旧 ignored outputs。
- MMW 新增训练和 replay 仍只读取 train/validation且test封存。DeepSense6G secondary panel没有兼容的五帧validation，固定训练40 epoch后只执行一次官方test评测；新增产物使用独立 ignored output 根目录，不覆盖既有 checkpoint。
