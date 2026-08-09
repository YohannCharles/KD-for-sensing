## Why

旧 PCPF-T 将四模态 sensing、可选 sparse CSI、evidential risk 与解析动态融合拆成三个训练 stage。第二创新点现已确定为 TBCP-7 finite RF probing；继续保留第五模态和 Stage 2/3 会混淆两个创新点、污染配置与 checkpoint 语义，并使四模态消融不可解释。

## What Changes

- 以原生 `image/radar/gps/lidar` 单阶段模型替代旧分阶段预测/融合模型。
- 保留四个 encoder、共享 Temporal Transformer、唯一 Beam Prototype Bank、邻近 beam soft supervision 与 prototype alignment。
- 可用模态 probability 只做无参数 masked mean，直接得到 `p_sense[64]`；MAP、均值、方差、spread 与 entropy 均由该 posterior 无状态派生。
- 删除 sparse CSI 第五模态、evidential/risk head、static prior、temperature/tau、analytic fusion、Stage 2 gate、Stage 3、跨 stage checkpoint 初始化与续跑。
- 模型 evaluation 原生生成四模态全部 15 个非空 mask evidence，不再从五模态 31-mask cache 过滤 CSI-off rows。
- 保留 train-only topology likelihood、TBCP-7、matched baselines、batch schedule、robustness 与 covariance/预算诊断。
- topology `off/on` × seed `1/2/3` 必须在相同当前 protocol 上重新训练六个单阶段模型；旧五模态结果不得进入最终消融。
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
- 不访问 test，不启动训练；清理完成后再单独预检六条新训练线。
