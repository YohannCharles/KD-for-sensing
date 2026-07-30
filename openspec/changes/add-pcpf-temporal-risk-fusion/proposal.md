## Why

当前 Clean MMW U0 通过自由 supervised Router 直接学习四模态权重，无法隔离验证“共享 beam prototype 语义下、样本级连续拓扑风险是否可被观测，以及低自由度解析融合是否优于静态先验”。PCPF-T 现确定为唯一 active research mainline；它与 U0 默认路径隔离，在不访问 outer test、不改变 canonical recipe 的前提下完成三阶段训练、风险门控和同专家对照。新增实验进一步检验：在 `mmw_trajectory_disjoint_v1` 的 37,510/6,365 trajectory-disjoint train/validation 上，只增加五帧历史窗口内固定 2x2 sparse CSI 时，PCPF-T 是否仍能在完整缺失子集上保持可审计的动态融合。

## What Changes

- 新增注册模型 `pcpf_temporal_risk_fusion`：复用四个当前 encoder 与唯一 `BeamPrototypeBank`，在风险估计前仅执行逐模态共享 Temporal Transformer，不进行跨模态 attention 或特征拼接。
- 新增 topology-aware 单模态监督、概率嵌入、四项解析风险、train-only 归一化/静态能力先验和 `a_m * exp(-risk_m / tau)` 概率级融合。
- 新增 `stage1_expert`、`stage2_risk`、`stage3_fusion` 的冻结、初始化 checkpoint、metadata、validation-best 与 fail-closed 契约。
- 新增本地研究配置、stage launcher、15-mask/天气/domain/校准/权重/风险诊断和 uniform、static prior、direct Router、CUAF-style `local_adaptation` 对照；不复制通用 trainer。
- 新增 opt-in 历史 sparse CSI 第五专家：仅从样本自身 `csi1..csi5` channel 引用确定性生成固定 TSPC 2x2 pilot，保留复数信息，不加入 AWGN、dropout、corruption 或虚构 SNR；默认四模态构造、state dict 和数值路径不变。
- 新增五模态全部 31 个非空子集的等频训练 schedule、公平 R0--R7 矩阵、D0--D3 机制诊断与按独立分组 bootstrap。
- sparse-CSI 正式开发路线固定绑定 `mmw_trajectory_disjoint_v1` 的 37,510 train 与 6,365 validation window，复用其 train-only GPS scaler；outer test 2,985 window 继续封存。旧 clean-inner 3,600/900 checkpoint 与结果仅保留为背景材料，不得初始化 trajectory 模型或进入公平矩阵。
- 全部当前开发结果标记 `claim_ineligible: true`；只允许绑定且隔离的 train/validation 与显式历史 development evaluation。除 opt-in 历史 sparse CSI 外，禁止 outer test、当前/未来 CSI、未来 channel、path/beam-power、历史 beam、天气/场景标签进入模型或风险目标。
- 保留 U0、AMBER-Full、RMBP-MM 与 DeepSense6G 稳定 recipe；未声明 PCPF-T 时不创建任何 PCPF 参数或状态。
- 删除非 PCPF 本地实验 runner、配置、专用模型、诊断和测试；只保留 clean/trajectory 数据协议、PCPF sparse-CSI 所需 channel 原语及全部本地 cache。

## Capabilities

### New Capabilities

- `pcpf-temporal-risk-fusion`: 定义 PCPF-T 的 temporal expert、共享 prototype、拓扑风险、解析融合、三阶段训练、门控评估及诊断契约。

### Modified Capabilities

- `u0-mainline`: 允许隔离的 PCPF-T 注册模型和本地研究 workflow，同时继续禁止其成为 canonical U0 替代或改变 U0 数值路径。
- `clean-data-integrity`: 允许 PCPF-T 对明确标记的 historical development split 做只读、claim-ineligible 评估，并继续封存 outer test 与所有 train-only 拟合状态。
- `repo-boundaries`: 允许 PCPF-T 的窄模型/loss owner、本地工具配置和 launcher，同时不增加公共 CLI、canonical MMW recipe 或受跟踪运行产物。

## Impact

- 模型与目标：`src/kd_sensing/models/`、`src/kd_sensing/losses/`、模型注册和训练 extension 选择。
- 通用训练契约：只增加 opt-in stage preparation/checkpoint metadata 能力，U0 和保留 baseline 未启用时路径不变。
- 本地研究面：`tools/`、`tools/configs/pcpf/`、`scripts/`、focused tests；resolved config、报告、checkpoint 和 smoke 产物仅写入 `outputs/`。
- 数据与评估：默认复用 canonical 四模态顺序、5 帧 temporal mask 与 15-mask metrics；opt-in 路线在 PCPF 本地 sidecar 中追加固定历史 sparse CSI、五模态 31-mask metrics 和样本级诊断字段，不改变全局 modality registry、MMW canonical recipe 或公共 CLI。
- 仓库治理：`openspec/changes/` 只保留本 change；历史 change 使用仓库外快照追溯，不建立仓库内 `archive/`。
