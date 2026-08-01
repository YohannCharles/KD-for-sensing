## Why

当前 MMW U0 通过自由 supervised Router 直接学习四模态权重，无法隔离验证“共享 beam prototype 语义下、样本级连续拓扑风险是否可被观测，以及低自由度解析融合是否优于静态先验”。PCPF-T 现确定为唯一 active research mainline；它与 U0 默认路径隔离，在默认只使用 train/validation、不改变 canonical recipe 的前提下完成三阶段训练、风险门控和同专家对照。新增实验进一步检验：在唯一 `mmw_id_stratified_block_v1` seed 0 manifest 上，只增加五帧历史窗口内固定 2x2 sparse CSI 时，PCPF-T 是否仍能在完整缺失子集上保持可审计的动态融合。

## What Changes

- 新增注册模型 `pcpf_temporal_risk_fusion`：复用四个当前 encoder 与唯一 `BeamPrototypeBank`，在风险估计前仅执行逐模态共享 Temporal Transformer，不进行跨模态 attention 或特征拼接。
- 新增 topology-aware 单模态监督、概率嵌入、四项解析风险、train-only 归一化/静态能力先验和 `a_m * exp(-risk_m / tau)` 概率级融合。
- 新增 `stage1_expert`、`stage2_risk`、`stage3_fusion` 的冻结、初始化 checkpoint、metadata、validation-best 与 fail-closed 契约。
- 新增本地研究配置、stage launcher、15-mask/天气/domain/校准/权重/风险诊断和 uniform、static prior、direct Router、CUAF-style `local_adaptation` 对照；不复制通用 trainer。
- 新增显式、失败关闭的本地三阶段续跑动作：等待已有 Stage 1 正常完成后，依次解析并运行 Stage 2、无界 validation gate 与 Stage 3；任一进程、checkpoint、协议或 gate 校验失败时立即停止。
- 新增 opt-in 历史 sparse CSI 第五专家：仅从样本自身 `csi1..csi5` channel 引用确定性生成固定 TSPC 2x2 pilot，保留复数信息，不加入 AWGN、dropout、corruption 或虚构 SNR；默认四模态构造、state dict 和数值路径不变。
- 新增五模态全部 31 个非空子集的等频训练 schedule、公平 R0--R7 矩阵、D0--D3 机制诊断与按独立分组 bootstrap。
- 新增 trajectory train/validation 数据画像工具：绑定唯一协议、split seed 与审计身份，只读 manifest 声明的开发样本，系统分析 split 代表性、跨天气同 scene/CAV/序号的轨迹内容重合、标签长尾、时序冗余、几何/beam-power 模糊度、四模态与 sparse-CSI 质量/漂移，并用固定预算 diagnostic probe 为后续改进排序提供证据。
- 将 MMW 划分统一为 `mmw_id_stratified_block_v1`：trajectory key 固定为 `(scene_id,cav_id)`，三天气通过 verified `seq_index` 基础时间映射绑定；每条轨迹先按 128 个基础时间点切 block，再以确定性标签平衡目标分配 70/15/15，最后只 materialize block 内完整窗口。5 个 scene 与全部 CAV 在三个 split 中均覆盖，trajectory overlap 是设计目标，block/base/weather/window-frame overlap 必须为零。
- canonical manifest 固定为 `splits/mmw_id_stratified_block_v1/seed_0.json`，报告固定写入 `outputs/split_reports/`；旧 `mmw_trajectory_disjoint`、11/2/3、11/5/0、clean-inner、80/10/10、group-safe/window split 与其 split-specific cache/checkpoint 不得继续作为当前运行输入。
- sparse-CSI 正式开发路线固定绑定当前 seed manifest 的实际 train/validation windows并复用 train-only GPS scaler；原始内容寻址 CSI cache可保留，但 split-specific index/bundle 必须按新 manifest 重建。默认不加载 test，只有显式 `--evaluate-test` 才允许只读最终评估。
- 全部当前开发结果标记 `claim_ineligible: true`；开发选择只允许绑定且隔离的 train/validation。除 opt-in 历史 sparse CSI 外，禁止 test、当前/未来 CSI、未来 channel、path/beam-power、历史 beam、天气/场景标签进入模型或风险目标。
- 保留 U0、AMBER-Full、RMBP-MM 与 DeepSense6G 稳定 recipe；未声明 PCPF-T 时不创建任何 PCPF 参数或状态。
- 删除非 PCPF 本地实验 runner、配置、专用模型、诊断和测试；只保留 clean/trajectory 数据协议、PCPF sparse-CSI 所需 channel 原语及全部本地 cache。

## Capabilities

### New Capabilities

- `pcpf-temporal-risk-fusion`: 定义 PCPF-T 的 temporal expert、共享 prototype、拓扑风险、解析融合、三阶段训练、门控评估及诊断契约。

### Modified Capabilities

- `u0-mainline`: 允许隔离的 PCPF-T 注册模型和本地研究 workflow，同时继续禁止其成为 canonical U0 替代或改变 U0 数值路径。
- `clean-data-integrity`: 将 MMW 收敛为唯一 ID-stratified block protocol，默认只构建 train/validation、显式授权 test，并保护所有 train-only 拟合状态。
- `mmw-id-stratified-block-protocol`: 定义 verified weather binding、连续 block、标签平衡 70/15/15、manifest/cache/report/leakage 契约，并为 PCPF-T `use_sparse_csi=true` 约束同样本历史 sidecar。
- `repo-boundaries`: 允许 PCPF-T 的窄模型/loss owner、本地工具配置和 launcher，同时不增加公共 CLI、canonical MMW recipe 或受跟踪运行产物。

## Impact

- 模型与目标：`src/kd_sensing/models/`、`src/kd_sensing/losses/`、模型注册和训练 extension 选择。
- 通用训练契约：只增加 opt-in stage preparation/checkpoint metadata 能力，U0 和保留 baseline 未启用时路径不变。
- 本地研究面：`tools/`、`tools/configs/pcpf/`、`scripts/`、focused tests；resolved config、报告、checkpoint 和 smoke 产物仅写入 `outputs/`。
- 数据与评估：默认复用 canonical 四模态顺序、5 帧 temporal mask 与 15-mask metrics；opt-in 路线在 PCPF 本地 sidecar 中追加固定历史 sparse CSI、五模态 31-mask metrics 和样本级诊断字段，不改变全局 modality registry、MMW canonical recipe 或公共 CLI。
- 数据画像：本地工具与全部 CSV/JSON/Markdown/图表/特征 cache 只写入 ignored `outputs/`，不新增 public CLI；开发画像默认只读 train/validation，不把 test、future beam power、geometry、天气或场景诊断字段提升为模型输入。
- 仓库治理：`openspec/changes/` 只保留本 change；历史 change 使用仓库外快照追溯，不建立仓库内 `archive/`。
