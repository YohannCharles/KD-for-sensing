# MMW ID Stratified Block Protocol Specification

## Purpose

定义 MMW 唯一的同分布、天气绑定、连续基础时间块、标签平衡 70/15/15 划分，以及 manifest、缓存、报告和显式 test 访问契约。

## Requirements

### Requirement: MMW 必须使用唯一稳定基础样本身份

系统 MUST 只接受 protocol `mmw_id_stratified_block_v1`、protocol version `1`。trajectory key MUST 为 `(scene_id,cav_id)`；base sample key MUST 为 `(scene_id,cav_id,base_frame_index)`。`sensor_scenario`、`agent`、`condition` MUST 分别映射为 `scene_id`、`cav_id`、`weather`，strict sequence index 的 `seq_index` 与显式历史/未来 frame 列表 MUST 用于建立跨天气 `base_frame_index`。物理 frame id 不同的 sunny、rainy、foggy 副本 MUST 映射到同一 base sample；长度、序号、标签或窗口结构无法精确对齐时 MUST 失败，不得按文件排序猜测或静默退化。

#### Scenario: 三天气使用不同物理 frame id

- **WHEN** 同一 scene/CAV 的三天气 frame id 范围不同但 strict `seq_index` 与窗口 frame 列表一致
- **THEN** 系统 MUST 以验证后的局部基础时间序号绑定天气副本
- **AND** 任一对应关系冲突 MUST 在划分前失败

### Requirement: 连续基础时间 block 必须先于最终窗口 materialization

系统 MUST 按每条 `(scene_id,cav_id)` 的 `base_frame_index` 升序切成不重叠连续 block；默认 `block_size=32`，尾部不足 32 个基础时间点时保留为不可拆分 block。系统 MUST 在 block assignment 完成后，只 materialize 历史与目标全部位于同一 block 的窗口；跨 block 候选窗口 MUST 丢弃，不得 padding、复制邻接 split 数据或跨 block 取帧。`block_size` MUST 按基础时间点计数，而非天气展开样本或窗口数。

#### Scenario: 窗口跨越 block 边界

- **WHEN** 候选窗口任一历史或未来 frame 的 base index 不属于同一个 block
- **THEN** 该窗口 MUST 不进入任何 split
- **AND** manifest/report MUST 记录边界丢弃数量

### Requirement: block assignment 必须满足 70/15/15 与标签平衡

系统 MUST 以完整 block 为最小单位分配 train、validation、test，目标比例固定为 `0.70/0.15/0.15`。分配 MUST 使用只由 `split_seed` 控制的局部 RNG，并优化总基础样本或预计窗口比例、各 split 与全量 beam 分布差异、每 trajectory 比例和每 scene 比例。标签目标 MUST 同时包含全局、按 scene/domain 及按 `(scene_id,cav_id)` trajectory 的 train--validation 与 train--test TV，并惩罚 validation/test beam 质量在对应 scene 或 trajectory 的 train 中缺失；不得只优化全局 histogram，也不得只用 Pearson correlation。每条 trajectory 和每个 scene 在三个 split 中 MUST 均有 block；train MUST 为最大 split，validation/test MUST 非空。同 seed、同数据与任意输入遍历顺序 MUST 得到完全相同 manifest。

#### Scenario: 纯连续 70/15/15 标签偏移更大

- **WHEN** block-level 优化完成
- **THEN** report MUST 同时计算简单前段 70/15/15 block baseline
- **AND** MUST 报告两者的 TV/JSD/correlation，不得为改善分布拆散 block

#### Scenario: trajectory block 不足

- **WHEN** 任一 trajectory 少于三个 block，无法为三个 split 各提供一个 block
- **THEN** 构建 MUST 报告具体 scene/CAV 与 block 数并失败
- **AND** MUST NOT 回退到 trajectory held-out、窗口随机或逐样本分层

#### Scenario: 全局标签接近但 domain 内标签错位

- **WHEN** 一个 assignment 的全局 train--validation TV 较低，但任一 scene/domain 或 trajectory 的条件 TV、held-out 未覆盖 beam 质量更差
- **THEN** assignment objective MUST 披露并惩罚该条件失配
- **AND** report MUST 分别输出 scene/domain 与 trajectory 的 macro、worst 和未覆盖质量，不得用全局指标掩盖

### Requirement: manifest 必须完整绑定来源、窗口和 split 身份

canonical manifest MUST 位于 `splits/mmw_id_stratified_block_v1/seed_<N>.json`，使用 manifest schema version `2`，并记录 dataset、protocol/version、assignment algorithm、split seed、70/15/15 ratios、block size、trajectory/base key、weather binding、data source hash、window config/hash、三个 split 的 block、每 block 的范围、基础样本数、天气数、beam histogram、预计/实际窗口数与统计。已存在 manifest 只有在 manifest/protocol/assignment version、seed、block size、source hash、window hash、weather mapping、split CSV hash 和全部 leakage invariant 一致时才能复用；不一致 MUST 明确要求显式 regenerate，禁止静默覆盖。

#### Scenario: 旧 manifest 或 cache 被请求

- **WHEN** 输入来自 `mmw_trajectory_disjoint`、clean-inner、group-safe、随机窗口、旧 ratio 或缺少新身份字段
- **THEN** loader MUST 在 dataset、optimizer 或 checkpoint 创建前拒绝
- **AND** 不得提供兼容 fallback

### Requirement: 所有 split-dependent cache 必须携带新协议身份

MMW token、CSI split index、sample/window index、modality feature、prototype、contrastive queue、GPS scaler、normalization 与 label-frequency cache MUST 记录 `split_protocol`、`protocol_version`、`split_seed`、`block_size`、`split_manifest_hash`、`data_source_hash`、`window_config_hash` 和 `weather_binding`。缺少或不匹配任一字段 MUST 失效。按原始 sample ID 构建且与 split 无关的 CSI 内容 cache MAY 保留，但 split-specific index/bundle MUST 重建并验证 sample identity。

#### Scenario: 加载旧 normalization 或 sparse-CSI bundle

- **WHEN** cache 没有完整 block protocol identity 或 sample coverage 与 manifest 不一致
- **THEN** 系统 MUST 拒绝该 cache 并要求重建 split-specific artifact

### Requirement: loader 前必须统一验证全部泄漏与覆盖 invariant

`validate_mmw_id_block_split` MUST 检查 block 集合、base sample、天气副本和窗口引用原始 frame 在 train/validation/test 间无交集；每个窗口不跨 block；三个 split 均覆盖全部 scene 与 trajectory。trajectory overlap MUST 标记为允许且是协议目标；base-frame、block、window-frame 与 weather-copy overlap MUST 为零。验证还 MUST 输出各 split 64 类 histogram、TV、JSD、Pearson/Spearman、unseen beam、最大比例差，按 scene/domain 与 trajectory 的条件 TV/未覆盖 beam 质量，以及按 block、base frame、天气样本和最终窗口计数的比例。失败 MUST 在 loader 前终止。

#### Scenario: 同一基础帧的天气副本跨 split

- **WHEN** sunny、rainy、foggy 中任一对应 frame 被分到不同 role
- **THEN** audit MUST 失败并报告 base sample identity

### Requirement: seed 0 必须生成机器可读与 Markdown 报告

开发阶段默认 `split_seed=0`、`train_seed=0`。seed 0 MUST 生成 `outputs/split_reports/mmw_id_stratified_block_seed0.json` 与 `.md`，包含协议/manifest/window identity、总体/scene/trajectory 规模、64 类标签分布、简单连续 baseline 对比、最大偏差 beam 和逐项 leakage PASS/FAIL。报告是 ignored 本地产物，不得提交为源码证据。

#### Scenario: 分布目标无法达到

- **WHEN** 满足硬约束的最优 assignment 仍超过 TV 目标或 correlation 目标
- **THEN** 系统 MUST 保留该 block-level assignment并报告实际指标和偏差最大 beam
- **AND** MUST NOT 回退到逐样本随机划分

### Requirement: test 必须默认不加载且只可显式访问

普通 MMW 训练 MUST 只构建 train 与 validation；只有显式 `--evaluate-test` 才能构建或读取 test，并必须记录 `test_evaluated=true`。test MUST 不参与 checkpoint selection、early stopping、router/融合/mask/超参数选择、prototype 或 calibration。`split_seed` 只控制 block assignment；`train_seed` 只控制初始化、shuffle、dropout 与训练采样，并分别写入 resolved config、checkpoint 和结果 provenance。

#### Scenario: 默认开发训练

- **WHEN** 用户未传 `--evaluate-test`
- **THEN** loader 集合 MUST 只有 train/validation
- **AND** metadata MUST 记录 `test_evaluated=false`

### Requirement: 所有可拟合数据状态必须只来自 train

GPS scaler、normalization、beam frequency/class weight、prototype、cluster center、contrastive memory/negative queue、feature statistics、calibration 与 modality reliability prior MUST 只使用 train split。validation/test 特征 MUST 不写入训练队列、prototype 或任何可拟合状态；跨天气正样本也只能在 train 内构造。

#### Scenario: 构建训练统计或 memory bank

- **WHEN** 统计构建器收到 validation 或 test dataset
- **THEN** 系统 MUST 拒绝更新并保持已有 train-only provenance
