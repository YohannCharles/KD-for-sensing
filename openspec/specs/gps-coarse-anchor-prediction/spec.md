# gps-coarse-anchor-prediction Specification

## Purpose
定义 GPS coarse anchor prediction 的输入边界、输出契约、校准来源和评估产物，确保几何或轻量 neural anchor 只使用可观测 GPS/pose 信息，并能作为 HiST-Beam residual/fusion 工作流的可审计粗粒度 beam 先验。
## Requirements
### Requirement: GPS coarse anchor 输入边界
系统 MUST 提供显式 opt-in 的 GPS coarse anchor prediction profile。该 profile MUST 只使用预测时刻可观测的 GPS/pose、GPS-Rel-Polar 特征、RSU pose、历史时间戳和合法 calibration split metadata，不得使用 target_test oracle 字段。

#### Scenario: 只使用 GPS 可观测输入
- **WHEN** 系统为 target_test 样本生成 GPS coarse anchor
- **THEN** anchor 输入 MUST 只包含预测时刻之前或预测时刻可观测的 GPS/pose、GPS-Rel-Polar、RSU pose、时间戳和配置允许的 calibration metadata
- **AND** 系统 MUST NOT 使用 future beam label、beam_power argmax、path/radio/channel oracle、image/radar/lidar/mmwave 特征或 target_test 标签生成 anchor
- **AND** run metadata MUST 记录实际使用字段列表

#### Scenario: target_test 不参与校准
- **WHEN** 系统启用 boresight、beam direction、beam offset 或 neural anchor 参数选择
- **THEN** 校准或参数选择 MUST 只使用 source split 或 target_adapt support split
- **AND** target_test MUST 只用于最终评价
- **AND** metadata MUST 记录 `used_target_test_for_calibration=false`

#### Scenario: GPS 字段缺失时记录不可用原因
- **WHEN** 样本或场景缺少 GPS coarse anchor 所需字段
- **THEN** 系统 MUST 跳过 anchor 生成或使用配置允许的 fallback
- **AND** 输出 artifact MUST 记录缺失字段、受影响样本数和 fallback 状态

### Requirement: GPS coarse anchor 输出契约
系统 MUST 将 GPS coarse anchor 表示为统一结构，供独立评估和后续 residual/fusion 模型消费。默认 64 类 beam codebook 与 `group_size=8` 时 coarse group 数 MUST 为 8。

#### Scenario: 输出 coarse anchor 字段
- **WHEN** GPS coarse anchor profile 完成 forward 或几何预测
- **THEN** 系统 MUST 输出 `coarse_logits`，形状为 `[B, H, G]`
- **AND** 系统 MUST 输出 `center_beam` 或 `residual_anchor_beam`，形状为 `[B, H]`
- **AND** 系统 MUST 输出 `confidence`，形状为 `[B, H]`
- **AND** `G` MUST 等于 `num_classes // group_size`

#### Scenario: 可选输出 beam score
- **WHEN** anchor source 能产生 beam-level 分布
- **THEN** 系统 MUST 输出 `beam_scores`，形状为 `[B, H, C]`
- **AND** Top-K、DBA 或 beam power 指标 MUST 基于该 beam-level score 或其明确的 top-k 近邻展开计算

#### Scenario: anchor metadata 可审计
- **WHEN** 系统保存 GPS anchor prediction artifact
- **THEN** artifact MUST 包含 sample id、scene、split、anchor source、coarse top-k、center beam、confidence、GPS coverage 和 calibration metadata
- **AND** artifact MUST 不复制大型原始传感器数据、channel tensor 或 path/radio oracle tensor

### Requirement: BeamBench-style 几何 anchor
系统 MUST 支持可解释的几何校准 GPS anchor。几何 anchor MUST 先将 GPS/pose 转为相对方位，经 boresight 中心化后映射到 beam codebook，并从 beam center 汇总 coarse logits。

#### Scenario: boresight 中心化后映射 beam
- **WHEN** 用户配置 `anchor_source=geometry_calibrated`
- **THEN** 系统 MUST 计算相对方位并应用 `calibrated_azimuth = relative_azimuth - boresight_angle_degrees`
- **AND** 系统 MUST 使用配置的 beam direction、beam offset 和 codebook 参数映射到合法 beam id
- **AND** 输出 metadata MUST 记录 effective boresight、direction、offset 和校准 split

#### Scenario: beam center 汇总 coarse logits
- **WHEN** 几何 anchor 生成 center beam
- **THEN** 系统 MUST 根据 `group_size` 计算 coarse group
- **AND** coarse logits MUST 能表达该 coarse group 及其邻近 group 的 score
- **AND** `center_beam` MUST 位于 `[0, num_classes)` 范围内

#### Scenario: 几何 anchor 不训练神经网络
- **WHEN** 用户只运行 `geometry_calibrated` GPS anchor
- **THEN** 系统 MUST 不执行 backward、不创建 optimizer、不加载 checkpoint
- **AND** run metadata MUST 记录 `uses_neural_network=false`

### Requirement: GPS neural coarse anchor
系统 MUST 支持轻量 GPS neural coarse head 作为 opt-in 训练变体。该 head MUST 基于 GPS encoder 或 GPS-Rel-Polar 特征预测 coarse group，并可选输出 beam-level auxiliary logits。

#### Scenario: 构建 GPS neural coarse head
- **WHEN** 配置启用 `anchor_source=gps_neural_coarse`
- **THEN** 模型注册或构建流程 MUST 创建 GPS coarse head
- **AND** coarse head MUST 接收 GPS feature 或 temporal representation
- **AND** coarse head 输出 MUST 满足 GPS coarse anchor 输出契约

#### Scenario: 训练 coarse loss
- **WHEN** 训练 GPS neural coarse anchor
- **THEN** 系统 MUST 根据 beam label 和 `group_size` 生成 coarse label
- **AND** 系统 MUST 对 `coarse_logits` 计算 coarse cross-entropy loss
- **AND** 若配置启用 beam auxiliary loss，系统 MUST 记录 auxiliary loss 权重和 beam-level loss

#### Scenario: 保持现有 GPS 模型兼容
- **WHEN** 用户运行未启用 GPS coarse anchor 的现有 `gps_teacher` 或 `gps_student` 配置
- **THEN** forward 返回契约 MUST 保持现有 `(pred, input_features, output_features)` 或既有模型输出语义
- **AND** 系统 MUST NOT 静默新增 coarse anchor loss

### Requirement: 跨场景 GPS anchor 评估
系统 MUST 提供 GPS coarse anchor 跨场景评估 profile，报告 source/target 场景拆分、seen/unseen 场景指标和 distribution-shift 诊断。评估 MUST 支持 DeepSense6G Scenes 31-34 或本地可用 MMW LOSO 场景。

#### Scenario: 输出 anchor 指标
- **WHEN** GPS anchor evaluation 完成
- **THEN** metrics MUST 至少包含 coarse accuracy、center beam Top-1/Top-3、circular beam error、样本数和 anchor confidence summary
- **AND** 若 beam power 或 DBA 所需字段可用，metrics MUST 输出 DBA、normalized received power 或 beam power loss dB
- **AND** 若不可用，metrics MUST 记录不可用原因

#### Scenario: 标记未见场景
- **WHEN** evaluation 使用 source scenes 和 target scene 拆分
- **THEN** summary MUST 记录每个 run 的 source scenes、target scene、target 是否 seen during training/calibration 和 split protocol
- **AND** DeepSense6G Scene 31 held-out 或 MMW target LOSO MUST 被标记为 unseen target

#### Scenario: 输出 distribution shift 诊断
- **WHEN** anchor evaluation 同时包含 source 和 target split
- **THEN** summary MUST 输出 GPS/azimuth/coarse-label 分布差异或不可用原因
- **AND** summary MUST 输出 anchor error 按场景、range 或 azimuth 分桶的统计

### Requirement: Residual anchor 预览
系统 MUST 为后续其它模态 residual learning 输出 residual preview。Residual preview MUST 基于 GPS anchor beam 与真实 beam 的环形差值，并且只作为评估/诊断结果写出，不得反向污染 anchor 生成。

#### Scenario: 生成 residual preview
- **WHEN** evaluation split 存在真实 beam label
- **THEN** 系统 MUST 计算 `residual = (true_beam - residual_anchor_beam) mod num_classes`
- **AND** summary MUST 记录 residual histogram、residual entropy 和 residual 在 top-k anchor 邻域内的比例

#### Scenario: residual preview 不用于预测
- **WHEN** 系统生成 target_test residual preview
- **THEN** residual preview MUST NOT 作为 GPS anchor 输入、校准输入或参数选择依据
- **AND** metadata MUST 标记 residual preview 只用于 evaluation diagnostics

#### Scenario: 后续模态可消费 anchor
- **WHEN** downstream residual/fusion 模型启用 GPS anchor 条件输入
- **THEN** batch 或 model input MUST 能提供 `coarse_logits`、`center_beam`、`confidence` 和 `residual_anchor_beam`
- **AND** 缺失 anchor 时系统 MUST 抛出清晰错误或记录配置允许的 fallback

### Requirement: DeepSense6G GPS v2 prior artifact export
GPS v2 workflow MUST support exporting beam-level prior logits and an index file for downstream residual correction without changing existing prediction semantics.

#### Scenario: 保存 GPS v2 logits
- **WHEN** 用户显式启用 GPS v2 `save_logits`
- **THEN** 系统 MUST 写出 `gps_logits.npy`，形状为 `[N, 64]`
- **AND** 系统 MUST 写出 `gps_logits_index.csv`，包含 scene、sample id 和 row index
- **AND** 当配置启用 probability export 时，系统 MUST 写出 `gps_prior_probs.npy`
- **AND** predictions 与 summary 的既有字段语义 MUST 保持兼容

#### Scenario: logits index 可追踪
- **WHEN** downstream residual manifest 读取 GPS logits
- **THEN** `gps_logits_index.csv` MUST 能把每个 logits row 映射回 scene 与 sample id
- **AND** index 中重复或缺失映射 MUST 被清晰拒绝

### Requirement: DeepSense6G GPS prior fallback
当 GPS v2 logits 不可用时，下游 residual workflow MUST 能从 GPS top1 prediction 构造 circular Gaussian fallback prior，并显式记录 fallback 来源。

#### Scenario: fallback Gaussian 不使用 target label
- **WHEN** residual workflow 使用 fallback Gaussian prior
- **THEN** prior center MUST 来自 GPS predicted top1
- **AND** prior MUST NOT 使用 target label、query label 或 beam power oracle
- **AND** metadata MUST 记录 `gps_prior_source=fallback_gaussian_from_top1`

#### Scenario: fallback sigma 可配置
- **WHEN** 用户设置 `residual.gps_prior_fallback_sigma`
- **THEN** fallback prior MUST 使用该 sigma 生成 circular Gaussian logits 或 probability
- **AND** 默认 sigma MUST 为 `2.0`

