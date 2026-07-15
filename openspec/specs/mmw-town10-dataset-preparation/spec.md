# mmw-town10-dataset-preparation Specification

## Purpose
定义 MMW Town10 skybridge 本地 zip 输入、准备输出和 sanity metadata 契约。
## Requirements
### Requirement: MMW Town10 zip inputs are prepared under the canonical layout
系统 MUST 提供 MMW Town10 skybridge 数据准备入口，接受本地传感器 zip `Town10_skybridge_seed24.zip` 和信道 zip `Town10.zip`，并将解包结果组织到 `dataset/MMW/<condition>/Sensor_Data` 与 `dataset/MMW/<condition>/Channel_Data` 下。系统 MUST 不下载数据，不把 zip 内容提交到源码目录，并 MUST 支持通过配置或 CLI 覆盖输入 zip、输出根目录、condition 和 scenario 名称。

#### Scenario: 准备本地 sunny Town10 skybridge zip
- **WHEN** 用户运行 MMW 准备入口并提供 `sensor_zip=Town10_skybridge_seed24.zip`、`channel_zip=Town10.zip`、`condition=sunny`
- **THEN** 系统 MUST 将传感器数据解包或索引到 `dataset/MMW/sunny/Sensor_Data`
- **AND** 系统 MUST 将信道数据解包或索引到 `dataset/MMW/sunny/Channel_Data`
- **AND** 系统 MUST 在 metadata 中记录输入 zip 绝对路径、文件大小、内容摘要或 mtime、condition、town 和 scenario

#### Scenario: 输入 zip 缺失
- **WHEN** 用户提供的传感器 zip 或信道 zip 不存在
- **THEN** 系统 MUST 失败并输出包含缺失绝对路径的错误
- **AND** 系统 MUST 不创建不完整的 manifest 或 split CSV

### Requirement: MMW sensor and channel frames are indexed by agent and frame
系统 MUST 参考 MMW V2I 文件层级，按 `Town10/Town10_skybridge_seed24/<agent>/<frame_id>` 索引 CAV、RSU 和信道文件，其中 `frame_id` MUST 支持至少六位的数字帧号。系统 MUST 将 CAV 的 `.yaml`、`.pcd`、`_camera0.png` 到 `_camera3.png`、channel `_paths` 文件，以及 RSU 的 `.yaml`、`.pcd`、camera/depth/radar 文件纳入索引；缺失的启用模态 MUST 以可机器读取的 skip reason 记录。

#### Scenario: CAV 帧包含同步模态
- **WHEN** 某个 CAV agent 的同一数字帧号同时存在 yaml、LiDAR pcd、四路 RGB camera 和 channel paths 文件
- **THEN** 系统 MUST 将该帧标记为可用于生成样本
- **AND** frame manifest MUST 记录 agent id、frame id、CAV 模态相对路径、匹配到的 RSU frame 路径和 channel paths 相对路径

#### Scenario: 启用模态缺失
- **WHEN** 某个候选帧缺少当前配置要求的 channel paths 或 CAV 前向 camera
- **THEN** 系统 MUST 跳过该帧
- **AND** sanity report MUST 按 agent、frame 和 reason 统计跳过数量
- **AND** 系统 MUST 继续处理其它有效帧

### Requirement: Channel paths are converted into 64-beam power vectors and labels
系统 MUST 从 MMW channel `_paths.npy` 或 `_paths.npz` 文件读取多径信道字段，并为每个有效 CAV frame 生成固定长度 beam power vector。默认 beam 数 MUST 为 64，默认标签 MUST 为 power vector 最大值的 0-based `argmax`，输出 beam power 文件 MUST 可由现有 beam label 读取逻辑和 mmWave power vector 读取逻辑消费。

#### Scenario: 从 channel 文件生成 beam power
- **WHEN** channel paths 文件包含可解析的复信道增益、AoD/AoA 或等效 channel 字段
- **THEN** 系统 MUST 生成长度为 `num_beams` 的 finite 浮点 power vector
- **AND** 系统 MUST 将 power vector 写入 `Prepared/Town10_skybridge_seed24/beam_power/<agent>/<frame>.txt` 或等价稳定相对路径
- **AND** 系统 MUST 在 frame manifest 中记录该 power vector 路径作为 `mmwave` 和 `beam` 来源
- **AND** metadata MUST 记录 `num_beams`、codebook 类型、天线数量、使用的 channel 字段和算法版本

#### Scenario: channel 文件不可派生标签
- **WHEN** channel paths 文件缺少必要字段、shape 非法、没有有效路径或派生结果全为 NaN/Inf
- **THEN** 系统 MUST 跳过对应帧
- **AND** sanity report MUST 记录失败路径、失败原因和涉及字段

### Requirement: Sequence CSV and split artifacts are generated
系统 MUST 从有效 frame manifest 生成 beam 预测可用的序列 CSV。序列窗口 MUST 在同一 CAV agent 和连续 frame 片段内生成，不得跨 agent 或跨不连续 frame 拼接历史输入和未来标签。CSV MUST 至少包含 `seq_index`、历史 `camera*`、`lidar*`、`gps*`、`mmwave*`、`beam*` 列和 `future_beam*` 标签列，并 MUST 写出 train/test split metadata。默认 train/test split MUST 使用 group-safe 协议，按连续片段、agent、时间块或等价 group 分配窗口，并保留足够 guard band，避免 train/test 共享相邻滑窗上下文；公开准备流程和 split builder 不再支持随机窗口切分。

#### Scenario: 生成历史 8 帧和未来 3 帧窗口
- **WHEN** 配置 `seq_len=8` 且 `pred_len=3`
- **THEN** 每个输出样本 MUST 包含 `beam1..beam8`、`mmwave1..mmwave8` 和 `future_beam1..future_beam3`
- **AND** `future_beam1` MUST 对应当前历史窗口后的第一个未来帧
- **AND** 所有历史和未来 frame MUST 属于同一 CAV agent 和同一连续片段

#### Scenario: split 以 group-safe 单位生成
- **WHEN** 系统生成默认 train/test CSV
- **THEN** 同一个 `seq_index`、连续片段 group、time block 或 guard band 内的窗口 MUST 不得同时出现在 train 和 test
- **AND** train/test 之间 MUST 不共享完整历史+未来窗口中的 frame id
- **AND** split metadata MUST 记录 split strategy、protocol version、group key、guard band、split seed、比例、train/test group 列表、窗口数和 beam label 分布摘要
- **AND** split metadata MUST 标记 `strict_validation_eligible=true`

#### Scenario: unsupported split strategy 不生成随机窗口 split
- **WHEN** 用户通过配置或内部 API 请求不在支持列表中的 split strategy
- **THEN** 系统 MUST 按普通 unsupported strategy 处理
- **AND** 系统 MUST NOT 生成随机窗口 train/test CSV
- **AND** 系统 MUST NOT 为旧随机窗口策略写出专门 metadata 分支

### Requirement: MMW preparation writes reproducibility and sanity artifacts
系统 MUST 为每次 MMW 准备运行写出 metadata 和 sanity report。报告 MUST 足以审计输入、输出、有效样本数、缺失模态、channel 派生失败、beam 分布和 manifest/split 路径。

#### Scenario: 准备完成后写出报告
- **WHEN** MMW Town10 数据准备成功完成
- **THEN** 系统 MUST 写出 `metadata.json`
- **AND** 系统 MUST 写出 `sanity_report.json`
- **AND** 报告 MUST 包含 frame 总数、有效 frame 数、窗口数、agent 列表、模态缺失统计、channel 派生失败统计、beam label 直方图和输出 CSV 路径

#### Scenario: 无有效样本
- **WHEN** 输入 zip 可读取但没有任何 frame 能生成完整历史和未来窗口
- **THEN** 系统 MUST 失败并输出清晰错误
- **AND** sanity report MUST 保留导致无有效样本的主要原因统计

### Requirement: MMW local download processing
MMW Town10 preparation MUST 通过显式配置或 CLI overrides 支持位于 `dataset/_downloads/MMW/<condition>/Sensor_Data` 和 `dataset/_downloads/MMW/<condition>/Channel_Data` 的 zip inputs。Processing MUST 将 prepared artifacts 写入 `dataset/MMW/<condition>/Prepared/<scenario>`，并 MUST 不移动或删除已下载 zip 文件。

#### Scenario: 处理 sunny 已下载 zip
- **WHEN** 用户提供 sensor zip `dataset/_downloads/MMW/sunny/Sensor_Data/Town10_skybridge_seed24.zip` 和 channel zip `dataset/_downloads/MMW/sunny/Channel_Data/Town10.zip`
- **THEN** preparation MUST extract or index them into `dataset/MMW/sunny/Sensor_Data` and `dataset/MMW/sunny/Channel_Data`
- **AND** prepared artifacts MUST be written under `dataset/MMW/sunny/Prepared/Town10_skybridge_seed24`
- **AND** metadata MUST record source zip absolute paths and fingerprints

### Requirement: Sensor/channel scenario alias matching
MMW preparation MUST explicitly handle cases where sensor scenario names and channel scenario directories differ, such as sensor `Town10_skybridge_seed24` and channel `Town10/Town10_skybridge`. Matching MUST be based on declared alias, frame id and CAV agent, and MUST be recorded in metadata.

#### Scenario: channel agent 匹配正确
- **WHEN** a sensor frame belongs to agent `cav_1` and frame `008362`
- **THEN** preparation MUST prefer channel paths under channel agent `cav_1` for frame `008362`
- **AND** it MUST NOT silently match the frame to `cav_2` or `cav_3`
- **AND** frame manifest MUST make the matched channel agent auditable

#### Scenario: alias 匹配写入 metadata
- **WHEN** sensor scenario and channel scenario names differ but are matched by alias
- **THEN** metadata MUST record sensor scenario, channel scenario, alias rule and matched frame count
- **AND** unmatched frames MUST be counted by reason

### Requirement: Prepared artifact validity checks
MMW preparation MUST run validity checks before declaring prepared status. Checks MUST include finite beam power vectors, non-empty sequence windows, CAV/channel agent consistency, frame continuity and required modality coverage.

#### Scenario: agent 错配导致失败
- **WHEN** frame manifest contains a CAV agent whose channel path points to a different CAV agent without explicit override
- **THEN** preparation MUST fail or mark the artifact invalid
- **AND** sanity report MUST include examples of mismatched rows

#### Scenario: 有效 prepared summary
- **WHEN** preparation succeeds
- **THEN** sanity report MUST include valid frame count, window count, agent frame counts, modality coverage, channel failure counts, beam histogram, train/test window counts and artifact paths

### Requirement: Incremental MMW preparation
MMW preparation MUST support incremental processing as additional condition/town/scenario zips arrive. Incremental processing MUST preserve existing prepared artifacts unless `force` is explicitly requested for the same condition/scenario.

#### Scenario: 新 condition 到达
- **WHEN** rainy or foggy zip files become available after sunny has already been prepared
- **THEN** preparation MUST process the new condition into its own `dataset/MMW/<condition>` directory
- **AND** existing sunny prepared artifacts MUST remain unchanged unless explicitly forced

#### Scenario: force 重建单个 scenario
- **WHEN** user requests force rebuild for `sunny/Town10_skybridge_seed24`
- **THEN** preparation MAY overwrite that prepared scenario artifacts
- **AND** it MUST not remove other condition or scenario prepared artifacts

### Requirement: MMW radio-semantic derivation metadata
MMW preparation and dataset runtime MUST expose enough metadata to derive radio-semantic labels from channel-derived beam power vectors. The system MUST record whether each frame or sequence can derive a radio label, the builder mode/config version, and the unavailable reason when derivation fails.

#### Scenario: frame manifest supports radio label derivation
- **WHEN** frame manifest contains a finite 64-beam power vector path and beam label for a CAV frame
- **THEN** manifest or derived metadata MUST identify that the frame is eligible for radio-semantic label construction
- **AND** metadata MUST include `num_beams`, codebook/profile information, beam_power path and label source

#### Scenario: derivation unavailable is explicit
- **WHEN** channel file cannot produce finite beam power or the beam power file is missing
- **THEN** preparation or dataset metadata MUST mark radio-semantic derivation as unavailable
- **AND** metadata MUST record a machine-readable reason such as missing beam power, invalid power vector or unsupported channel fields

### Requirement: MMW radio-semantic labels are not sensing inputs
MMW dataset configuration MUST keep radio-semantic labels, CSI/channel paths and beam_power separate from sensing input modalities. Enabling radio-semantic training MUST NOT implicitly enable CSI/channel/beam_power as model input.

#### Scenario: radio label enabled without channel input
- **WHEN** user enables `radio_semantic.enabled: true` for an MMW HiST-Beam run
- **THEN** dataset MAY return `radio_semantic_label` and `beam_power` for labels or metrics
- **AND** model input modalities MUST remain limited to the configured sensing modalities

#### Scenario: channel-derived metrics are evaluation-only on target_test
- **WHEN** target_test samples contain beam_power
- **THEN** evaluation MAY compute normalized received power and beam power loss dB
- **AND** target adaptation MUST NOT use target_test beam_power for training, threshold selection or prototype update

### Requirement: MMW split 与 radar CSV materialization 使用公开准备入口
MMW Town10 数据准备 MUST 提供公开 package utility、preprocessor 或 CLI，用于创建和校验 sensor-assisted split、sequence CSV 和 radar CSV materialization。训练 preflight MUST 调用该公开入口或读取已准备 artifact，不得依赖 dataset 私有 helper 来写出 CSV。

#### Scenario: 公开 utility 生成 radar CSV
- **WHEN** 用户或 preflight 调用公开 MMW split/radar CSV 准备入口
- **THEN** 系统 MUST 基于 prepared manifest 和 split metadata 生成需要的 sequence CSV 或 radar CSV
- **AND** 输出 metadata MUST 记录输入 manifest、split 配置、seq_len、num_pred、condition、scenario、样本数和输出路径
- **AND** 生成产物 MUST 位于 dataset 或显式本地输出目录，不得写入源码控制目录

#### Scenario: 训练 preflight 不导入私有 dataset helper
- **WHEN** HiST-Beam MMW LOSO executor 执行 preflight
- **THEN** preflight MUST NOT 从 dataset 模块导入 `_ensure_*` 私有 helper 来物化 radar CSV 或 split CSV
- **AND** preflight MUST 只读取已准备 artifact、调用公开准备 utility 或报告缺失 artifact

#### Scenario: 缺失 prepared artifact 给出可执行提示
- **WHEN** preflight 发现 sensor-assisted run 所需 split CSV 或 radar CSV 缺失
- **THEN** preflight MUST 失败并输出可执行修复提示
- **AND** 提示 MUST 包含公开 MMW 准备入口、关键参数和目标输出路径
- **AND** preflight MUST 不静默创建不完整或无 metadata 的 CSV

### Requirement: MMW split leakage diagnostics
MMW Town10 split metadata MUST 包含可机器读取的泄漏诊断，用于判断当前 train/test CSV 是否可作为 strict validation 协议。诊断 MUST 至少覆盖 train/test frame overlap、test window 与 train window 的最大 frame overlap、相邻窗口跨 split 比例和未来标签序列复用比例。未来标签序列复用 MUST 作为标签分布诊断保留；当 `pred_len=1` 时，beam 类别重复本身 MUST NOT 被解释为 frame、window 或 trajectory 泄漏，也 MUST NOT 单独使 split strict-ineligible。

#### Scenario: group-safe split 诊断通过
- **WHEN** 系统使用默认 group-safe 协议生成 split
- **THEN** leakage diagnostics MUST 记录 train/test frame overlap count 为 0
- **AND** test window 与任一 train window 的最大 frame overlap MUST 小于完整窗口长度
- **AND** summary MUST 包含 guard band frames、window length、train/test window counts 和 diagnostics 生成时间或版本
- **AND** P1 future label class 在 train/test 重复时 MUST 继续报告 reuse ratio，但 strict eligibility MUST 由结构性 overlap diagnostics 决定

#### Scenario: 诊断发现高重叠
- **WHEN** leakage diagnostics 发现 test window 与 train window 共享完整或近完整历史+未来上下文
- **THEN** split metadata MUST 标记 `strict_validation_eligible=false`
- **AND** metadata MUST 包含超阈值统计和可执行修复提示
- **AND** 训练或评估产物消费该 metadata 时 MUST 能显示该 split 不适合作为 strict 主结论

### Requirement: MMW prepared split strategy is auditable
MMW Town10 数据准备和公开 split builder MUST 在输出 metadata 中记录实际生效的 split strategy。未显式设置时，默认 MUST 使用 group-safe 策略；当前公开准备协议只支持 `group_safe_time_block`，不提供旧随机窗口切分兼容路径。

#### Scenario: 默认使用 group-safe strategy
- **WHEN** 用户运行 MMW Town10 preparation 或 `build_sequence_splits_from_manifest` 且未指定 split strategy
- **THEN** 系统 MUST 使用 group-safe split strategy
- **AND** 输出 split metadata MUST 记录默认来源和策略参数

#### Scenario: 公开 split builder 只暴露 group-safe strategy
- **WHEN** 用户查看 MMW Town10 preparation 或 `build_sequence_splits_from_manifest` 的公开参数
- **THEN** 可用 split strategy MUST 限定为 `group_safe_time_block`
- **AND** 输出 split metadata MUST 记录该策略和策略参数

### Requirement: MMW preparation records beam label calibration provenance
MMW Town10 preparation MUST preserve raw channel-derived beam label provenance and MAY record calibration candidate metadata without overwriting raw beam power vector semantics.

#### Scenario: frame manifest 保留 raw beam label
- **WHEN** preparation 从 channel 文件生成 beam power vector 和 raw `argmax` label
- **THEN** frame manifest MUST 记录 raw beam label、beam power path、num beams 和 label source
- **AND** preparation MUST NOT rewrite the beam power vector to express calibrated class order

#### Scenario: calibration metadata 可审计
- **WHEN** preparation 或后续诊断产出 scene-level calibration candidate
- **THEN** metadata MUST record direction、offset、num_classes、label_space name、fit source 和算法版本
- **AND** metadata MUST distinguish candidate calibration from the raw label used to generate beam power files

#### Scenario: split metadata 同时说明 raw/calibrated label 分布
- **WHEN** split builder receives an enabled calibration config
- **THEN** split metadata MAY include calibrated label histograms in addition to raw label histograms
- **AND** each histogram MUST declare its label space and mapping fingerprint

### Requirement: MMW prepared manifests are loadable by modality-aware datasets
数据构建流程 MUST 能识别 MMW 准备流程生成的 manifest/CSV，并在配置选择 `data.dataset.type: mmw` 与 `data.dataset.scene: town10_skybridge_seed24` 时构建对应 dataset。启用模态推导、按需读取、beam 历史标签和 future beam 目标标签的语义 MUST 与现有 beam 预测流程保持一致。

#### Scenario: MMW mmWave-only 按需读取
- **WHEN** 用户使用 MMW manifest 运行 `experiment.task: mmwave`
- **THEN** dataset MUST 只读取历史 `mmwave*` power vector、`beam*` 和 `future_beam*` 标签文件
- **AND** dataset MUST 不读取 image、LiDAR、GPS 或 RSU radar 文件
- **AND** 返回样本 MUST 包含 `mmwave`、`input_beam` 和 `target_beam`

#### Scenario: MMW image+mmWave fusion 按需读取
- **WHEN** 用户使用 MMW manifest 运行 fusion 配置且启用 `["image", "mmwave"]`
- **THEN** dataset MUST 读取历史前向 RGB image、历史 mmWave power vector、历史 beam 和 future beam 标签
- **AND** dataset MUST 不要求未启用的 LiDAR、GPS 或 RSU radar 文件存在
- **AND** 返回样本 MUST 只包含启用模态对应输入字段和标签字段

### Requirement: MMW dataset returns stable beam and modality tensors
MMW dataset MUST 返回与现有训练流程兼容的 `input_beam` 和 `target_beam` 张量。启用 MMW 派生 mmWave 输入时，`mmwave` MUST 为 `[seq_len, 64]` 的 `torch.float32` 张量；启用 image、LiDAR 或 GPS 时，对应字段 MUST 使用现有 batch 准备流程可消费的稳定 shape 和 dtype。

#### Scenario: MMW beam 标签 shape 稳定
- **WHEN** MMW dataset 配置 `seq_len=8` 且 `num_pred=3`
- **THEN** 单样本 `input_beam` MUST 为长度 8 的整数张量
- **AND** 单样本 `target_beam` MUST 为长度 3 的整数张量
- **AND** batch 后 `target_beam` MUST 保持 `[batch_size, 3]`

#### Scenario: MMW mmWave 张量 shape 稳定
- **WHEN** MMW dataset 启用 mmWave modality
- **THEN** 单样本 `mmwave` MUST 为 `torch.float32`
- **AND** `mmwave` shape MUST 为 `[seq_len, 64]`
- **AND** 每个时隙 MUST 与同一行 CSV 的 `beam*` 历史标签时隙对齐

### Requirement: MMW dataset 初始化内存有界
MMW dataset 初始化 MUST 避免为了 normalizer、CSV 派生列或 metadata 准备而无界持有所有样本的大数组。GPS、mmWave 和 CSI 等模态的 normalizer 拟合 MUST 使用 streaming 或可释放的临时统计，并 MUST 在拟合完成后避免把 per-sample sequence cache 常驻到 DataLoader worker。

#### Scenario: GPS/mmWave scaler 拟合不保留全量样本缓存
- **WHEN** MMW train dataset 启用 GPS 或 mmWave normalization
- **THEN** scaler 拟合 MUST 能通过 streaming 或临时数组完成
- **AND** 拟合完成后 dataset MUST 不保留所有样本的 GPS/mmWave sequence 大数组缓存
- **AND** runtime metadata MUST 记录 scaler 来源、样本数和是否使用 streaming 拟合

#### Scenario: DataLoader worker 不复制初始化大缓存
- **WHEN** MMW dataset 使用多 worker DataLoader
- **THEN** worker 进程 MUST 不因 dataset 初始化阶段的 per-sample feature cache 而复制全量样本大数组
- **AND** profile 或 metadata MUST 能报告 worker 内存风险相关配置

### Requirement: MMW image 序列按需加载与缓存等价
MMW image modality MUST 按 enabled modalities 和 seq_len 读取 RGB/ImageNet image 序列。启用 image-derived cache 时，dataset MUST 保持与原始 image 读取路径一致的样本字段、shape、dtype 和 label 语义。

#### Scenario: image-derived cache 保持 batch 契约
- **WHEN** MMW fusion 配置启用 image modality、`seq_len=8` 和 image-derived cache
- **THEN** 单样本 `image` tensor MUST 保持 `[seq_len, 3, H, W]`
- **AND** batch 后 image 输入 MUST 与未启用 cache 时的 shape 和 dtype 一致
- **AND** `input_beam`、`target_beam`、GPS 和 mmWave 字段 MUST 不因 image cache 改变

#### Scenario: 未启用 image 不读取 image 路径
- **WHEN** MMW fusion 配置的 modalities 为 `["gps", "mmwave"]`
- **THEN** dataset MUST 不读取 camera 列对应文件
- **AND** dataset MUST 不初始化 image transform 或 image-derived cache

### Requirement: MMW calibrated hard label loading
MMW dataset MUST support returning calibrated hard beam labels when `data.dataset.beam_label_calibration.enabled=true`. Calibration MUST apply to historical `input_beam` and future `target_beam` while preserving existing tensor shapes and modality-aware loading behavior.

#### Scenario: calibrated input 和 target beam shape 稳定
- **WHEN** MMW dataset 配置 `seq_len=8`、`num_pred=3` 且启用 beam label calibration
- **THEN** 单样本 `input_beam` MUST 仍为长度 8 的整数张量
- **AND** 单样本 `target_beam` MUST 仍为长度 3 的整数张量
- **AND** 所有合法 label MUST 位于 `[0, num_classes)` 的 calibrated label space

#### Scenario: 显式 future_beam_label 字段被映射
- **WHEN** MMW split CSV 包含 `future_beam_label1` 或等价显式 raw label 字段
- **THEN** dataset MUST 在启用 calibration 时将该 raw label 映射为 calibrated `target_beam`
- **AND** metadata MUST preserve the original raw label value for audit

#### Scenario: beam label cache 区分 mapping
- **WHEN** beam label cache 为 eager 或 lazy 且 calibration 配置发生变化
- **THEN** dataset MUST NOT reuse cached calibrated labels from a different mapping fingerprint
- **AND** cache diagnostics MUST record the active mapping fingerprint

#### Scenario: 未启用模态仍不读取
- **WHEN** MMW fusion 配置启用 `["gps", "mmwave"]` 且启用 beam label calibration
- **THEN** dataset MUST only read GPS、mmWave、beam labels and enabled targets
- **AND** calibration MUST NOT cause image、LiDAR、radar、CSI、channel 或 path 文件被额外读取 as sensing inputs

### Requirement: 已下载 MMW 多天气统一 H5/P1 准备
系统 MUST 支持对 `sunny`、`rainy` 和 `foggy` 条件下已下载的 MMW Town03 场景生成输入窗口 5、预测窗口 1 的准备产物。每个场景的 split metadata MUST 记录 `seq_len=5` 和 `pred_len=1`，且窗口不得跨 vehicle 或不连续 frame。

#### Scenario: rainy 和 foggy 场景生成 H5/P1 split
- **WHEN** 用户对 rainy 或 foggy 的已下载 Town03 场景运行准备流程
- **THEN** 对应 `Prepared/<scenario>/splits` MUST 生成 H5/P1 的 `all_sequences.csv`、`train.csv` 和 `test.csv`
- **AND** metadata MUST 记录 condition、scenario、输入窗口 5 和预测窗口 1

#### Scenario: sunny 旧窗口重建为 H5/P1
- **WHEN** sunny 场景已有其它窗口长度的准备产物且用户显式运行 H5/P1 重建
- **THEN** sequence split MUST 按已有 frame manifest 重建为 H5/P1
- **AND** 新 metadata MUST 不再声明旧的 8/3 窗口

### Requirement: MMW split-level LMDB 样本缓存
预处理系统 MUST 能使用当前 dataset registry 为 MMW dataset 生成 split-level LMDB 样本缓存，并 MUST 复用现有 LMDB key 和读取契约。缓存 metadata MUST 记录 dataset type、condition、scenario、split、样本数、`seq_len` 和 `num_pred`。

#### Scenario: MMW H5/P1 LMDB 生成
- **WHEN** 配置使用 `dataset.type: mmw`、`seq_len: 5` 和 `num_pred: 1` 生成 train/test LMDB
- **THEN** 每个 split MUST 写入与对应 dataset 长度一致的样本数
- **AND** metadata MUST 记录 `seq_len=5`、`num_pred=1` 和 MMW condition/scenario

#### Scenario: DeepSense6G 旧 LMDB 入口保持兼容
- **WHEN** 用户继续使用 `deepsense6g_sample_lmdb_cache` 预处理类型
- **THEN** 系统 MUST 继续生成可由现有 sample cache reader 读取的 LMDB
- **AND** 旧配置 MUST 不要求改名后才能运行

### Requirement: MMW 可再生成缓存产物边界
MMW 图像、LiDAR 和 sample LMDB 等可再生成缓存 MUST 默认写入 `outputs/cache/MMW/<condition>/`，并 MUST 按 condition、场景、cache kind 和窗口版本避免路径冲突。

#### Scenario: 三种天气缓存隔离
- **WHEN** sunny、rainy 和 foggy 的 H5/P1 缓存同时存在
- **THEN** 每种 condition MUST 使用独立的 `outputs/cache/MMW/<condition>/` 子树
- **AND** 任一 condition 的缓存生成 MUST 不覆盖其它 condition 或其它窗口版本的产物

### Requirement: H5/P1 metadata variants participate in readiness
MMW preparation availability writer MUST 识别同一 prepared scenario 下由显式 split tag 生成的 `metadata_<tag>.json` 与 `sanity_report_<tag>.json`，并 MUST 根据其中 manifest、window count、split eligibility 和 artifact path 判定 readiness，而不是只检查无后缀文件。

#### Scenario: rainy H5/P1 artifacts 已完整
- **WHEN** rainy scenario 具有 `metadata_h5p1.json`、`sanity_report_h5p1.json`、有效 manifest 和 strict split
- **THEN** condition availability MUST 将该 scenario 标记为可供对应 H5/P1 protocol 使用
- **AND** availability MUST 记录实际 metadata/sanity 路径和 split tag

### Requirement: MMW archive extraction 必须防止路径与资源逃逸
MMW archive preparation MUST 在删除或覆盖任何目标目录前完成 member 路径、数量、解压总大小、压缩比和完整 archive digest 校验。解压结果 MUST 先写入受控临时目录，再原子替换目标。

#### Scenario: ZIP member path traversal
- **WHEN** archive member 是绝对路径、包含 `..`，或 resolved destination 位于 extraction root 之外
- **THEN** preparation MUST 拒绝 archive
- **AND** 现有目标目录 MUST 不被删除或修改

#### Scenario: Archive 资源上限
- **WHEN** member 数、声明解压总大小、单文件大小或压缩比超过受控上限
- **THEN** preparation MUST 在写入 member 前失败
- **AND** error MUST 记录命中的上限类型

#### Scenario: 完整 digest 控制复用
- **WHEN** archive SHA256、算法版本或目标 inventory 与 extraction marker 不一致
- **THEN** runtime MUST 不复用旧 extraction
- **AND** MUST 在安全预检通过后重新生成受控 extraction

#### Scenario: 安全原子替换
- **WHEN** archive 全部 member 校验和临时解压成功
- **THEN** runtime MUST 原子发布新的 extraction root
- **AND** 任一失败 MUST 清理临时目录并保留原目标

