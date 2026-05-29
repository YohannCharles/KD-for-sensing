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
系统 MUST 参考 MMW V2I 文件层级，按 `Town10/Town10_skybridge_seed24/<agent>/<six_digit_frame>` 索引 CAV、RSU 和信道文件。系统 MUST 将 CAV 的 `.yaml`、`.pcd`、`_camera0.png` 到 `_camera3.png`、channel `_paths` 文件，以及 RSU 的 `.yaml`、`.pcd`、camera/depth/radar 文件纳入索引；缺失的启用模态 MUST 以可机器读取的 skip reason 记录。

#### Scenario: CAV 帧包含同步模态
- **WHEN** 某个 CAV agent 的同一六位帧号同时存在 yaml、LiDAR pcd、四路 RGB camera 和 channel paths 文件
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
系统 MUST 从有效 frame manifest 生成 beam 预测可用的序列 CSV。序列窗口 MUST 在同一 CAV agent 和连续 frame 片段内生成，不得跨 agent 或跨不连续 frame 拼接历史输入和未来标签。CSV MUST 至少包含 `seq_index`、历史 `camera*`、`lidar*`、`gps*`、`mmwave*`、`beam*` 列和 `future_beam*` 标签列，并 MUST 写出 train/test split metadata。

#### Scenario: 生成历史 8 帧和未来 3 帧窗口
- **WHEN** 配置 `seq_len=8` 且 `pred_len=3`
- **THEN** 每个输出样本 MUST 包含 `beam1..beam8`、`mmwave1..mmwave8` 和 `future_beam1..future_beam3`
- **AND** `future_beam1` MUST 对应当前历史窗口后的第一个未来帧
- **AND** 所有历史和未来 frame MUST 属于同一 CAV agent 和同一连续片段

#### Scenario: split 以序列组为单位
- **WHEN** 系统生成 train/test CSV
- **THEN** 同一个 `seq_index` 或连续片段的窗口 MUST 不得同时出现在 train 和 test
- **AND** split metadata MUST 记录 split seed、比例、train/test seq 列表、窗口数和 beam label 分布摘要

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

