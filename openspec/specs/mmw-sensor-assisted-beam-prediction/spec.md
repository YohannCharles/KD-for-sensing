# mmw-sensor-assisted-beam-prediction Specification

## Purpose
定义 MMW sensor-assisted beam prediction 的输入模态边界、快速验证矩阵、target sensitive usage metadata、summary eligibility 和负迁移诊断契约，确保 image/GPS/LiDAR/radar 传感器输入与 mmWave/CSI/beam/path/radio 等敏感监督字段保持可审计分离。
## Requirements
### Requirement: Sensor-assisted 输入模态边界
MMW sensor-assisted beam prediction MUST 将模型 sensing input 限定为配置启用的传感器模态，快速验证主配置 MUST 使用 `image`、`gps`、`lidar` 和 `radar`。`mmwave`、CSI/channel、beam_power、path_params、path_descriptor、path_semantic_label 和 radio_semantic_label MUST NOT 作为 sensing input 进入模型。

#### Scenario: 构建四传感器输入样本
- **WHEN** 用户加载 MMW sensor-assisted dataset 配置
- **THEN** dataset sample MUST 能返回 image、gps、lidar、radar_ra 和 radar_da 输入张量
- **AND** sample MUST 继续返回 beam label 作为 target
- **AND** sample metadata MUST 记录 condition、town、scenario、sample id 和 modality availability

#### Scenario: mmWave 不进入模型输入
- **WHEN** 用户加载 sensor-assisted HiST-Beam 配置
- **THEN** resolved enabled modalities MUST 等于或包含顺序一致的 `image`、`gps`、`lidar`、`radar`
- **AND** resolved enabled modalities MUST NOT 包含 `mmwave`、`csi`、`channel`、`path` 或 `beam_power`
- **AND** model forward kwargs MUST NOT 包含 `mmwave_batch`，除非用户显式加载非 sensor-assisted 配置

#### Scenario: auxiliary 字段只用于非输入路径
- **WHEN** sensor-assisted 配置同时启用 V6 radio 或 V8 path prototype
- **THEN** beam_power、radio_semantic_label、path_descriptor 和 path_semantic_label MAY 用于 source auxiliary/prototype 或 offline diagnostics
- **AND** target adaptation MUST 通过 leakage flags 记录这些字段是否被训练使用
- **AND** target unlabeled split MUST NOT 使用这些字段作为监督 loss

### Requirement: Sensor-assisted 评估与负迁移诊断
MMW sensor-assisted summary MUST 同时报告 absolute performance 和 adaptation 相对 source-only 的变化。summary MUST 能诊断 few-shot adaptation 是否产生负迁移，并 MUST 报告 last-beam diagnostic baseline。

#### Scenario: adapted-source delta
- **WHEN** target adaptation 和 source-only target evaluation 都完成
- **THEN** summary MUST 记录 adapted minus source-only 的 Top-1、Top-3、Top-5、normalized received power 和 beam power loss dB delta
- **AND** summary MUST 记录 delta win/loss 或等价 negative-transfer flag

#### Scenario: last-beam diagnostic baseline
- **WHEN** evaluation metrics 中存在 last-beam baseline
- **THEN** sensor-assisted summary MUST 汇总 last-beam Top-1 和 Top-3
- **AND** summary MUST 标记 last-beam 为 diagnostic baseline，除非配置显式声明它是可比较 baseline
- **AND** 主模型输入 MUST NOT 因报告 last-beam baseline 而包含历史 beam label shortcut

#### Scenario: 参数效率与泄漏标志
- **WHEN** adaptation run 完成
- **THEN** summary MUST 记录 trainable parameter ratio、adaptation time 和 enabled modalities
- **AND** summary MUST 记录 used_target_beam、used_target_beam_power、used_target_csi、used_target_path_label 和 used_target_radio_label 的训练使用标志
- **AND** 若 sensor-assisted run 使用了不允许的输入或 target sensitive supervision，run MUST 标记失败或不可用于主结论

### Requirement: LiDAR/Radar smoke 与缓存可控性
MMW sensor-assisted workflow MUST 提供 focused smoke，验证 LiDAR/radar 数据路径可读，并允许通过配置控制 LiDAR/radar cache 与 CPU worker 使用。

#### Scenario: 单样本 shape smoke
- **WHEN** 运行 sensor-assisted dataset smoke
- **THEN** smoke MUST 验证 image shape 为 `[T, 3, H, W]`
- **AND** smoke MUST 验证 gps shape 为 `[T, 3]`
- **AND** smoke MUST 验证 lidar shape 为 `[T, C, H, W]`
- **AND** smoke MUST 验证 radar_ra 和 radar_da shape 与 radar encoder 期望兼容

#### Scenario: cache 配置可追踪
- **WHEN** sensor-assisted run 使用 LiDAR 或 radar 派生 cache
- **THEN** run metadata MUST 记录 cache policy、cache dir、num_workers 和 CPU thread settings
- **AND** cache 或派生 `.npy` 产物 MUST 留在本地产物或 dataset 目录，不得纳入源码变更

### Requirement: Sensor-assisted 主结论 eligibility
MMW sensor-assisted run MUST 在 run metadata 和 summary 中明确记录是否可用于主结论。任何使用不允许 sensing input、target_test 训练信息、未授权 target sensitive supervision 或不符合 sensor-assisted profile 的 run MUST 失败或标记 `main_conclusion_eligible=false`，并记录机器可读原因。

#### Scenario: source auxiliary 不影响主结论 eligibility
- **WHEN** sensor-assisted run 只在 source split 使用 beam_power、radio semantic label、path descriptor 或 path semantic label 构造 source auxiliary head、prototype 或离线 diagnostics
- **THEN** run MUST NOT 仅因 source auxiliary/prototype 使用这些字段而被标记为不可用于主结论
- **AND** metadata MUST 记录这些字段未作为 target training supervision 使用
- **AND** sensing input modality 列表 MUST 仍不包含 mmWave、CSI/channel、beam_power、path 或 radio label

#### Scenario: target radio supervision 排除主结论
- **WHEN** sensor-assisted target adaptation 使用 labeled target `radio_semantic_label` 计算训练 loss
- **THEN** run MUST 记录 `used_target_radio_label_for_training=true`
- **AND** run MUST 标记 `main_conclusion_eligible=false`，除非对应实验规格明确允许该 target supervision 进入主结论
- **AND** summary MUST 将该 run 归入补充或诊断结果，而不是主结论比较集合

#### Scenario: target path supervision 排除主结论
- **WHEN** sensor-assisted target adaptation 使用 labeled target `path_semantic_label`、`path_descriptor` 或 `path_params` 计算训练 loss
- **THEN** run MUST 记录对应 `used_target_path_*_for_training` flag
- **AND** run MUST 标记 `main_conclusion_eligible=false`，除非对应实验规格明确允许该 target supervision 进入主结论
- **AND** summary MUST 记录 exclusion reason，不能只输出 accuracy 数值

#### Scenario: summary 写出 eligibility 字段
- **WHEN** MMW sensor-assisted summary 写出
- **THEN** 每个 run record MUST 包含 `main_conclusion_eligible`、`eligibility_reasons`、enabled sensing modalities、excluded sensitive fields 和 sensitive usage flags
- **AND** summary MUST 提供可过滤主结论 run 与补充/诊断 run 的机器可读字段

### Requirement: MMW sensor-assisted 不依赖 Hist
当前 MMW 主线 MUST 不依赖 HiST-Beam/Hist 模型、Hist LOSO executor 或 `configs/hist_beam/`。若保留 MMW sensor-assisted 或 GPS adapter workflow，必须通过非 Hist CLI、配置和模型注册名定义输入、输出和评估。

#### Scenario: MMW 当前主线不构建 Hist 模型
- **WHEN** 用户运行当前 MMW Town GPS v2、GPS adapter、candidate 或其它保留 MMW workflow
- **THEN** 系统 MUST 不构建 `hist_beam_fusion`
- **AND** 输出 metadata MUST 不声明 HiST-Beam variant

#### Scenario: 旧 sensor-assisted Hist 配置不可运行
- **WHEN** 用户引用 `configs/hist_beam/mmw_sensor_assisted_quick_validation.yaml` 或等价 Hist sensor-assisted 配置
- **THEN** 系统 MUST 报告该 Hist 配置已退役或不存在
- **AND** 系统 MUST 不生成 Hist LOSO plan

### Requirement: Physics-informed baseline eligibility boundary
系统 MUST 区分 MMW sensor-assisted 主结论 run 与 physics-informed research/diagnostic run。任何将当前完整 CSI/channel、path params、beam power、beamspace power 或 radio/path semantic label 作为模型输入或 target-side 训练监督的 run MUST 在 metadata 中标记 sensitive usage，并且 MUST 不自动进入 sensor-assisted 主结论集合。受限 CSI 输入 MUST 明确标记为 `csi_observed` profile，而不是当前完整 CSI。

#### Scenario: 当前完整 CSI 作为输入只能是 oracle
- **WHEN** physics-informed 配置启用 `csi_input_mode=oracle_full`
- **THEN** run metadata MUST 设置 `used_csi_as_input=true`
- **AND** run metadata MUST 设置 `used_current_full_csi_as_input=true`
- **AND** `main_conclusion_eligible` MUST 为 false
- **AND** summary MUST 保留该 run 作为 oracle upper-bound baseline 或 supplementary result

#### Scenario: 受限 CSI 输入可审计
- **WHEN** physics-informed 配置启用 `csi_input_mode=history`、`partial`、`noisy` 或 `compressed`
- **THEN** run metadata MUST 记录 `csi_input_mode`
- **AND** summary MUST 明确该输入为受限 `csi_observed` 或历史 CSI
- **AND** 系统 MUST 不把当前完整 `csi_target` 传入模型 forward

#### Scenario: target path 监督排除主结论
- **WHEN** target adaptation 使用 target-side path params 或 path descriptor 计算训练 loss
- **THEN** metadata MUST 设置 `used_target_path_label_for_training=true`
- **AND** summary MUST 记录 exclusion reason
- **AND** 该 run MUST 不进入 sensor-assisted adapted-vs-source 主结论比较

#### Scenario: source-only 物理监督不污染 target eligibility
- **WHEN** physics-informed run 只在 source split 使用 CSI、path 或 beam power 物理监督
- **THEN** metadata MUST 记录 source supervision fields
- **AND** target-side sensitive usage flags MUST 保持 false
- **AND** summary MUST 仍明确该 run 的 profile 为 physics-informed 而不是纯 sensor-assisted input profile

