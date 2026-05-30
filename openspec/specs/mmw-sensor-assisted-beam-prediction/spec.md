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

### Requirement: 快速验证实验矩阵
MMW sensor-assisted quick validation MUST 使用小矩阵进行快速反馈。默认矩阵 MUST 只包含 `label_budget=10` 和 2 个 seeds，除非用户显式选择 full matrix。默认 quick/smoke 配置 MUST 使用 `seq_len=5` 和 `num_pred=3`，以保留较短历史上下文并弱化 next-frame last-beam shortcut 对主结论的影响。

#### Scenario: 默认 budget 和 seeds
- **WHEN** 用户加载 sensor-assisted quick validation 配置或 LOSO matrix
- **THEN** `loso.budgets` MUST 默认为 `[10]`
- **AND** `loso.seeds` MUST 默认为两个 seed，建议为 `[0, 1]`
- **AND** `data.dataset.seq_len`、`model.seq_length_teacher` 和 `model.seq_length_student` MUST 默认为 `5`
- **AND** `data.dataset.num_pred`、`model.num_pred` 和 `model.student.num_pred` MUST 默认为 `3`
- **AND** plan metadata MUST 记录该矩阵为 quick validation，不等价于完整 budget/seed sweep

#### Scenario: 覆盖关键变体
- **WHEN** 生成 sensor-assisted quick validation run plan
- **THEN** plan MUST 至少覆盖 source-only、adapter-only、V6 radio prototype、V8 path prototype、path condition off 和 full fine-tuning baseline 中可用的变体
- **AND** 每个 run MUST 记录 variant、budget、seed、target scene、source scenes 和 modality profile

#### Scenario: 防止误跑大矩阵
- **WHEN** 用户运行默认 sensor-assisted quick validation 命令
- **THEN** runner MUST NOT 默默扩展到 budgets `[0,5,10,20,50]` 或 3 个以上 seeds
- **AND** 若用户通过 CLI override 扩展矩阵，plan metadata MUST 记录 override 来源

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

