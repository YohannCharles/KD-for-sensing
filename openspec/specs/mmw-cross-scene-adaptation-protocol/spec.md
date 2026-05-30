# mmw-cross-scene-adaptation-protocol Specification

## Purpose
定义 MMW Town10 跨场景适配的数据可用性、manifest、split protocol 和 target adaptation 契约，确保单场景 smoke 与可声明的 cross-scene LOSO 实验被清晰区分，并让 MMW HiST-Beam 训练、评估和报告可复现、可审计。
## Requirements
### Requirement: MMW 跨场景适配数据可用性登记
系统 MUST 为 Multimodal-Wireless 跨场景适配维护机器可读的数据可用性登记。登记 MUST 按 condition、town、scenario、sensor zip、channel zip、prepared root 和处理状态记录每个本地数据单元；当只有一个 scenario 可用时，系统 MUST 将其标记为 single-scene smoke 可用，而不是 LOSO 可用。

#### Scenario: sunny 单场景已准备
- **WHEN** 用户已处理 `dataset/_downloads/MMW/sunny` 中的 `Town10_skybridge_seed24.zip` 和 `Town10.zip`
- **THEN** 数据可用性登记 MUST 记录 condition 为 `sunny`、town 为 `Town10`、scenario 为 `Town10_skybridge_seed24`
- **AND** 登记 MUST 记录 prepared root、frame/window 数、zip fingerprint 和状态 `single_scene_ready`
- **AND** 系统 MUST 不生成声称跨场景的 LOSO fold

#### Scenario: 下载中的场景保持 pending
- **WHEN** 其它 condition、town 或 scenario 的 sensor/channel zip 尚未全部存在
- **THEN** 数据可用性登记 MUST 将对应数据单元标记为 `pending`
- **AND** planner MUST 跳过该数据单元并在 plan metadata 中记录跳过原因

### Requirement: MMW enriched frame manifest
系统 MUST 为 MMW cross-scene adaptation 生成 enriched frame manifest。manifest MUST 以 condition、town、scenario、agent 和 frame id 构造稳定 sample id，并 MUST 记录 CAV/RSU 传感器路径、V2I channel 路径、beam power、beam label、coarse sector、模态可用性和几何派生字段。

#### Scenario: 记录 CAV 与 RSU 同步模态
- **WHEN** 某个 CAV frame 与同 frame id 的 RSU frame 均存在
- **THEN** frame manifest MUST 记录 CAV LiDAR、四路 RGB camera、可用 depth camera、GPS/IMU yaml、bbox、radar point cloud 路径
- **AND** frame manifest MUST 记录匹配 RSU 的 pose yaml、LiDAR、camera/depth/radar 可用路径
- **AND** 缺失但未启用的模态 MUST 记录为 unavailable 而不是导致样本失败

#### Scenario: 记录 channel 与 beam 标签
- **WHEN** frame manifest 匹配到 V2I channel `_paths.npy` 或 `_paths.npz`
- **THEN** 系统 MUST 记录 channel 文件路径、派生 beam power 文件、beam label、coarse sector 和 channel 字段摘要
- **AND** beam power vector MUST 为固定长度 finite 数组
- **AND** 不可派生的 channel 文件 MUST 以 skip reason 记录

### Requirement: MMW 几何字段与 proxy 标签
系统 MUST 从 MMW frame manifest 派生可训练几何字段，并 MUST 标明每个字段是 direct observable 还是 proxy。直接可观测字段包括 RSU-CAV relative geometry、beam/codebook angular fields 和 channel-derived fields；scatterer/occluder、town layout 和局部遮挡语义 MUST 标记为 proxy。

#### Scenario: 派生 RSU-CAV 相对几何
- **WHEN** CAV 和 RSU pose metadata 可解析
- **THEN** 系统 MUST 派生 relative range、azimuth、elevation、heading difference 和 local-frame 坐标
- **AND** 这些字段 MUST 标记为 direct geometry
- **AND** 派生失败时 MUST 记录失败原因并允许样本在不启用 geometry loss 时继续使用

#### Scenario: 标记 occluder proxy
- **WHEN** 系统从 LiDAR occupancy、bbox、depth discontinuity、radar point density 或 channel path count 派生 occluder/scatterer 特征
- **THEN** metadata MUST 将这些字段标记为 proxy
- **AND** 训练和报告 MUST 不把这些 proxy 描述为真实 scatterer 或真实 occluder 标签

### Requirement: MMW scenario/town/weather split
系统 MUST 支持基于 MMW 数据可用性登记生成 scenario-level、town-level 和 weather/condition-level source-target split。split MUST 保留 target_adapt/target_test 防泄漏约束，并 MUST 支持单场景 smoke、scenario-LOSO、leave-one-town-out 和 leave-one-condition-out 四类 protocol。对于基于滑窗的 MMW sequence 数据，source、target_adapt 和 target_test 的隔离 MUST 不只检查 sample id，还 MUST 检查时间窗口上下文、frame overlap 和 guard band，避免相邻窗口跨 split 泄漏。

#### Scenario: 单场景只生成 smoke split
- **WHEN** 可用 MMW scenario 少于两个
- **THEN** planner MUST 只生成 single-scene smoke 或 within-scenario sanity split
- **AND** plan metadata MUST 标明该 split 不可用于跨场景结论

#### Scenario: scenario-LOSO split
- **WHEN** 同一 town/condition 下至少两个 scenario 可用
- **THEN** planner MUST 能选择一个 scenario 作为 target，其余 scenario 作为 source
- **AND** target scenario MUST 被确定性拆分为 target_adapt 和 target_test
- **AND** source、target_adapt 和 target_test 的 sample id MUST 无交集
- **AND** target_adapt 与 target_test MUST 使用 group-safe 或等价防泄漏协议，避免共享完整历史+未来窗口 frame id
- **AND** split metadata MUST 记录 target_adapt/target_test 的 guard band、group key、window overlap diagnostics 和 strict eligibility

#### Scenario: condition-level split
- **WHEN** sunny、rainy、foggy 或其它 condition 中至少两个 condition 可用
- **THEN** planner MUST 能将一个 condition 作为 target condition
- **AND** source condition 与 target condition MUST 在 metadata 中明确记录

### Requirement: MMW target adaptation protocol
系统 MUST 为 MMW 跨场景适配定义 target adaptation protocol。protocol MUST 支持 label budgets `0`、`5`、`10`、`20` 和 `50`，并 MUST 优先按 coarse sector 与 relative azimuth bin 进行 labeled subset 分层采样。

#### Scenario: 0-label target adaptation
- **WHEN** label budget 为 `0`
- **THEN** 系统 MUST 不读取 target_adapt 标签作为 supervised loss
- **AND** 系统 MAY 使用 entropy、angular consistency 和高置信 private prototype loss
- **AND** target_test MUST 不参与 threshold、prototype 或 normalizer 选择

#### Scenario: few-shot target adaptation
- **WHEN** label budget 大于 `0`
- **THEN** 系统 MUST 从 target_adapt 中选择 labeled subset
- **AND** 采样 manifest MUST 记录 sample id、beam label、coarse sector、relative azimuth bin 和采样 seed
- **AND** 未选中的 target_adapt 样本 MAY 作为 unlabeled adaptation 数据

### Requirement: MMW 论文声明边界
系统 MUST 在 MMW adaptation report 和 summary metadata 中区分可直接实现的论文概念和 proxy 概念。任何由 proxy 支撑的概念 MUST 在输出中带有 proxy 标记。

#### Scenario: 输出 direct/proxy 摘要
- **WHEN** MMW adaptation summary 写出
- **THEN** summary MUST 包含 direct fields 列表和 proxy fields 列表
- **AND** 若某个实验只使用 proxy occluder/scatterer 特征，summary MUST 明确标记为 proxy-based

### Requirement: MMW Sionna path 数据可用性与字段映射
系统 MUST 在 MMW 数据准备、manifest 或巡检阶段记录 Sionna channel/path 文件中的 path-level physical propagation parameters 可用性。字段识别 MUST 支持配置化 `data.field_map`，不得依赖单一硬编码字段名。

#### Scenario: 记录 path-level 字段摘要
- **WHEN** frame manifest 或 inspect 工具匹配到 V2I channel/path 文件
- **THEN** 系统 MUST 记录是否存在 path gain、delay、AoD/AoA azimuth/zenith、valid path mask 和 optional Tx/Rx/CAV/RSU pose
- **AND** 系统 MUST 记录字段 shape、dtype 或等价摘要
- **AND** 系统 MUST 不把原始 path tensor 复制进源码控制的 manifest

#### Scenario: 使用 data.field_map 覆盖字段名
- **WHEN** Sionna path 文件字段名与默认候选不一致
- **THEN** 用户 MUST 能通过 `data.field_map` 指定 gain、delay、AoD/AoA、mask 和 pose 字段映射
- **AND** 系统 MUST 在 metadata 中记录实际使用的字段映射

#### Scenario: path 数据缺失时保留样本可诊断
- **WHEN** 某个样本缺少 path-level parameters 但仍有合法 sensing inputs 和 beam label
- **THEN** 系统 MUST 允许该样本继续用于不依赖 path supervision 的训练或评估
- **AND** manifest 或 dataset metadata MUST 记录 path unavailable reason

### Requirement: MMW path-level split 与评估边界
MMW scenario/town/weather split MUST 将 path-level labels 和 descriptors 视为 auxiliary target 或 diagnostic data，而不是 sensing input。target adaptation 与 target_test 的防泄漏边界 MUST 对 path fields 生效。

#### Scenario: source split 可构造 path labels
- **WHEN** source train split 有可用 path parameters
- **THEN** 系统 MAY 基于 source path descriptors fit path semantic label artifact
- **AND** artifact MUST 记录 source town/scenario/weather、fit sample count 和 unavailable count

#### Scenario: target test path labels 只用于离线评价
- **WHEN** target test 样本可构造 path_semantic_label 或 path_descriptor
- **THEN** evaluation MAY 使用这些字段计算 path diagnostics
- **AND** target adaptation MUST NOT 使用 target test path fields 选择 threshold、更新 prototype 或计算训练 loss

#### Scenario: leave-one-town/scenario/weather 报告 path 分布
- **WHEN** 系统生成 MMW LOSO summary
- **THEN** summary MUST 能报告 source-target path class histogram 或 unavailable reason
- **AND** summary MUST 将 leave-one-town-out、leave-one-scenario-out 和 weather-shift protocol 的 path diagnostics 与 run metadata 关联

### Requirement: MMW sensor-assisted quick validation protocol
MMW cross-scene adaptation protocol MUST allow a sensor-assisted quick validation mode for rapid iteration. This mode MUST be machine-readable in plan metadata and MUST restrict the default matrix to `label_budget=10` and two seeds. Quick validation MUST consume split eligibility metadata; unknown 或高重叠 split 只能作为 debug/sanity 运行，不得作为主结论。

#### Scenario: quick validation matrix metadata
- **WHEN** planner builds an MMW sensor-assisted quick validation plan
- **THEN** plan metadata MUST include `profile=sensor_assisted_quick_validation` or equivalent machine-readable marker
- **AND** plan metadata MUST include `budgets=[10]`
- **AND** plan metadata MUST include exactly two seeds unless explicitly overridden
- **AND** plan metadata MUST record that results are quick validation rather than full budget sweep

#### Scenario: scenario-LOSO remains the only claim with current data
- **WHEN** local MMW availability contains only sunny/Town10 ready scenarios
- **THEN** sensor-assisted quick validation MAY produce scenario-LOSO conclusions
- **AND** it MUST NOT claim leave-one-town-out or weather-shift validation
- **AND** summary MUST record unavailable protocol scope when town/weather data is missing

#### Scenario: source-target split 防泄漏
- **WHEN** sensor-assisted quick validation samples target labeled subset with `label_budget=10`
- **THEN** selected labeled sample ids MUST come only from target_adapt
- **AND** target_test sample ids MUST remain disjoint from source and target_adapt
- **AND** target_test labels, beam_power, path fields and radio labels MUST NOT be used for adaptation threshold selection or training loss
- **AND** target_adapt labeled samples MUST NOT share full or guard-band-protected sliding-window context with target_test samples
- **AND** plan or split metadata MUST expose diagnostics proving the target_adapt/target_test split is strict-validation eligible

#### Scenario: modality profile 可审计
- **WHEN** plan、run metadata 或 summary 写出
- **THEN** metadata MUST include enabled sensing modalities and excluded radio/channel/path fields
- **AND** metadata MUST show whether LiDAR/radar derived cache was used
- **AND** metadata MUST allow downstream reports to filter sensor-assisted runs separately from mmWave-assisted runs

#### Scenario: strict-ineligible split quick validation is excluded
- **WHEN** quick validation run 使用 `strict_validation_eligible=false` 的 split metadata
- **THEN** run metadata 和 summary MUST 标记该 run 不可用于主结论
- **AND** summary MUST 将该 run 归入 debug/sanity 或 excluded results
- **AND** exclusion reason MUST 包含 split strategy 或 leakage diagnostics 中的机器可读原因

### Requirement: History beam split 防泄漏
MMW cross-scene adaptation protocol MUST 在启用 history-anchored prediction 时审计历史 beam 窗口来源。`input_beam` MUST 只来自样本自身预测时刻之前的历史窗口；target future beam、target_test label、beam_power argmax 或任何由 future/channel/path/radio 派生的目标字段 MUST NOT 被写入模型历史输入。

#### Scenario: target_adapt 与 target_test 历史窗口隔离
- **WHEN** planner 生成启用 history-anchored profile 的 scenario-LOSO split
- **THEN** split metadata MUST 检查 target_adapt 和 target_test 不共享受 guard-band 保护的完整预测窗口
- **AND** overlap diagnostics MUST 覆盖用于 `input_beam` 的历史 frame 和用于 future label 的预测 frame
- **AND** strict validation eligibility MUST 反映该检查结果

#### Scenario: input beam 来自历史字段
- **WHEN** dataset 为 history-anchored run 返回 `input_beam`
- **THEN** `input_beam` MUST 来自历史 beam 字段或历史 beam label cache
- **AND** `input_beam` MUST NOT 从当前 future target beam、beam_power vector argmax 或 target_test diagnostic label 反推

#### Scenario: 历史窗口不可审计时排除主结论
- **WHEN** split metadata 无法证明 `input_beam` 与 future label/test label 防泄漏
- **THEN** run metadata 和 summary MUST 标记该 run 不可用于 history-anchored 主结论
- **AND** exclusion reason MUST 包含 `history_window_leakage_unknown` 或等价机器可读原因

### Requirement: History-anchored quick validation protocol
MMW cross-scene adaptation protocol MUST 支持独立的 history-anchored quick validation mode。该 mode MUST 与默认 sensor-assisted quick validation 分开声明、分开汇总，并默认使用一个 source 场景泛化到其它两个 target 场景、两个 seed 和 `label_budget=10` 的最小矩阵。

#### Scenario: 生成 history-anchored quick validation plan
- **WHEN** 用户选择 history-anchored quick validation 配置
- **THEN** plan metadata MUST include `profile=history_anchored_quick_validation` 或等价机器可读标记
- **AND** plan metadata MUST include `budgets=[10]`
- **AND** plan metadata MUST include exactly two seeds unless explicitly overridden
- **AND** plan MUST record source scene、target scenes、history anchor mode 和 residual target mode

#### Scenario: 使用现有 sunny Town10 三场景
- **WHEN** local MMW availability 包含 `Town10_skybridge_seed24`、`Town10_Hroad_seed42` 和 `Town10_crossroad_seed24`
- **THEN** history-anchored quick validation MAY 使用其中一个场景作为 source、其它两个场景作为 target
- **AND** summary MUST 记录这是 scenario-level 泛化，不得声称 leave-one-town-out 或 weather-shift 验证

#### Scenario: quick validation 不自动扩展完整 sweep
- **WHEN** 用户运行默认 history-anchored quick validation 命令
- **THEN** runner MUST NOT 默默扩展到 budgets `[0,5,10,20,50]` 或三个以上 seeds
- **AND** 若用户通过 CLI override 扩展矩阵，plan metadata MUST 记录 override 来源

### Requirement: History-anchored adaptation 防泄漏
history-anchored target adaptation MUST 遵守 MMW target adaptation protocol。`label_budget=0` 不得读取 target future beam 作为 supervised residual loss；`label_budget>0` 只能从 target_adapt labeled subset 读取 future beam label；target_test MUST 始终只用于最终评估。

#### Scenario: 0-label residual adaptation 禁止 target future label
- **WHEN** history-anchored target adaptation 的 `label_budget=0`
- **THEN** adaptation MUST NOT 读取 target_adapt future beam label 计算 residual supervised loss
- **AND** adaptation MAY 使用 entropy、consistency、prototype confidence 或其它不读取 target future label 的无监督 loss
- **AND** metadata MUST 记录 `used_target_beam_for_supervised_loss=false`

#### Scenario: few-shot residual adaptation 只读 labeled target_adapt
- **WHEN** history-anchored target adaptation 的 `label_budget>0`
- **THEN** supervised residual loss MUST 只使用 sampled labeled target_adapt subset
- **AND** sampling manifest MUST 记录 sample id、last_beam、future beam、residual label、coarse sector、relative azimuth bin 和 sampling seed
- **AND** 未选中的 target_adapt 样本 MAY 作为 unlabeled adaptation 数据但 MUST NOT 读取其 future beam label 作为监督

#### Scenario: target_test 只参与最终评价
- **WHEN** history-anchored source-only 或 adapted target_test evaluation 执行
- **THEN** target_test future beam label MAY 用于计算最终 metrics
- **AND** target_test future beam label、beam_power、path fields 和 radio labels MUST NOT 用于 adaptation threshold selection、prototype update、temperature fitting 或 early stopping

### Requirement: History-anchored run metadata
MMW LOSO run artifacts MUST 记录 history-anchored residual 的关键配置和 eligibility 字段，使后续分析可以复现、过滤和诊断 absolute-ID prior collapse。

#### Scenario: run metadata 记录 history 配置
- **WHEN** history-anchored run 完成任一 stage
- **THEN** run metadata MUST 包含 `history_anchor_enabled`、`history_anchor_mode`、`residual_target_enabled`、`num_delta_classes`、`uses_input_beam_as_model_input` 和 enabled sensing modalities
- **AND** metadata MUST 包含 source scenes、target scene、budget、seed、seq_len、num_pred 和 split eligibility

#### Scenario: summary 记录 collapse 诊断输入
- **WHEN** history-anchored summary 汇总 source-only absolute baseline 和 residual run
- **THEN** summary MUST include source train beam histogram、target test beam histogram 和 model predicted beam histogram 或对应 artifact path
- **AND** summary MUST 能据此标记 source prior collapse、history residual recovery 或不可判定原因

#### Scenario: artifacts 保留配置快照
- **WHEN** history-anchored run 启动
- **THEN** run output directory MUST 保存 resolved config 或等价配置快照
- **AND** 配置快照 MUST 包含 history anchor、residual loss、private calibration 和 dataloader split 参数

