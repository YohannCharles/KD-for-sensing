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

### Requirement: MMW quick validation run eligibility audit
MMW sensor-assisted quick validation MUST audit run eligibility using split leakage diagnostics and actual target-side oracle usage. A run MUST NOT be marked ineligible only because optional path/radio supervision fields exist in the dataset when those fields were not consumed by adaptation, model selection, threshold selection, temperature fitting, prototype update or training loss.

#### Scenario: 无 oracle 的 sensor-assisted run 可进入主结论
- **WHEN** quick validation run 使用 strict-validation eligible split，且 adaptation 只读取 sensing inputs 与允许的 target_adapt labeled support beam labels
- **THEN** run metadata MUST 标记该 run 可用于 sensor-assisted 主结论
- **AND** summary MUST NOT 因数据集中存在 path/radio/channel 文件而把该 run 排除
- **AND** metadata MUST 记录 `used_target_oracle_fields=[]` 或等价空列表

#### Scenario: 使用禁用 target oracle 的 run 被排除
- **WHEN** run 在 adaptation、threshold selection、temperature fitting、prototype update、early stopping 或 loss 计算中读取 target_test label、target_test beam_power、target_test path fields、target-side radio/channel labels 或禁用 oracle 字段
- **THEN** run metadata MUST 标记该 run 不可用于主结论
- **AND** summary MUST 将该 run 归入 excluded/debug results
- **AND** exclusion reason MUST 包含实际使用的字段、使用阶段和机器可读 reason code

#### Scenario: eligibility reason 可审计
- **WHEN** quick validation summary 计算 eligible run count
- **THEN** summary MUST 为每个 excluded run 记录 `eligibility_status`、`eligibility_reasons`、split diagnostics path 和 oracle usage summary
- **AND** `eligible_run_count=0` 时 summary MUST 能解释是 split 不严格、oracle 使用违规、产物缺失还是 validator 条件无法判定

### Requirement: MMW target oracle usage metadata
MMW quick validation run artifacts MUST 记录 target-side field usage，使 sensor-assisted、mmWave-assisted、path/radio-assisted 和 debug run 可以被机器过滤。该记录 MUST 覆盖配置声明与运行时实际消费字段。

#### Scenario: 记录允许与禁用字段
- **WHEN** run 启动或完成 adaptation/evaluation
- **THEN** run metadata MUST 记录 enabled sensing modalities、excluded target oracle fields、allowed target labels 和实际 consumed fields
- **AND** metadata MUST 区分 `target_adapt_labeled_support_label`、`target_adapt_unlabeled_input`、`target_test_evaluation_label` 与禁用 target-side path/radio/channel 字段

#### Scenario: target_test 标签只用于最终评价
- **WHEN** target_test label 被用于计算最终 Top-K、within3、MAE、histogram、KL 或 confusion artifact
- **THEN** metadata MUST 将该用途标记为 `evaluation_only`
- **AND** eligibility checker MUST NOT 因 evaluation-only target_test label 使用而排除 run
- **AND** 该标签 MUST NOT 出现在 adaptation optimizer、prior 初始化、prototype update、threshold selection、temperature fitting 或 early stopping 记录中

#### Scenario: 运行时字段消费未知时保守排除
- **WHEN** run metadata 无法证明 target-side oracle 字段未被 adaptation 或选择逻辑消费
- **THEN** eligibility checker MUST 将该 run 标记为 `unknown_oracle_usage`
- **AND** summary MUST 将该 run 排除出主结论
- **AND** exclusion reason MUST 指向缺失的 metadata 或审计字段

### Requirement: MMW v9 quick validation protocol metadata
MMW quick validation protocol MUST 能声明 v9 input-conditioned target adaptation 实验矩阵，并 MUST 将 eligibility、collapse diagnostics 和 prototype diagnostics 接入 summary。

#### Scenario: v9 quick validation plan metadata
- **WHEN** planner builds v9 quick validation plan
- **THEN** plan metadata MUST include profile、source scenes、target scene、budgets、seeds、v9 group ids 和每个 mode 的实验目的
- **AND** 默认 v9 quick validation MUST 限定为小矩阵，不得静默扩展到完整 budget/seed sweep

#### Scenario: v9 summary 汇总 collapse diagnostics
- **WHEN** v9 run 完成 source-only 或 adapted target_test evaluation
- **THEN** summary MUST 引用 `prediction_hist.json` 和 `collapse_diagnostics.json` 或等价 artifact path
- **AND** summary MUST 汇总 Top-K、within3、MAE、unique predicted beams、pred top beams、histogram KL 和 beta/prototype diagnostics

#### Scenario: Group C 需要 protocol 许可
- **WHEN** v9 Group C 使用未标注 target_adapt 做 distribution regularization 或 consistency loss
- **THEN** plan/run metadata MUST 证明未标注样本来自 target_adapt 且不包含 target_test
- **AND** metadata MUST 证明 target_test label、beam_power、path fields 和 radio labels 未参与训练或选择
- **AND** 若证明缺失，Group C run MUST 被标记为 disabled、debug 或 ineligible

### Requirement: Image-only quick validation eligibility audit
MMW quick validation eligibility audit MUST judge image-only legal probe runs by strict split eligibility and actual target-side oracle consumption. A run MUST NOT be marked ineligible only because raw dataset files or manifest rows contain path、radio、channel、beam_power、GPS、LiDAR or other disabled fields when those fields were not consumed by model input、loss、adaptation、threshold selection、temperature fitting、prototype update、early stopping or summary selection.

#### Scenario: 合法 image-only run 不因原始字段存在被排除
- **WHEN** image-only probe run 使用 strict-validation eligible split
- **AND** consumed fields 只包含 image 输入和允许的 beam labels
- **THEN** eligibility metadata MUST 记录 `target_oracle_fields_used=false`
- **AND** eligibility metadata MUST 记录 `target_radio_label_supervision=false`
- **AND** eligibility metadata MUST 记录 `target_path_label_supervision=false`
- **AND** summary MUST NOT 因 raw dataset 中存在 path、radio、channel 或 beam_power 字段而排除该 run

#### Scenario: 禁用 oracle 实际被消费时排除
- **WHEN** image-only probe run 在 adaptation、threshold selection、temperature fitting、prototype update、early stopping、loss 计算或 summary selection 中消费 target_test label、target_test beam_power、target-side path/radio/channel label 或禁用 oracle 字段
- **THEN** eligibility metadata MUST 将 run 标记为 ineligible
- **AND** `eligibility_reasons` MUST 包含实际字段名、使用阶段和机器可读 reason code
- **AND** summary MUST 将该 run 排除出主结论

### Requirement: Image-only split eligibility 明确化
MMW image-only legal probe MUST 明确记录 split eligibility。系统 MUST NOT 默默输出 `split_eligibility_unknown`；当无法判断 split 是否严格合法时，eligibility metadata MUST 给出缺失 metadata、config path 或 leakage diagnostic path。

#### Scenario: split metadata 完整时可进入主结论
- **WHEN** source、target_support 和 target_test split metadata 能证明 sample id、窗口上下文和 guard-band 约束满足 strict validation
- **THEN** run metadata MUST 记录 `split_eligibility_unknown=false`
- **AND** run metadata MUST 记录 strict split eligibility 的诊断路径或摘要

#### Scenario: split metadata 缺失时给出具体原因
- **WHEN** eligibility checker 无法判断 split eligibility
- **THEN** run metadata MUST 记录 `split_eligibility_unknown=true`
- **AND** `eligibility_reasons` MUST 包含缺失字段、缺失文件或配置路径
- **AND** summary MUST 将该 run 标记为 excluded/debug，而不是把 unknown 当成 eligible

### Requirement: Image-only oracle usage metadata
MMW image-only legal probe MUST 在 run metadata 中记录 enabled modalities、disabled modalities、excluded sensitive fields、consumed fields 和 stage-level oracle usage summary。该 metadata MUST 能支持 downstream report 过滤合法 image-only run。

#### Scenario: metadata 记录模态和禁用字段
- **WHEN** image-only probe run 启动
- **THEN** run metadata MUST 记录 `enabled_modalities=["image"]`
- **AND** run metadata MUST 记录 disabled modalities
- **AND** run metadata MUST 记录 excluded sensitive fields
- **AND** run metadata MUST 记录 `used_target_oracle_fields=[]`，除非实际消费了禁用 target-side oracle 字段

#### Scenario: metadata 记录 stage-level consumed fields
- **WHEN** image-only probe run 完成 source training、target adaptation 或 target_test evaluation stage
- **THEN** run metadata MUST 记录每个 stage 的 consumed input fields 和 consumed label fields
- **AND** target adaptation stage MUST 仅记录 target support image 和 support beam label 作为合法 consumed fields
- **AND** target_test evaluation stage MUST 标记 target_test beam label 仅用于 final metrics

### Requirement: Image-only quick validation conclusion
MMW quick validation summary MUST 为 image-only legal probe 输出机器可读结论。结论 MUST 汇总 eligible run count、ineligible reasons、target oracle flags、split eligibility flags 和各 probe mode 的核心指标。

#### Scenario: eligible run count 大于零
- **WHEN** image-only probe summary 生成
- **THEN** summary MUST 输出 `eligible_run_count`
- **AND** 只有满足 strict split eligibility 且未消费禁用 target oracle 的 run 才能计入 eligible count

#### Scenario: ineligible run 说明可定位
- **WHEN** 任一 image-only probe run 被排除
- **THEN** summary MUST 记录 mode、run directory、eligibility_status、eligibility_reasons、split diagnostics path 和 oracle usage summary
- **AND** reason MUST 足以定位到对应 config path、artifact path 或 stage

### Requirement: MMW 5% target-shot split artifact
MMW cross-scene adaptation protocol MUST support a 5% target-shot split artifact for scenario-level、town-level 和 weather/condition-level experiments. The artifact MUST be derived from MMW availability/manifest metadata and MUST preserve existing group-safe window leakage diagnostics.

#### Scenario: MMW scenario target-shot split
- **WHEN** MMW manifest 包含至少一个 source scenario 和一个 target scenario
- **THEN** split builder MUST 能生成 source、target_labeled、target_unlabeled 和 target_test split
- **AND** target_labeled MUST 默认占 target adaptation pool 的 5%
- **AND** split metadata MUST 保留 sample id、window overlap、guard band 和 strict eligibility diagnostics

#### Scenario: MMW weather target-shot split
- **WHEN** MMW availability 包含多个 weather/condition 且配置选择 condition-level target domain
- **THEN** split builder MUST 将 target condition 与 source condition 写入 metadata
- **AND** summary MUST 不把缺少其它 condition 的 run 声称为 weather-shift 验证

### Requirement: MMW geometry-residual label 统计
MMW cross-scene adaptation protocol MUST support geometry-residual label statistics using direct RSU-CAV relative geometry when available. The protocol MUST record whether `beam_geo` is derived from direct geometry, uniform angle quantization, codebook mapping or unavailable.

#### Scenario: MMW manifest geometry 可用
- **WHEN** MMW frame manifest 包含 direct relative azimuth 或可解析 RSU/CAV pose
- **THEN** geometry-residual label builder MUST 能生成 `beam_geo`、`beam_residual` 和 `geo_sector`
- **AND** split diagnostics MUST 写出 source/target absolute 与 residual histogram

#### Scenario: MMW geometry 不可用
- **WHEN** 某个 MMW sample 缺少 direct geometry 且配置要求 geometry residual
- **THEN** 系统 MUST 按 `label_space.geometry.required` 决定失败或标记 unavailable
- **AND** unavailable reason MUST 写入 manifest 或 diagnostics artifact

### Requirement: MMW target-shot 防 oracle 边界
MMW target-shot adaptation MUST only use `target_labeled` beam/residual labels for supervised target loss. `target_unlabeled` and `target_test` beam labels、beam_power、path fields、radio labels and channel-derived labels MUST NOT be used for adaptation threshold selection、prototype update、temperature fitting、early stopping or training loss.

#### Scenario: target_test label 不参与 calibration
- **WHEN** 后续 calibration 或 adaptation 使用 MMW target-shot split artifact
- **THEN** target_test labels MUST only be available in final evaluation scope
- **AND** eligibility audit MUST mark the run ineligible if target_test labels influence training, threshold, prototype, temperature or early stopping

#### Scenario: target_labeled residual 监督合法
- **WHEN** MMW adaptation 使用 `target_labeled` subset 且 label budget 大于 0
- **THEN** supervised beam 或 residual loss MAY use target_labeled labels
- **AND** usage metadata MUST record selected sample ids and target_label_fraction

### Requirement: MMW Town GPS-only v2 protocols
系统 MUST 为 MMW Town GPS-only v2 支持 `source_other_three`、`target_adapt_beambench` 和 `within_scene_train` 三类协议。协议输出 MUST 明确标记 scene、source scenes、target scene、support/query/test split 和 strict eligibility。

#### Scenario: source_other_three 留一场景评估
- **WHEN** runner 执行 `source_other_three`
- **THEN** 系统 MUST 每次选择一个 MMW Town scene 作为 target
- **AND** 系统 MUST 使用其它三个 scene 训练 source model
- **AND** target scene label MUST 只用于最终评估

#### Scenario: target_adapt_beambench 使用 support/query
- **WHEN** runner 执行 `target_adapt_beambench`
- **THEN** 系统 MUST 使用其它三个 scene 训练 source backbone
- **AND** target support set MUST 只来自 target adaptation pool
- **AND** target query/test MUST 不参与 adapter 初始化、优化或模型选择

#### Scenario: within_scene_train 标记为上界
- **WHEN** runner 执行 `within_scene_train`
- **THEN** 系统 MUST 将该 protocol 标记为同场景上界或 sanity protocol
- **AND** summary MUST NOT 将其作为跨场景泛化结论

### Requirement: MMW Town GPS v2 support selection
系统 MUST 为 v2 target adaptation 支持 `temporal_first`、`random` 和 `trajectory` support selection。默认 support mode MUST 为 `temporal_first`，support 数量可由 `support_ratio` 或 `support_num` 控制。

#### Scenario: temporal_first 默认支持集
- **WHEN** 用户未显式指定 support mode
- **THEN** 系统 MUST 使用 `temporal_first`
- **AND** support manifest MUST 记录 sample id、timestamp/order key、target label、scene、selection mode 和 seed

#### Scenario: support_num 覆盖 support_ratio
- **WHEN** 用户同时配置 `support_num` 和 `support_ratio`
- **THEN** 系统 MUST 使用 `support_num` 或抛出清晰的优先级错误
- **AND** metadata MUST 记录实际使用的 support count

