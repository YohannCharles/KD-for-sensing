## ADDED Requirements

### Requirement: MMW beam label calibration 配置
系统 MUST 为 MMW dataset 提供显式 opt-in 的 beam label calibration 配置。默认未启用时，系统 MUST 保持 raw beam label 语义；启用时，系统 MUST 使用可逆 mapping 将 raw beam label 映射到 calibrated beam label space。

#### Scenario: 默认保持 raw label
- **WHEN** 用户加载 `data.dataset.type: mmw` 且未启用 `data.dataset.beam_label_calibration.enabled`
- **THEN** dataset MUST 按现有 raw `argmax` label 返回 `input_beam` 和 `target_beam`
- **AND** run metadata MUST 将 `beam_label_space` 记录为 `raw`

#### Scenario: affine calibration 生效
- **WHEN** 用户配置 `beam_label_calibration.enabled=true`、`num_classes=64`、`direction=-1` 和 `offset=52`
- **THEN** 系统 MUST 按 `(direction * raw + offset) mod 64` 计算 calibrated label
- **AND** inverse mapping MUST 能将 calibrated label 还原到 raw label

#### Scenario: 非 MMW dataset 不受影响
- **WHEN** 用户加载 DeepSense6G、Raymobtime 或 synthetic dataset 配置
- **THEN** MMW beam label calibration MUST 不改变这些 dataset 的 label 解析结果
- **AND** 系统 MUST 不要求这些 dataset 提供 MMW calibration metadata

### Requirement: calibrated label provenance
系统 MUST 在启用 calibration 的样本、run metadata、prediction artifact 和诊断 artifact 中记录 label space、mapping 参数、mapping fingerprint 和 raw/calibrated label 可追溯信息。

#### Scenario: 样本 metadata 保留 raw 与 calibrated label
- **WHEN** MMW dataset 启用 beam label calibration 并返回一个样本
- **THEN** 样本 metadata MUST 记录当前 `beam_label_space` 和 mapping fingerprint
- **AND** metadata MUST 能追溯 raw `input_beam`/`target_beam` 与 calibrated `input_beam`/`target_beam` 的对应关系

#### Scenario: prediction artifact 声明 label space
- **WHEN** evaluation 或 viewer prediction export 写出 prediction CSV/JSON
- **THEN** artifact MUST 记录预测 class 所属的 label space
- **AND** 若写出 raw label 或 inverse-mapped label，artifact MUST 明确字段名和 mapping 来源

### Requirement: class-indexed 分布重排
系统 MUST 对所有作为 beam class distribution 消费的 64 维标签分布应用同一 raw→calibrated class order 重排。重排规则 MUST 为 `distribution_calibrated[mapping(raw)] = distribution_raw[raw]`。

#### Scenario: soft target 分布重排
- **WHEN** source-domain soft target 从 raw beam power vector 构造并启用 calibration
- **THEN** `target_beam_distribution` 的 class 维 MUST 按 calibrated label space 排列
- **AND** `argmax(target_beam_distribution)` MUST 与 calibrated hard `target_beam` 一致，除非配置的 smoothing 明确改变 argmax

#### Scenario: sensing mmwave 输入不重排
- **WHEN** MMW dataset 启用 mmWave modality 并启用 beam label calibration
- **THEN** 样本中的 `mmwave` sensing feature MUST 保持原始 power vector 顺序和 shape
- **AND** 只有当同一 power vector 被声明为 label distribution、physical label 或 metric reference 时才按 calibration 重排

### Requirement: calibration fitting 边界
系统 MAY 支持从诊断或 support split 拟合 calibration 参数，但训练和 target adaptation MUST NOT 使用 target_test label 拟合训练期 calibration。

#### Scenario: target support 拟合可审计
- **WHEN** calibration 参数来自 labeled target support split
- **THEN** metadata MUST 记录 fitted split、样本数、参数和算法版本
- **AND** target_test label MUST NOT 参与该参数拟合

#### Scenario: target_test 只用于离线诊断
- **WHEN** 离线诊断读取 target_test label 评估 calibration 质量
- **THEN** artifact MUST 标记该读取为 offline diagnostics scope
- **AND** 该 artifact MUST NOT 被训练或 adaptation 作为监督输入消费
