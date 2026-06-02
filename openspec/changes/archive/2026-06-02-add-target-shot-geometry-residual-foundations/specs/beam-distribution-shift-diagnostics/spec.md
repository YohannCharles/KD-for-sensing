## ADDED Requirements

### Requirement: source-target label histogram 诊断
系统 MUST 提供无需训练模型即可运行的分布诊断命令，用于输出 source、target_labeled、target_unlabeled 可选和 target_test 的 absolute beam、geometry beam 与 residual beam histogram。

#### Scenario: 输出基础 histogram
- **WHEN** 用户运行 distribution shift analysis 命令并传入配置与 split artifact
- **THEN** 系统 MUST 写出 source 和 target_test 的 absolute beam histogram
- **AND** 若 geometry-residual label 可用，系统 MUST 写出 geometry beam 与 residual beam histogram
- **AND** 若 target_labeled 存在，系统 MUST 单独写出 target_labeled histogram

#### Scenario: target_unlabeled 不暴露监督标签给训练
- **WHEN** 诊断命令为了统计目的读取 target_unlabeled label
- **THEN** 该读取 MUST 标记为 offline diagnostics scope
- **AND** 诊断 artifact MUST 不被 adaptation/training 作为监督输入消费

### Requirement: 分布距离指标
系统 MUST 计算 source 与 target split 之间的 KL divergence、JS divergence、Wasserstein/Earth Mover distance 和 total variation distance。指标 MUST 对空 bin 做平滑，且 ordered beam/residual distance MUST 保持 beam class 的顺序或 circular ordering 语义。

#### Scenario: 计算 absolute 与 residual EMD
- **WHEN** source 与 target_test absolute/residual histogram 均可用
- **THEN** 诊断命令 MUST 输出 `emd_absolute` 和 `emd_residual`
- **AND** 输出 MUST 说明 residual histogram 使用的 residual convention 和 class order

#### Scenario: histogram 有空 bin
- **WHEN** 某个 split 的 histogram 存在空 bin
- **THEN** KL 和 JS 计算 MUST 使用配置化 smoothing
- **AND** smoothing 系数 MUST 写入 metrics JSON

### Requirement: 诊断产物结构
系统 MUST 将分布诊断保存为机器可读 JSON/CSV，并 MAY 生成 PNG/PDF 图。结构化产物 MUST 至少包含 histogram、距离指标、split metadata 路径、输入 fingerprint、label_space 配置和 unavailable reason。

#### Scenario: 写出 JSON 和 CSV
- **WHEN** distribution shift analysis 成功完成
- **THEN** 输出目录 MUST 包含 `distribution_shift_metrics.json`
- **AND** 输出目录 MUST 包含可表格读取的 histogram CSV 或 JSON
- **AND** metrics JSON MUST 包含 source/target 样本数与 split artifact 路径

#### Scenario: 可视化依赖不可用
- **WHEN** matplotlib 或等价可视化依赖不可用且用户未要求图片为 required
- **THEN** 诊断命令 MUST 继续写出 JSON/CSV
- **AND** metadata MUST 记录 figure generation skipped reason

### Requirement: 分布诊断解释字段
系统 MUST 在 summary 中明确比较 absolute label distribution shift 与 residual label distribution shift。若 `emd_residual < emd_absolute`，系统 MUST 只报告该诊断事实，不得自动声明模型性能提升。

#### Scenario: residual EMD 更小
- **WHEN** 诊断结果显示 residual EMD 小于 absolute EMD
- **THEN** summary MUST 记录 residual label space 减小了当前 split 的标签分布距离
- **AND** summary MUST 明确这不是模型泛化提升的充分证据
