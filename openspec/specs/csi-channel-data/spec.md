# csi-channel-data Specification

## Purpose
定义 CSI 序列数据列、样本字段和与多模态数据加载的契约，确保 CSI 数据能按统一模态接口参与训练、诊断和缓存流程。
## Requirements
### Requirement: CSI 序列列与样本字段
启用 CSI 模态时，dataset MUST 从序列 CSV 中读取历史 `csi1..csiN` 路径列，并返回样本字段 `csi`。默认情况下，`csi` MUST 表示 clean CSI 历史输入；当显式启用 `csi_degradation` 时，`csi` MUST 表示由 clean 历史 CSI 派生的 degraded 模型输入。无论是否退化，`csi` MUST 不包含未来 CSI。

#### Scenario: 默认读取 clean CSI 历史序列
- **WHEN** dataset 配置启用 `use_csi: true`、`seq_len: 8` 且未启用 `csi_degradation`
- **THEN** dataset MUST 要求 CSV 包含 `csi1` 到 `csi8`
- **AND** 返回样本 MUST 包含 clean `csi`
- **AND** 返回样本 MUST 不包含未来 CSI 字段

#### Scenario: 显式读取 degraded CSI 历史序列
- **WHEN** dataset 配置启用 `use_csi: true`、`seq_len: 8` 且启用 `csi_degradation`
- **THEN** dataset MUST 先从 `csi1` 到 `csi8` 读取 clean 历史 CSI 来源
- **AND** 返回样本字段 `csi` MUST 是按配置退化后的历史 CSI 模型输入
- **AND** 返回样本 MUST 不读取或包含未来 CSI 字段

#### Scenario: 未启用 CSI 不要求 CSI 列
- **WHEN** 当前任务或 fusion `modalities` 不包含 `csi`
- **THEN** dataset MUST 不要求 CSV 包含 `csi*` 列
- **AND** dataset MUST 不读取任何 CSI 文件

### Requirement: CSI 张量加载与形状规范
系统 MUST 提供 CSI 加载逻辑，将磁盘上的复数 CSI 文件转换为模型输入可消费的稳定张量。输出 MUST 支持 real/imag 末维格式 `[T, Nsc, Nant, 2]`；如果底层文件是 complex tensor，加载后 MUST 显式转换为 real/imag 末维格式。

#### Scenario: 加载 real/imag CSI tensor
- **WHEN** CSI 文件包含形状 `[Nsc, Nant, 2]` 或 `[T, Nsc, Nant, 2]` 的 finite 数值数组
- **THEN** loader MUST 返回 dtype 为 `float32` 的 real/imag 张量
- **AND** 单帧文件组成序列后 MUST 形成 `[seq_len, Nsc, Nant, 2]`

#### Scenario: 加载 complex CSI tensor
- **WHEN** CSI 文件包含 complex dtype 的 `[Nsc, Nant]` 或 `[T, Nsc, Nant]` 数组
- **THEN** loader MUST 将其实部和虚部堆叠到末维
- **AND** 输出 MUST 满足 `[T, Nsc, Nant, 2]` 契约

#### Scenario: 拒绝非法 CSI shape
- **WHEN** CSI 文件缺少复数维度、包含 NaN/Inf 或无法解析出 `[Nsc, Nant]` 信道维度
- **THEN** 系统 MUST 抛出包含文件路径和实际 shape 的错误

### Requirement: CSI 训练集 RMS 统计
系统 MUST 支持在训练 split 上计算 CSI 全局 RMS，并将该统计作为 normalizer artifact 复用到测试 split 和评估入口。CSI RMS MUST 基于 clean CSI 的平均功率计算，不得基于 degraded CSI、pilot noisy CSI 或逐样本归一化结果计算。

#### Scenario: 训练集 fit CSI RMS
- **WHEN** 训练 dataloader 构建时启用 CSI RMS 统计
- **THEN** 系统 MUST 只扫描训练 split 的 clean CSI 样本计算 RMS
- **AND** 计算结果 MUST 能被保存到运行 artifact 或传入后续 test dataset

#### Scenario: 启用 degradation 时仍使用 clean RMS
- **WHEN** 训练 dataloader 同时启用 CSI RMS 统计和 `csi_degradation`
- **THEN** 系统 MUST 使用退化前的 clean CSI 计算 RMS
- **AND** degraded CSI 输出 MUST 使用训练 split 的 clean RMS normalizer

#### Scenario: 测试集复用 CSI RMS
- **WHEN** 构建 test dataset 或运行评估入口
- **THEN** 系统 MUST 复用训练 split 的 CSI RMS
- **AND** 系统 MUST 不在 test split 上重新 fit CSI RMS

### Requirement: CSI batch 输入准备
训练、验证和评估路径 MUST 提供 CSI batch 准备逻辑，从 batch 字段 `csi` 构建模型输入 `csi_batch`。该逻辑 MUST 采用与其它序列模态一致的历史截断和 future zero padding 策略，并 MUST 兼容 clean CSI 与显式配置的 degraded CSI。

#### Scenario: 准备 CSI-only batch
- **WHEN** `experiment.task: csi`、`seq_length: 8` 且 `num_pred: 3`
- **THEN** batch 准备 MUST 读取 `batch["csi"]`
- **AND** 输出 `csi_batch` MUST 包含最近 8 个历史时隙和 2 个 zero padding 时隙
- **AND** batch 准备 MUST 不要求 image、radar、GPS、LiDAR 或 mmWave 字段

#### Scenario: 标签保持 clean future beam
- **WHEN** CSI 输入在数据侧被 `csi_degradation` 弱化或在模型中被 pilot estimation noise 弱化
- **THEN** 训练标签 MUST 继续来自 `target_beam[:, :num_pred]`
- **AND** 系统 MUST 不使用 degraded CSI 或 noisy CSI 重新生成 beam label

### Requirement: MMW channel 数据可派生 CSI 输入
MMW Town10 数据准备或后处理路径在启用 CSI 导出时 MUST 能从已有 `channel_path` 生成可被 CSI loader 读取的历史 `csi*` 列，或在 channel 文件缺少必要字段时给出可诊断失败原因。

#### Scenario: 从 channel path 写出 CSI 序列列
- **WHEN** MMW 准备配置启用 CSI 序列导出且 channel 文件可解析
- **THEN** 输出序列 CSV MUST 包含 `csi1..csiN`
- **AND** 每个 `csi*` 路径 MUST 指向稳定的 clean CSI tensor 文件或可直接读取的 channel tensor

#### Scenario: channel 文件不可派生 CSI
- **WHEN** channel 文件无法转换为 `[Nsc, Nant]` 或 `[Nsc, Nant, 2]` CSI
- **THEN** 准备流程 MUST 跳过对应样本或失败
- **AND** sanity report MUST 记录失败路径和失败原因
