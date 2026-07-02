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

### Requirement: CSI reconstruction supervision for physics baseline
系统 MUST 允许 physics-informed baseline 将当前完整 clean CSI 作为 `csi_target` reconstruction target。该 target MUST 沿用 `[T, Nsc, Nant, 2]` real/imag 末维契约，MUST 不作为默认 sensing input，并且在目标缺失时 MUST 通过 mask 跳过 CSI reconstruction loss。

#### Scenario: clean CSI 作为 reconstruction target
- **WHEN** 配置启用 physics supervision 和 `physics.loss.csi_reconstruction.enabled=true`
- **THEN** dataset/batch adapter MUST 提供 clean `csi_target` 或显式 unavailable mask
- **AND** reconstruction loss MUST 使用 clean CSI target 计算 NMSE/MSE
- **AND** metadata MUST 记录 CSI target 来源和是否使用 degradation

#### Scenario: 受限 CSI 才能作为模型输入
- **WHEN** 配置启用 `data.use_csi_input=true`
- **THEN** dataset/batch adapter MUST 根据 `data.csi_input_mode` 提供 `csi_input`
- **AND** `history`、`partial`、`noisy`、`compressed` 模式 MUST 不直接暴露当前完整 CSI
- **AND** 只有 `oracle_full` 模式在显式 `allow_oracle_full_csi_input=true` 时 MAY 将当前完整 CSI 作为模型输入

#### Scenario: CSI shape 对齐失败可诊断
- **WHEN** `h_hat` 和 clean CSI target 的 subcarrier、antenna 或 time/horizon 维度无法按配置对齐
- **THEN** physics loss MUST 抛出包含 `h_hat` shape、CSI shape、num_subcarriers 和 antenna 维度的错误
- **AND** 系统 MUST 不静默 broadcast 或截断到错误维度

#### Scenario: 未启用 CSI 不要求 CSI 列
- **WHEN** physics-informed 配置关闭 CSI 输入和 CSI reconstruction loss
- **THEN** dataset MUST 不要求 `csi*` 列
- **AND** loss diagnostics MUST 标记 CSI reconstruction disabled 而不是 unavailable error

### Requirement: Sparse pilot CSI observation mask
受限 CSI 输入 MAY 包含 `csi_observation_mask`，用于标记 sparse pilot 观测位置。mask MUST 与 `csi_input` 的 time/subcarrier/antenna 维度对齐，且未观测位置不得携带真实 CSI 值。

#### Scenario: mask 与 csi_input 对齐
- **WHEN** dataset/batch adapter 生成 sparse pilot CSI 输入
- **THEN** `csi_observation_mask` MUST 覆盖 `[T, Nsc, Nant]` 或 batch 后 `[B, T, Nsc, Nant]`
- **AND** `csi_input[..., ~mask, :]` 的 real/imag 值 MUST 为 0 或等价 missing sentinel
- **AND** 完整 clean CSI MUST 只保留在 `csi_target`

### Requirement: CSI 按模态选择加载样本
DeepSense6G/MMW dataset MUST 根据启用模态决定是否加载 CSI。未启用 CSI 时，CSI 路径列或文件缺失不得阻止当前任务运行；启用 CSI 时，dataset MUST 返回 `csi` 字段并保持其它未启用模态不读取。

#### Scenario: CSI-only 不读取其它输入模态文件
- **WHEN** 用户运行 `experiment.task: csi` 的训练或评估配置
- **THEN** dataset MUST 只读取 CSI、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image、radar map、GPS、LiDAR 或 mmWave 加载逻辑
- **AND** 返回样本 MUST 包含 `csi`

#### Scenario: fusion 按 modalities 读取 CSI
- **WHEN** 用户运行 `experiment.task: fusion` 且配置 `modalities: ["gps", "csi"]`
- **THEN** dataset MUST 只读取 GPS、CSI、`input_beam` 和 `target_beam` 所需文件
- **AND** 返回样本 MUST 只包含启用模态对应输入字段和标签字段

### Requirement: CSI normalizer artifact 复用
数据构建流程 MUST 将训练集 CSI RMS normalizer 从 train dataset 传递给 test dataset，并允许训练/评估 metadata 记录该统计。

#### Scenario: dataloader 复用 CSI RMS
- **WHEN** `build_dataloaders` 构建启用 CSI 的 train 和 test dataset
- **THEN** train dataset MUST 先准备 CSI RMS normalizer
- **AND** test dataset MUST 接收同一个 CSI RMS normalizer 或等价数值
