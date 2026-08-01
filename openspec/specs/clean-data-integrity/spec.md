# Clean Data Integrity Specification

## Purpose

定义 MMW 唯一 ID stratified block protocol 的 train/validation/test 数据完整性、train-only 拟合状态与默认 test 封存，并保持 DeepSense6G 独立数据契约。

## Requirements

### Requirement: MMW 只能使用唯一且精确绑定的 block protocol

MMW 训练和开发验证 MUST 通过 protocol `mmw_id_stratified_block_v1` 建立数据域。配置 MUST 显式绑定 manifest 路径/hash、protocol version、split seed、block size、data source hash、window config hash、train/validation/test role 与通过的 audit report。clean-inner、trajectory-disjoint、历史 group-safe、窗口随机拆分和未知 MMW protocol MUST 在创建 dataset 前失败。

#### Scenario: 构建合法 MMW loader

- **WHEN** 配置与受支持 manifest、audit 和 split CSV 完全一致
- **THEN** 系统 MUST 默认只构建 train 与 validation loader
- **AND** 每个 loader 的运行元数据 MUST 记录 protocol、split seed 与 manifest identity

### Requirement: split 间必须隔离 block、基础帧、天气副本与窗口帧

协议校验 MUST 允许并要求同一 `(scene_id,cav_id)` 的不同 block 出现在 train、validation、test，同时拒绝跨 split block、base sample、天气副本和窗口实际引用帧。系统 MUST 审计 sample/target identity、完整 source row、历史/未来 frame、weather binding、block identity 与 split assignment。共享 RSU context MUST 作为 diagnostic overlap 披露，不得用于改变 block assignment。

#### Scenario: 其他 split 的 block 被并入训练输入

- **WHEN** validation/test block 或窗口被加入 train，或任一窗口跨 block
- **THEN** audit MUST 失败
- **AND** 配置不得进入训练或开发评估

### Requirement: 可拟合状态只能来自训练集

GPS scaler、CSI codebook、prototype 统计、contrastive memory/negative queue 与其他 normalization artifact MUST 只由绑定 protocol 的 train loader 拟合。validation/test MUST 不参与 optimizer、scheduler、extension state、训练采样、prototype 初始化、class prior 或可拟合统计更新；checkpoint selection 只能读取 validation loss。test 只有显式 `--evaluate-test` 才可构建，且只能用于最终只读评估。

#### Scenario: 运行开发验证

- **WHEN** 系统对 MMW validation loader 评估
- **THEN** validation MUST 是只读输入
- **AND** 默认不得构建 test dataset，metadata MUST 保持 `test_evaluated=false`

### Requirement: DeepSense6G 不受 MMW protocol 重解释

DeepSense6G MUST 保留 Scene31--34、四模态和 64 类 future-beam 数据契约。它可以有 train/test 或显式 validation CSV，但不得要求 MMW trajectory protocol。

#### Scenario: 加载 DeepSense6G recipe

- **WHEN** recipe 没有 MMW data protocol
- **THEN** 数据工厂 MUST 使用其独立 split 契约
