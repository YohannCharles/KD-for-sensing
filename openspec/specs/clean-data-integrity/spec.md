# Clean Data Integrity Specification

## Purpose

定义 Clean MMW 训练和开发验证之间不可绕过的数据隔离边界，防止验证样本、原始资源、时间窗口或拟合统计重新进入训练流程。

## Requirements

### Requirement: MMW 只能使用已绑定的 clean protocol

MMW 训练、开发验证和 MMW matrix MUST 通过 `mmw_clean_inner_development_v1` protocol 建立数据域。配置 MUST 显式绑定 protocol id、fingerprint、审计报告、`inner_train`、`inner_validation`，并显式关闭 outer test 与 confirmation training。

#### Scenario: 构建合法 MMW loader

- **WHEN** 配置与 protocol、审计报告和域清单完全一致
- **THEN** 系统 MUST 仅构建 train 与 validation loader
- **AND** 每个 loader 的运行元数据 MUST 记录 protocol 与 audit identity

#### Scenario: 使用未绑定或历史 MMW 配置

- **WHEN** MMW 配置缺少 protocol identity、审计报告、inner role 或 outer-test 禁止字段
- **THEN** 系统 MUST 在创建 dataset、optimizer 或 checkpoint 前失败

### Requirement: train 与 validation 必须完全隔离

协议校验 MUST 拒绝 confirmation、trainval、merged train/validation 训练路径，并对每个 train domain 与每个 validation domain 审计 sample id、target id、完整行、原始资源、时间窗口和目标帧的重叠。任一重叠 MUST 失败。

#### Scenario: 验证集被并入训练输入

- **WHEN** 任一 train/validation 域对共享要求隔离的身份或资源
- **THEN** 审计 MUST 失败
- **AND** 配置不得进入训练或开发评估

### Requirement: 可拟合状态只能来自训练集

GPS scaler 与其他 normalization artifact MUST 只由 train loader 拟合。validation MUST 不参与 optimizer、scheduler、extension state、采样、checkpoint selection 或可拟合统计更新。

#### Scenario: 运行开发验证

- **WHEN** 系统对 Clean MMW validation loader 评估
- **THEN** validation MUST 是只读输入
- **AND** outer test MUST 保持未访问

### Requirement: DeepSense6G 不受 MMW protocol 重解释

DeepSense6G MUST 保留 Scene31--34、四模态和 64 类 future-beam 数据契约。它可以有 train/test 或显式 validation CSV，但不得将 test 隐式转换为 validation，也不得要求 MMW clean protocol。

#### Scenario: 加载 DeepSense6G recipe

- **WHEN** recipe 没有 MMW data protocol
- **THEN** 数据工厂 MUST 使用其独立 split 契约
