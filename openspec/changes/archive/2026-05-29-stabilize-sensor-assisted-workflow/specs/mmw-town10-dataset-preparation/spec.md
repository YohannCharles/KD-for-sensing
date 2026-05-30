## ADDED Requirements

### Requirement: MMW split 与 radar CSV materialization 使用公开准备入口
MMW Town10 数据准备 MUST 提供公开 package utility、preprocessor 或 CLI，用于创建和校验 sensor-assisted split、sequence CSV 和 radar CSV materialization。训练 preflight MUST 调用该公开入口或读取已准备 artifact，不得依赖 dataset 私有 helper 来写出 CSV。

#### Scenario: 公开 utility 生成 radar CSV
- **WHEN** 用户或 preflight 调用公开 MMW split/radar CSV 准备入口
- **THEN** 系统 MUST 基于 prepared manifest 和 split metadata 生成需要的 sequence CSV 或 radar CSV
- **AND** 输出 metadata MUST 记录输入 manifest、split 配置、seq_len、num_pred、condition、scenario、样本数和输出路径
- **AND** 生成产物 MUST 位于 dataset 或显式本地输出目录，不得写入源码控制目录

#### Scenario: 训练 preflight 不导入私有 dataset helper
- **WHEN** HiST-Beam MMW LOSO executor 执行 preflight
- **THEN** preflight MUST NOT 从 dataset 模块导入 `_ensure_*` 私有 helper 来物化 radar CSV 或 split CSV
- **AND** preflight MUST 只读取已准备 artifact、调用公开准备 utility 或报告缺失 artifact

#### Scenario: 缺失 prepared artifact 给出可执行提示
- **WHEN** preflight 发现 sensor-assisted run 所需 split CSV 或 radar CSV 缺失
- **THEN** preflight MUST 失败并输出可执行修复提示
- **AND** 提示 MUST 包含公开 MMW 准备入口、关键参数和目标输出路径
- **AND** preflight MUST 不静默创建不完整或无 metadata 的 CSV
