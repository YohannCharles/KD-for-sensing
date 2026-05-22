## ADDED Requirements

### Requirement: 可视化诊断内部轻量模块边界
Manifest/viewer 诊断内部 MUST 区分轻量 helper 与重依赖运行模块。配置解析、采样候选选择、metadata 写出和 JSON payload 规范化 MUST 保持轻量导入；数据集构建、processed asset 生成、PNG 渲染和模型预测导出 MAY 导入 pandas、torch、PIL、matplotlib 或 engine builders，但这些依赖 MUST 限定在对应职责模块或函数内。

#### Scenario: 解析 viewer 诊断配置不导入渲染依赖
- **WHEN** 开发者导入或调用 viewer 诊断配置解析 helper
- **THEN** 系统 MUST 能解析 `diagnostics.visualization` 的输出目录、splits、sample count、seed、filters、modalities 和 scene comparison 配置
- **AND** 该路径 MUST 不导入 matplotlib、PIL 或 PNG render helper

#### Scenario: 采样 helper 不读取 dataset
- **WHEN** 开发者导入采样 helper 并传入候选记录
- **THEN** 系统 MUST 能按 seed、seq_index、label 和 per-seq sample count 选择样本
- **AND** 采样 helper MUST 不构建 dataset、不读取 CSV 文件、不加载 checkpoint

#### Scenario: 写出 helper 只负责文件序列化
- **WHEN** 开发者调用 JSON、JSONL 或 CSV metadata 写出 helper
- **THEN** helper MUST 只负责目标路径创建和 payload 序列化
- **AND** helper MUST 不导入 dataset builder、model builder、matplotlib 或 PIL

### Requirement: Manifest 行为保持兼容
收紧诊断内部 import 边界时，manifest 导出和 viewer prediction 导出的公开行为 MUST 保持兼容。输出 manifest、metadata、processed asset 路径、prediction bundle 合并和 viewer 启动命令语义 MUST 不因内部模块整理而改变。

#### Scenario: manifest 导出 payload 兼容
- **WHEN** 用户运行 `kd-sensing-export-viewer-manifest` 或包内 CLI 导出 viewer manifest
- **THEN** 输出 manifest MUST 保持当前字段语义
- **AND** manifest 记录 MUST 继续包含 sample id、scene、split、sequence、raw/processed assets、label、enabled modalities 和 statistics

#### Scenario: 模型预测导出兼容
- **WHEN** 用户使用 viewer 模型预测导出能力
- **THEN** 输出 prediction 文件 MUST 继续包含每个样本和模态的 beam distribution、confidence curves 和 future labels
- **AND** 导出流程 MUST 继续复用统一 runtime forward 和 future-only label 语义
