## ADDED Requirements

### Requirement: 源码热点模块必须按职责收敛
项目 MUST 将继续增长的大文件拆分到职责明确的窄模块中。拆分后，公开入口 MAY 保留薄 facade 或兼容导出，但主要实现 MUST 位于按职责命名的模块中，不得重新形成新的私有聚合层。

#### Scenario: 修改 viewer 过滤逻辑不触碰图表渲染
- **WHEN** 开发者调整 Gradio viewer 的 scene、split、show mode 或低质量样本过滤逻辑
- **THEN** 主要变更 MUST 位于 viewer 过滤或 manifest IO 相关模块
- **AND** 不需要修改 Plotly 图表构造、prediction summary 表格或 Gradio 布局编排实现

#### Scenario: 修改互补性 schema adapter 不触碰 case summary
- **WHEN** 开发者新增 Conditional Utility Audit 输入字段别名或 subset 名称映射
- **THEN** 主要变更 MUST 位于互补性 schema adapter 相关模块
- **AND** 不需要修改 case mining、bucket summary 或输出写出实现

#### Scenario: 修改 CSI hardening 不触碰 tokenizer
- **WHEN** 开发者调整 CSI hardening、pilot estimation 或噪声诊断逻辑
- **THEN** 主要变更 MUST 位于 CSI estimation 或 hardening 相关模块
- **AND** 不需要修改 CSI view tokenizer、view fusion 或 encoder registry glue

### Requirement: Raymobtime s008 预处理必须模块化
Raymobtime s008 预处理实现 MUST 将路径解析、文件审计、index 构建、beam label normalization、ray feature 提取和 cache 写出放到职责明确的模块或 helper 中。保留的 preprocessor registry 入口 MUST 只负责参数编排和调用这些模块。

#### Scenario: 修改 index split 不触碰 ray feature
- **WHEN** 开发者调整 Raymobtime s008 index split、official split 检测或 split metadata 写出逻辑
- **THEN** 主要变更 MUST 位于 index 相关模块
- **AND** 不需要修改 ray-tracing HDF5/CSV feature 解析或 beam label normalization 实现

#### Scenario: 修改 beam label normalization 不触碰文件审计
- **WHEN** 开发者调整 Raymobtime s008 beam pair、beam score 或 class label normalization 规则
- **THEN** 主要变更 MUST 位于 beam label 相关模块
- **AND** 不需要修改 required path 审计、NPZ shape summary 或 raw CSV audit 实现

### Requirement: 入口生命周期必须可审计
项目 MUST 为 `scripts/`、`tools/analysis/` 和 `tools/visualization/` 中保留的入口维护生命周期分类。新增或保留入口 MUST 属于包内 CLI、薄 alias、研究诊断脚本、数据准备脚本、viewer entrypoint、viewer support 或 shell orchestration 中的一类，并在架构检查 allowlist 或 inventory 中记录原因。

#### Scenario: 新增脚本入口需要分类
- **WHEN** 开发者新增 `scripts/`、`tools/analysis/` 或 `tools/visualization/` 下的 Python 或 shell 入口
- **THEN** 架构边界检查 MUST 要求该入口出现在生命周期 allowlist 或 inventory 中
- **AND** 如果该入口复制已有 console script 或包内 CLI 工作流，检查 MUST 拒绝该入口

#### Scenario: 重复 wrapper 不作为推荐入口
- **WHEN** 包内 CLI 或 console script 已覆盖训练、评估、预处理或 viewer manifest 导出工作流
- **THEN** README 和工具文档 MUST 推荐包内 CLI 或 console script
- **AND** 对应 `scripts/` 或 `tools/` fallback wrapper MUST 删除或被明确标注为短期薄 alias

### Requirement: 架构优化不得触碰本地数据和产物
源码、配置和入口表面积优化 MUST 不移动、删除、压缩或重写 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、下载压缩包或其它本地运行产物。相关检查 MUST 只验证源码控制范围内的文件和忽略规则。

#### Scenario: 实施源码拆分不清理产物
- **WHEN** 开发者实施本 change 中的源码拆分、配置瘦身或入口收敛任务
- **THEN** 变更 MUST 不包含对 `dataset/`、`outputs/`、`logs/` 中真实文件的删除、移动或压缩操作
- **AND** 架构检查 MUST 继续只拒绝已跟踪源码表面积中的本地产物污染

#### Scenario: 数据目录策略不随本变更改变
- **WHEN** 本 change 完成并归档
- **THEN** 默认数据目录、legacy data_root 兼容规则和用户显式 data_root 行为 MUST 保持不变
- **AND** 本 change MUST 不要求用户迁移本地数据才能继续运行既有配置
