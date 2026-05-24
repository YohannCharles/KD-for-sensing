## ADDED Requirements

### Requirement: 源码表面积优化必须保持核心 workflow 兼容
删除冗余配置、拆分源码模块或收敛入口后，训练、评估、预处理、viewer manifest 导出和研究诊断的公开工作流 MUST 保持现有用户可见语义。实现 MAY 调整内部模块位置，但 MUST 不要求用户改用未记录的新命令。

#### Scenario: 核心 CLI help 继续可用
- **WHEN** 本 change 完成后开发者运行核心入口 help 检查
- **THEN** `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess` 和 `kd-sensing-export-viewer-manifest` MUST 正常退出
- **AND** 对应包内 CLI 模块 MUST 继续可通过 `python -m kd_sensing.cli.<name> --help` 调用

#### Scenario: 拆分模块不改变公共返回结构
- **WHEN** 用户通过既有公开函数或 CLI 运行训练、评估、预处理或 viewer manifest 导出
- **THEN** 返回 payload、日志字段、manifest 字段和主要输出路径语义 MUST 与拆分前兼容
- **AND** 内部模块重命名 MUST 不要求用户修改配置文件中的公共字段

### Requirement: 删除实体配置后 workflow 必须可复现
当实体 YAML 被 recipe/overlay 替代后，训练和评估 workflow MUST 继续保存足够的 resolved/final 配置、运行元数据和 checkpoint 来源信息，保证不恢复被删除 YAML 也能理解实际运行参数。

#### Scenario: virtual 配置训练记录完整
- **WHEN** 用户使用 virtual/overlay 配置完成训练或 dry-run artifact 写出
- **THEN** 运行目录 MUST 包含完整 `final_config.yaml`、`resolved_config.yaml`、训练元数据和 checkpoint 来源信息
- **AND** 这些 artifact MUST 能说明实际模型、数据、loss、训练参数和输出 run name

#### Scenario: 删除 YAML 不影响评估入口
- **WHEN** 某个实体 YAML 被删除但对应 virtual/overlay 配置仍被声明支持
- **THEN** `kd-sensing-evaluate --config <deleted-yaml-path>` MUST 通过配置加载器解析等价最终配置
- **AND** 如果该路径未被声明支持，系统 MUST 抛出清晰缺失配置错误

### Requirement: 入口收敛不得让研究脚本成为核心依赖
保留的 `scripts/`、`tools/analysis/` 和 `tools/visualization/` 研究或支持脚本 MUST 不成为核心训练、评估、预处理或 manifest 导出 workflow 的必需依赖。核心 workflow MUST 通过包内模块、console script 或明确保留的薄 alias 完成。

#### Scenario: 训练入口不依赖研究脚本
- **WHEN** 用户运行 `kd-sensing-train` 或 `python -m kd_sensing.cli.train`
- **THEN** 训练 workflow MUST 不要求调用 `scripts/analyze_*`、`tools/analysis/*` 或 viewer 支持脚本
- **AND** 研究脚本删除或重分类 MUST 不破坏核心训练入口

#### Scenario: viewer 支持脚本边界清晰
- **WHEN** 用户启动 Gradio viewer 或导出 viewer manifest
- **THEN** manifest 导出 MUST 通过 `kd-sensing-export-viewer-manifest` 或包内 CLI 完成
- **AND** `tools/visualization/gradio_multimodal_viewer.py` MAY 作为 viewer entrypoint 保留，但 MUST 不复制 manifest 导出 CLI 的 parser 和业务逻辑

### Requirement: 本 change 不改变本地产物策略
源码、配置和入口优化完成后，本地产物策略 MUST 保持现状。工作流 MAY 继续生成 outputs、logs、cache 和 checkpoint，但本 change MUST 不要求清理、压缩、迁移或提交这些产物。

#### Scenario: 训练输出仍位于忽略路径
- **WHEN** 用户在本 change 后运行训练或评估并生成输出
- **THEN** 新的 logs、outputs、cache 和 checkpoint MUST 继续位于 `.gitignore` 覆盖路径或显式本地输出目录
- **AND** 文档 MUST 不要求将这些本地产物加入源码变更

#### Scenario: 不要求清理已有产物
- **WHEN** 开发者实施本 change 的任务
- **THEN** 任务验收 MUST 不包含删除、压缩或迁移既有 `dataset/`、`outputs/`、`logs/` 文件
- **AND** 测试和 OpenSpec 校验 MUST 能在不修改这些本地产物的情况下完成
