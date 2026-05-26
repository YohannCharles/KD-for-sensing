## MODIFIED Requirements

### Requirement: 安装入口与 pyproject 声明一致
项目 MUST 确保 editable install 后的 console scripts 与 `pyproject.toml` 的 `[project.scripts]` 声明一致。README 或工具文档中推荐的包内 CLI MUST 可在 `kd_mm_beam` 环境中直接调用。保留的兼容 console script MUST 是薄 alias，不得复制长期维护的 parser 或主实现。项目 MUST 不再要求安装 `kd-sensing-raymobtime-analysis`。

#### Scenario: 可视化 manifest 导出入口可用
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 包含 manifest 导出参数，例如 `--config`、`--cache-dir`、`--scenes` 和 `--run-models`

#### Scenario: 可视化兼容入口可用
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 明确该入口导出 viewer manifest 或指向 Gradio viewer 工作流
- **AND** 该入口 MUST 委托 manifest 导出 CLI，不得复制独立 parser 或旧静态 PNG 主流程

#### Scenario: 安装元数据刷新后入口齐全
- **WHEN** 开发者在 `kd_mm_beam` 中执行 `python -m pip install -e .`
- **THEN** 安装生成的 entry points MUST 包含 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`、`kd-sensing-runs`、`kd-sensing-visualize-modalities` 和 `kd-sensing-export-viewer-manifest`
- **AND** 安装生成的 entry points MUST 不要求包含 `kd-sensing-raymobtime-analysis`

### Requirement: 项目健康检查可分层运行
项目 MUST 提供或记录一组快速健康检查，用于在不启动真实训练的情况下验证导入边界、CLI 入口和当前保留的核心诊断逻辑。所有项目相关 Python 检查 MUST 使用 `kd_mm_beam` 环境。健康检查 MUST 不再要求 Phase 1.5 或互补分析测试存在。

#### Scenario: 轻量导入 smoke
- **WHEN** 开发者运行项目健康检查中的轻量导入 smoke
- **THEN** 检查 MUST 验证配置、路径、模态契约、engine 轻量子模块、diagnostics 轻量子模块和 distillation 工具子模块可导入
- **AND** 检查 MUST 验证这些导入不触发指定重依赖模块

#### Scenario: 快速回归命令覆盖当前红点
- **WHEN** 开发者运行项目健康检查中的快速回归命令
- **THEN** 检查 MUST 覆盖架构导入边界、console script help 和当前仍保留的核心诊断逻辑
- **AND** 命令 MUST 能在全量 pytest 之前快速暴露项目结构回归

#### Scenario: 全量测试仍作为最终验收
- **WHEN** 变更实现完成
- **THEN** 开发者 MUST 使用 `conda run -n kd_mm_beam pytest -q` 作为最终回归验收
- **AND** 全量测试 MUST 通过

### Requirement: 源码热点模块必须按职责收敛
项目 MUST 将继续增长的大文件拆分到职责明确的窄模块中。拆分后，公开入口 MAY 保留薄 facade 或兼容导出，但主要实现 MUST 位于按职责命名的模块中，不得重新形成新的私有聚合层。已退役的互补性分析模块 MUST 不再作为源码热点分层要求。

#### Scenario: 修改 viewer 过滤逻辑不触碰图表渲染
- **WHEN** 开发者调整 Gradio viewer 的 scene、split、show mode 或低质量样本过滤逻辑
- **THEN** 主要变更 MUST 位于 viewer 过滤或 manifest IO 相关模块
- **AND** 不需要修改 Plotly 图表构造、prediction summary 表格或 Gradio 布局编排实现

#### Scenario: 修改 CSI hardening 不触碰 tokenizer
- **WHEN** 开发者调整 CSI hardening、pilot estimation 或噪声诊断逻辑
- **THEN** 主要变更 MUST 位于 CSI estimation 或 hardening 相关模块
- **AND** 不需要修改 CSI view tokenizer、view fusion 或 encoder registry glue
