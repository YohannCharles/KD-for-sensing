## ADDED Requirements

### Requirement: 训练运行时编排职责拆分
训练引擎 MUST 将训练运行时状态、单 batch step、epoch metrics/history、checkpoint/sidecar、TensorBoard 和最终 artifact 写出拆到职责明确的窄模块或 helper。`kd_sensing.engine.trainer.train` MAY 保留为公开入口和顶层生命周期编排器，但 MUST 不继续直接承载这些细节的主要实现。

#### Scenario: batch step 逻辑位于窄模块
- **WHEN** 开发者查看训练中单 batch 的 prepare、forward、loss、backward 和 optimizer step 编排
- **THEN** 主要实现 MUST 位于 batch step runner 或等价窄模块
- **AND** `trainer.py` MUST 只负责调用该 runner 并消费其返回的 loss、diagnostics 和状态更新

#### Scenario: checkpoint 写出位于 checkpoint manager
- **WHEN** 开发者调整 `best.pth`、`best_top1.pth`、`last.pth`、sidecar 或 checkpoint registry archive 的写出逻辑
- **THEN** 主要变更 MUST 限定在 checkpoint manager 或等价窄模块
- **AND** 不需要编辑训练 batch 主循环

#### Scenario: 训练 artifact 写出位于 artifact writer
- **WHEN** 开发者调整 `train_log.json`、`training_outputs.npz`、`final_config.yaml`、训练曲线或 debug artifact 的写出逻辑
- **THEN** 主要变更 MUST 限定在 artifact writer、history recorder 或等价窄模块
- **AND** 不需要编辑模型 forward、KD loss 或 optimizer step 逻辑

### Requirement: config/io 不承载业务规则实现
`kd_sensing.config.io` MUST 保持配置入口协调职责，负责加载实体 YAML 或 virtual config、应用命令行覆盖、调用 normalization pipeline 和调用 validation pipeline。objective 默认补全、模态推导、dataset-specific rules、迁移拒绝和 schema validation 的主要实现 MUST 位于独立 helper。

#### Scenario: Raymobtime 规则不写在 io 入口
- **WHEN** 开发者调整 Raymobtime s008 禁止 history/future/transition 配置的规则
- **THEN** 主要实现 MUST 位于 Raymobtime dataset rule、config validation helper 或等价窄模块
- **AND** `config/io.py` MUST 只调用该 helper

#### Scenario: removed image motion guard 不写在 io 入口
- **WHEN** 开发者调整已删除 image motion profile、cache 或 encoder 的拒绝逻辑
- **THEN** 主要实现 MUST 位于 migration guard 或 image profile validation helper
- **AND** `config/io.py` MUST 不直接维护该迁移规则的完整实现

#### Scenario: objective 默认值不写在 io 入口
- **WHEN** 开发者新增或调整 prediction objective 的默认 early stopping metric、loss weights 或 required target/head
- **THEN** 主要实现 MUST 位于 objective metadata、normalization helper 或 validation helper
- **AND** `config/io.py` MUST 不维护 objective 专属分支表

## MODIFIED Requirements

### Requirement: 安装入口与 pyproject 声明一致
项目 MUST 确保 editable install 后的 console scripts 与 `pyproject.toml` 的 `[project.scripts]` 声明一致。README 或工具文档中推荐的包内 CLI MUST 可在 `kd_mm_beam` 环境中直接调用。保留的兼容 console script MUST 是薄 alias，不得复制长期维护的 parser 或主实现。

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
- **THEN** 安装生成的 entry points MUST 包含 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`、`kd-sensing-raymobtime-analysis`、`kd-sensing-visualize-modalities` 和 `kd-sensing-export-viewer-manifest`

### Requirement: 兼容冗余入口已删除
项目 MUST 删除已经迁移到 canonical 模块的兼容入口。源码、测试、文档和推荐命令 MUST 不再依赖 `the builder facade module`、`the transform facade module`、`the transform aggregate module`、场景专用 dataset 兼容模块或复制旧实现的可视化脚本入口。明确保留的 console-script 兼容入口 MUST 作为薄 alias 存在，并 MUST 指向当前包内主实现。

#### Scenario: 兼容 facade 不再作为公开入口
- **WHEN** 开发者在源码、测试、README 或扩展指南中搜索已删除的兼容 facade
- **THEN** 不得出现 `the builder facade module`、`the transform facade module` 或 `the transform aggregate module` 的运行时引用
- **AND** 对应功能 MUST 通过职责明确的窄模块导入

#### Scenario: 旧入口引用检查
- **WHEN** 开发者运行架构边界检查
- **THEN** 检查 MUST 拒绝新增 `scene-specific dataset class alias`、`the scene-9 dataset-type spelling`、legacy fusion 配置路径或兼容 facade 引用
- **AND** 检查 MUST 在不读取真实数据和不加载 checkpoint 的情况下完成

#### Scenario: 保留的可视化兼容入口是薄 alias
- **WHEN** 项目保留 `kd-sensing-visualize-modalities` console script
- **THEN** 该入口 MUST 调用 `kd_sensing.cli.export_viewer_manifest` 或等价当前包内主实现
- **AND** 该入口 MUST 不承载独立业务逻辑、重复 parser 或旧静态 PNG 总览图主流程
