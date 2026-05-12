## ADDED Requirements

### Requirement: 内部代码不得新增二级兼容聚合层依赖
项目 MUST 区分公开兼容 facade 和私有二级兼容聚合层。新内部代码 MUST 不再引用 `kd_sensing.engine._builders_impl` 或 `kd_sensing.data.transform_ops._legacy`；需要 builder 或 transform 功能时 MUST 使用窄模块或公开 facade。二级兼容聚合层若继续存在，MUST 只服务历史私有 import 过渡。

#### Scenario: 内部代码使用窄 builder 模块
- **WHEN** 开发者新增或修改训练、评估、诊断脚本
- **THEN** 代码 MUST 优先从 `engine.data_factory`、`engine.optim`、`engine.run_metadata`、`engine.cache_policy` 或其它窄模块导入
- **AND** 不得新增对 `kd_sensing.engine._builders_impl` 的依赖

#### Scenario: 内部代码使用模态 transform 模块
- **WHEN** 开发者新增或修改数据集、预处理或诊断读取逻辑
- **THEN** 代码 MUST 优先从 `kd_sensing.data.transform_ops.<modality>` 或通用 transform 子模块导入
- **AND** 不得新增对 `kd_sensing.data.transform_ops._legacy` 的依赖

### Requirement: 重复 CLI 脚本不得作为推荐入口
当包内 CLI 与 `tools/` 脚本提供同一工作流时，项目 MUST 以包内 CLI 或 `python -m kd_sensing.cli.<name>` 作为推荐入口。`tools/` 中的重复脚本 MAY 短期保留为开发 fallback，但 MUST 不再承载唯一功能或复制长期维护的 parser/main 实现。

#### Scenario: viewer manifest 推荐包内 CLI
- **WHEN** 文档或 orchestration 脚本需要导出 Gradio viewer manifest
- **THEN** 推荐命令 MUST 使用 `kd-sensing-export-viewer-manifest` 或包内 CLI 模块
- **AND** 不得把 `tools/visualization/export_viewer_manifest.py` 作为唯一推荐路径
