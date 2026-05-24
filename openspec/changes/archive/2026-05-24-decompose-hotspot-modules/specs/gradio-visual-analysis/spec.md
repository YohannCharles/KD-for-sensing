## ADDED Requirements

### Requirement: Viewer manifest helper 拆分兼容
Viewer manifest、prediction merge、cache metadata、sample record schema 和 writer 相关 helper 拆分后，公开 viewer 入口、manifest 导出 CLI 和 Gradio viewer 行为 MUST 保持兼容。拆分 MUST 不改变 manifest JSON/JSONL 的公开字段语义。

#### Scenario: manifest 导出入口保持
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help`
- **THEN** 命令 MUST 正常退出
- **AND** 拆分后的实现 MUST 继续支持原有 manifest 参数和输出路径语义

#### Scenario: prediction merge 保持
- **WHEN** 用户提供 predictions、quality 或 gate JSON
- **THEN** viewer manifest 构建 MUST 继续合并这些诊断字段
- **AND** manifest 中既有 prediction/quality/gate 字段语义 MUST 保持兼容

#### Scenario: viewer 支持模块轻量导入
- **WHEN** 开发者导入 manifest schema 或 viewer path helper 子模块
- **THEN** 导入 MUST 不启动 Gradio app
- **AND** 导入 MUST 不加载模型 checkpoint 或真实数据集
