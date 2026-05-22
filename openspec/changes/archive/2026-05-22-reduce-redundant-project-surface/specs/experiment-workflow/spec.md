## ADDED Requirements

### Requirement: 推荐实验文档保持精简入口
实验工作流文档 MUST 将 README 作为入口地图，而不是完整实验手册。README MUST 指向 canonical config、docs 和 OpenSpec；详细实验矩阵、分析流程和调参说明 MUST 放在 `docs/` 或对应 specs 中。

#### Scenario: README 提供最短可运行路径
- **WHEN** 新用户阅读 README
- **THEN** 用户 MUST 能找到安装命令、快速健康检查、训练/评估/预处理/manifest 导出入口和数据产物边界
- **AND** 用户 MUST 能通过链接进入详细实验矩阵或 viewer 文档

#### Scenario: 长实验说明迁移到 docs
- **WHEN** README 中的某段内容主要描述 G2D、CRAF、MARF、CSI hardening、viewer 或 Raymobtime 的详细实验流程
- **THEN** 该内容 MUST 迁移到对应 `docs/` 文件或 OpenSpec spec
- **AND** README MUST 保留简短摘要和链接

### Requirement: 表面积收敛保持实验 artifact 兼容
删除冗余配置、入口或文档后，训练和评估 workflow MUST 继续保存完整运行 artifact。使用 virtual/overlay 配置时，运行目录 MUST 记录足够信息用于复现，不得要求用户恢复已删除的实体 YAML。

#### Scenario: virtual 配置训练 artifact 完整
- **WHEN** 用户使用 virtual/overlay 配置启动训练并完成 artifact 写出
- **THEN** 运行目录 MUST 包含完整 `final_config.yaml`、`resolved_config.yaml`、`train_log.json`、checkpoint metadata 和 split/runtime metadata
- **AND** 这些 artifact MUST 足以说明实际模型、数据、loss、训练参数和 checkpoint 来源

#### Scenario: 删除 fallback 入口不影响 console script
- **WHEN** 重复脚本 wrapper 被删除
- **THEN** 对应 console script 或 `python -m kd_sensing.cli.*` 入口 MUST 继续通过 `--help` 检查
- **AND** README 推荐命令 MUST 使用仍存在的入口

#### Scenario: 研究脚本不进入核心 workflow 兼容承诺
- **WHEN** 保留的研究脚本未声明为包内 CLI
- **THEN** 核心训练、评估、预处理和 manifest 导出 workflow MUST 不依赖该脚本
- **AND** 该脚本的输出产物 MUST 继续位于 `.gitignore` 覆盖路径或显式本地输出目录
