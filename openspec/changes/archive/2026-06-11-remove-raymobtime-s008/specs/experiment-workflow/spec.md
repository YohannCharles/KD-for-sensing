## ADDED Requirements

### Requirement: 推荐实验工作流不包含 Raymobtime s008
README、实验矩阵、快速健康检查和配置驱动 workflow 文档 MUST 不再把 Raymobtime s008 作为当前推荐或可运行实验。历史 archive MAY 保留 Raymobtime 记录，但 MUST 不作为当前入口、教程或验证命令。

#### Scenario: README 和实验矩阵移除 Raymobtime 入口
- **WHEN** 用户阅读 README、docs/experiment_matrix.md 或研究笔记中的当前推荐流程
- **THEN** 文档 MUST 不再推荐运行 Raymobtime s008 预处理、训练或评估命令
- **AND** 文档 MUST 明确当前主线使用仍保留的数据集和 viewer workflow

#### Scenario: 健康检查不要求 Raymobtime focused test
- **WHEN** 开发者执行快速验证说明中的 focused tests
- **THEN** 验证命令 MUST 不要求 `tests/test_raymobtime_s008_selection.py`
- **AND** 验证 MUST 覆盖通用 CLI、架构边界、配置退役 guard 和当前保留 workflow

#### Scenario: 旧 Raymobtime 配置不可作为 workflow
- **WHEN** 用户传入 `configs/raymobtime/` 或 `configs/preprocess/raymobtime_s008_*.yaml` 下的旧配置路径
- **THEN** 系统 MUST 拒绝该 workflow 或这些配置文件 MUST 已被删除
- **AND** 错误信息 MUST 指出 Raymobtime s008 已退役
