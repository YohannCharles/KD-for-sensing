## ADDED Requirements

### Requirement: Residual workflow 使用包内 CLI
DeepSense6G residual workflow 的新运行入口 MUST 位于 `src/kd_sensing/` 包内，并通过包内 CLI 模块或 pyproject console script 暴露。项目 MUST NOT 新增顶层 `src.*` 运行模块或绕过 `kd_sensing` 包结构的兼容包装。

#### Scenario: 包内 inspection CLI
- **WHEN** 用户运行 residual input inspection
- **THEN** 入口 MUST 委托 `kd_sensing` 包内实现
- **AND** 命令参数 MUST 支持 GPS sweep root、label space 和输出检查
- **AND** import 该 CLI 模块 MUST 不触发训练或读取大型数据

#### Scenario: 包内 residual train/plot/compare CLI
- **WHEN** 用户运行 residual manifest、train/eval、plot 或 compare 命令
- **THEN** 入口 MUST 位于 `src/kd_sensing/cli/` 或等价包内 CLI 模块
- **AND** pyproject console script 若新增 MUST 委托同一包内实现
- **AND** 项目 MUST NOT 创建 `src/inspect_deepsense6g_residual_inputs.py` 这类绕过包结构的模块
