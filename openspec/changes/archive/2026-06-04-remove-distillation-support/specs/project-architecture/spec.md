## MODIFIED Requirements

### Requirement: 可导入包结构
项目 MUST 提供 `src/kd_sensing/` Python 包，并将数据、模型、loss、训练引擎、评估、预处理、诊断和通用工具放入职责清晰的子模块。包内模块 MUST 使用包内绝对导入或明确相对导入，不得依赖仓库根目录脚本名作为运行时导入条件。项目 MUST 不再要求或暴露 `kd_sensing.distillation` 子包。

#### Scenario: 从项目根目录导入包
- **WHEN** 开发者在项目根目录安装或设置本地包路径后执行 `import kd_sensing`
- **THEN** 导入 MUST 成功
- **AND** 导入 MUST 不触发数据集读取、模型权重加载或训练逻辑

#### Scenario: 导入核心子模块
- **WHEN** 开发者导入 `kd_sensing.models`、`kd_sensing.data`、`kd_sensing.engine`、`kd_sensing.preprocessing` 和当前保留的 loss/evaluation 子模块
- **THEN** 每个子模块 MUST 成功导入
- **AND** 系统 MUST 不要求 `kd_sensing.distillation` 存在

### Requirement: 项目健康检查可分层运行
项目 MUST 提供或记录一组快速健康检查，用于在不启动真实训练的情况下验证导入边界、CLI 入口和当前保留的核心诊断逻辑。所有项目相关 Python 检查 MUST 使用 `kd_mm_beam` 环境。健康检查 MUST 不再要求 distillation 工具子模块、distiller registry、legacy KD config 或 fusion KD virtual alias 可用。

#### Scenario: 轻量导入 smoke
- **WHEN** 开发者运行项目健康检查中的轻量导入 smoke
- **THEN** 检查 MUST 验证配置、路径、模态契约、engine 轻量子模块和 diagnostics 轻量子模块可导入
- **AND** 检查 MUST 验证这些导入不触发 dataset、trainer、diagnostics render 或大型第三方依赖
- **AND** 检查 MUST 不导入 `kd_sensing.distillation`

#### Scenario: 快速回归命令覆盖当前表面
- **WHEN** 开发者运行项目健康检查中的快速回归命令
- **THEN** 检查 MUST 覆盖架构导入边界、console script help 和当前仍保留的核心诊断逻辑
- **AND** 检查 MUST 验证旧 KD 入口被拒绝

