## ADDED Requirements

### Requirement: Hist 研究线不属于当前包结构
项目当前包结构 MUST 不再要求或暴露 HiST-Beam/Hist 专用 CLI、engine、model、evaluation 或 config 模块。`src/kd_sensing/engine` 与 `src/kd_sensing/models` MUST 保留当前主线职责模块，退役 Hist 专用文件后不得新增旧入口 facade。

#### Scenario: 包导入不要求 Hist 模块
- **WHEN** 开发者执行 `import kd_sensing`、`import kd_sensing.engine` 或 `import kd_sensing.models`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不要求 `kd_sensing.engine.hist_beam_*` 或 `kd_sensing.models.fusion.hist_beam` 存在

#### Scenario: 架构边界拒绝 Hist 旧入口
- **WHEN** 开发者运行架构边界检查
- **THEN** 检查 MUST 验证当前源码不再从 Hist 专用 engine/model/evaluation 模块导入运行逻辑
- **AND** 检查 MUST 验证没有新增 `hist_beam` 兼容聚合层

### Requirement: 退役研究线不触发本地产物隐式迁移
源码删除和包结构整理 MUST 与本地产物清理解耦。删除 Hist 源码 MUST 不自动移动、压缩或删除 `outputs/`、`logs/`、cache 或 checkpoint；本地产物删除 MUST 通过 runtime cleanup manifest 和显式删除阶段完成。

#### Scenario: 源码删除不隐式清理 outputs
- **WHEN** 实施者删除 Hist 源码、配置和文档入口
- **THEN** 该源码变更 MUST 不在同一步骤中用 ad hoc 命令删除 `outputs/`
- **AND** 需要删除的运行产物 MUST 先出现在 cleanup manifest 中
