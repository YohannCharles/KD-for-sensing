## ADDED Requirements

### Requirement: 通用 baseline 与 workflow baseline 分层
项目 MUST 区分通用可训练 baseline 和 workflow/paper reproduction baseline。通用 baseline MUST 复用配置驱动训练、共享 batch/runtime 和模型 registry；workflow baseline MUST 只在需要官方协议、多阶段训练、特殊 metric 或报告产物时保留专用 orchestration，并 MUST 放在包内职责清晰的位置并记录生命周期、产物边界和 claim caveat。

#### Scenario: 通用 baseline 不修改训练循环
- **WHEN** 开发者新增普通 supervised/adaptation baseline
- **THEN** 变更 MUST 限定在配置、模型子组件、registry/default component 和 focused tests
- **AND** 不得为了该 baseline 修改 dataset 解析、训练主循环或公共 CLI 入口

#### Scenario: 论文复现 workflow 有边界
- **WHEN** 开发者新增包含官方协议、多阶段训练、特殊 metrics 或报告产物的 workflow baseline
- **THEN** 代码 MUST 位于 `src/kd_sensing/baselines/<family>/`、包内 CLI 或明确生命周期的薄 alias
- **AND** 文档 MUST 标记其不是普通 `modular_sequence` baseline，并说明输出只写入 ignored runtime artifact root

### Requirement: 新模型不得扩大入口表面
新增模型架构能力 MUST 不新增 root-level 旧脚本、兼容聚合层、退役研究线实体配置或绕过 `src/kd_sensing` 包结构的运行方式。若需要新增 CLI，MUST 是包内 console script 或 lifecycle 登记的薄 alias，并同步 pyproject、README/docs、inventory 和架构边界测试。

#### Scenario: 新模型需要命令入口
- **WHEN** whole-model exception 或 workflow baseline 需要新的用户命令
- **THEN** 入口 MUST 通过包内 CLI 或登记的薄 alias 暴露
- **AND** 系统 MUST 不新增仓库根长期训练脚本或未登记脚本入口
