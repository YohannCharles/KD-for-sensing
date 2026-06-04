## ADDED Requirements

### Requirement: Legacy KD 已删除
系统 MUST 不再保留 legacy KD 代码、配置、测试或运行时入口。`logits_kd`、`rkd`、teacher-student KD、KD baseline summary 和 KD virtual alias MUST 被视为已删除能力。

#### Scenario: 显式 legacy KD 被拒绝
- **WHEN** 用户请求运行 `logits_kd`、`rkd` 或等价 legacy KD baseline
- **THEN** 系统 MUST 拒绝该请求
- **AND** 错误信息 MUST 指向当前 supervised/adaptation workflow

#### Scenario: 新方法不得复用 legacy KD runtime
- **WHEN** 开发者新增 HiST-Beam residual、prototype、calibration 或其它方法
- **THEN** 实现 MUST 不依赖 legacy KD teacher-student forward 逻辑
- **AND** 仓库 MUST 不包含可复用的 legacy KD runtime 聚合入口

## REMOVED Requirements

### Requirement: Legacy KD 隔离边界
**Reason**: KD 不再是隔离保留能力，而是删除能力。
**Migration**: 使用 supervised/adaptation 入口；未来 KD 需重新提案。

#### Scenario: 显式 legacy KD 才启用蒸馏
- **WHEN** 用户运行旧 legacy KD 配置
- **THEN** 系统 MUST 拒绝该配置

### Requirement: KD baseline 不参与默认主结论
**Reason**: 新运行不再产生 KD baseline。
**Migration**: 历史 KD 结果只读保留为旧产物，不进入新 summary contract。

#### Scenario: KD run 默认不可用于主结论
- **WHEN** 新 summary 读取运行产物
- **THEN** 产物 MUST 不包含可运行 KD baseline 分组

### Requirement: KD 历史代码保留策略
**Reason**: 用户要求删除 KD 代码和配置以精简项目。
**Migration**: 历史复现实验保留在 archive 或外部历史提交，不作为当前源码入口。

#### Scenario: 历史 KD 配置可追溯
- **WHEN** 仓库源码中出现 `logits_kd`、`rkd` 或 teacher-student KD 配置
- **THEN** 检查 MUST 失败

### Requirement: KD 运行时职责不得扩散
**Reason**: KD 运行时整体删除，无需定义职责边界。
**Migration**: 新 loss 放入 loss/objective/extension 模块。

#### Scenario: 导入 KD 算法不构建运行对象
- **WHEN** 开发者尝试导入 KD 算法模块
- **THEN** 导入 MUST 失败或触发清晰删除错误

### Requirement: KD 可选增强必须重新提案
**Reason**: 该约束迁移到 distillation-free 项目表面的旧入口拒绝规则中。
**Migration**: 未来蒸馏能力必须以新的 OpenSpec change 创建新 capability。

#### Scenario: 新 KD 增强不能静默加入主线
- **WHEN** 开发者新增 KD 相关配置或代码
- **THEN** 架构边界检查 MUST 失败

### Requirement: Legacy KD virtual 配置入口收窄
**Reason**: virtual KD 入口不再是“收窄”，而是完全删除。
**Migration**: 使用 supervised strong/lightweight canonical 配置。

#### Scenario: fusion KD virtual alias 被拒绝
- **WHEN** 用户请求 fusion KD virtual alias
- **THEN** 配置加载 MUST 失败并说明 KD support 已删除

### Requirement: no-KD 配置不携带 KD-only 超参
**Reason**: `no-KD` 配置命名和 `distillation` block 均被删除。
**Migration**: 使用 strong/lightweight/supervised 配置，且不包含 KD-only 超参。

#### Scenario: no-KD config 最小化
- **WHEN** 用户加载旧 `*_no_kd` 配置
- **THEN** 系统 MUST 拒绝旧路径或要求迁移到新配置名

