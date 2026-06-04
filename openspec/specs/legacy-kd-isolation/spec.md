# legacy-kd-isolation Specification

## Purpose
定义历史 teacher-student KD 入口退役后的拒绝、历史读取和 summary 隔离边界。

## Requirements

### Requirement: Legacy KD 入口必须拒绝
系统 MUST 不再运行 teacher-student KD。旧 `logits_kd`、`rkd`、`distillation.*`、`teacher_model_name` 或旧 `*_no_kd` 路径只允许作为 migration guard、历史 artifact 读取或 archive 说明中的命中。

#### Scenario: 旧 KD 配置被拒绝
- **WHEN** 用户请求旧 `logits_kd`、`rkd` 或等价 KD 配置
- **THEN** 配置加载 MUST 失败
- **AND** 错误信息 MUST 指向 strong、lightweight、supervised 或 adaptation 入口

#### Scenario: 旧 distillation override 被拒绝
- **WHEN** 用户通过命令行传入 `distillation.*`、`teacher_model_name` 或 `kd_mode`
- **THEN** 配置加载 MUST 失败
- **AND** 系统 MUST 不构建 frozen teacher 或 distiller

### Requirement: 历史 KD metadata 只读隔离
历史 run metadata MAY 继续包含 `distillation_enabled` 或 `method_family=legacy_kd`，但新训练产物 MUST 不写出这些字段作为当前运行模式。读取历史 summary 时，系统 MUST 将这些记录标记为历史或 supplemental，不得纳入当前主结论排名。

#### Scenario: 历史 KD run 被识别
- **WHEN** summary 读取一个历史 `method_family=legacy_kd` 或 `distillation_enabled=true` 的 run
- **THEN** 系统 MAY 展示该 run 作为历史资料
- **AND** 系统 MUST 不把该 run 计入当前 mainline ranking

#### Scenario: 新训练产物不写 KD lineage
- **WHEN** 当前 supervised/adaptation 训练完成
- **THEN** run metadata MUST 不新增 `method_family=legacy_kd`
- **AND** run metadata MUST 不新增 `distillation_enabled=true`
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
