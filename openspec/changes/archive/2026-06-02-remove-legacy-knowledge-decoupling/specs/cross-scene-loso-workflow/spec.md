## MODIFIED Requirements

### Requirement: Quick validation 最小执行矩阵
系统 MUST 提供 HiST-Beam quick validation 的最小可执行矩阵。默认 CLI smoke MUST 是资源探针；完整方法验证 MUST 通过显式 quick validation 配置支持先执行 target scene `34`，并能用同一入口扩展到 target scenes `33`、`32` 和 `31`。Quick validation MUST NOT 默认计划旧 `v2_shared_private`、`v3_decoupled` 或等价旧简单 shared/private 解耦变体。

#### Scenario: target scene 34 快速验证
- **WHEN** 用户请求默认 quick validation 或显式配置 target scene `34`
- **THEN** 系统 MUST 生成并执行 target scene `34` 的 LOSO fold
- **AND** source scenes MUST 为 `[31, 32, 33]`，除非用户显式覆盖合法 source scenes

#### Scenario: resource smoke 默认保持轻量
- **WHEN** 用户未显式指定配置运行 `kd-sensing-hist-beam-loso`
- **THEN** 默认配置 MUST 只生成轻量资源探针矩阵
- **AND** 矩阵 MUST 能在单个 target scene、单个合法 variant、单个 budget 和单个 seed 上运行
- **AND** 配置 MUST 使用短 epoch 和小数据比例，避免默认 CLI 启动长方法验证矩阵

#### Scenario: method quick validation variants budgets seeds 最小覆盖
- **WHEN** 用户请求完整 quick validation 方法验证矩阵
- **THEN** 系统 MUST 覆盖当前合法 variants，例如 `v0_flat`、`v1_hierarchical`、`v4_adapter`、`v5_adapter_proto` 和 `v6_full_finetune`
- **AND** 系统 MUST NOT 覆盖 `v2_shared_private`、`shared_private`、`v3_decoupled` 或 `decoupled`
- **AND** 系统 MUST 覆盖 label budgets `0` 和 `10`
- **AND** 系统 MUST 覆盖 seed `0`

#### Scenario: DeepSense6G method quick validation 使用全量数据和 40 epoch
- **WHEN** 用户使用完整 HiST-Beam quick validation 配置运行 DeepSense6G LOSO execute
- **THEN** 配置 MUST 使用 `data.dataset.portion: 1.0`
- **AND** 配置 MUST 使用 `training.epochs: 40`
- **AND** `portion` MUST NOT 默认缩小 DeepSense6G 训练或测试数据

#### Scenario: 扩展到完整 31-34 target scenes
- **WHEN** 用户请求完整 quick validation scenes
- **THEN** 系统 MUST 运行 target scenes `34`、`33`、`32` 和 `31`
- **AND** 每个 target scene MUST 使用其余三个 scenes 作为 source scenes

#### Scenario: 用户可缩小矩阵
- **WHEN** 用户通过 CLI 或配置指定 variants、budgets、seeds 或 target scenes 的子集
- **THEN** 系统 MUST 只执行指定子集
- **AND** summary metadata MUST 记录矩阵被用户缩小后的实际组合

#### Scenario: 用户可限制 run 数量
- **WHEN** 用户通过 CLI 或配置指定最大 run 数量
- **THEN** 系统 MUST 只计划并执行该数量以内的 run
- **AND** plan metadata MUST 记录原始 planned run count 和实际 run count

## ADDED Requirements

### Requirement: LOSO 旧解耦 baseline 退役
LOSO workflow MUST NOT 将旧 `v3_decoupled` 或等价简单 shared/private 解耦路线作为默认 source checkpoint、summary comparison baseline、quick conclusion 主线或 prototype source fallback。需要 baseline 时，系统 MUST 使用现行合法 source-only、image-only legal、residual/calibration 或显式配置的非旧解耦 baseline。

#### Scenario: adaptation source variant 不回退到 v3
- **WHEN** runner 为 adapter/prototype/target-prior 变体选择 source checkpoint
- **THEN** source variant MUST 是合法的非旧解耦 variant 或用户显式指定的合法 source variant
- **AND** 系统 MUST NOT 自动返回 `v3_decoupled`

#### Scenario: summary comparison 不使用 v3 主 baseline
- **WHEN** LOSO summary 聚合多个 variant 的结果
- **THEN** comparison metadata MUST NOT 把 `v3_decoupled` 设为默认 baseline
- **AND** 缺少旧 `v3_decoupled` run MUST NOT 被记录为方法矩阵缺失
