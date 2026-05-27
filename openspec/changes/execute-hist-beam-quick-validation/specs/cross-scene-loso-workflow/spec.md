## ADDED Requirements

### Requirement: LOSO execute 执行闭环
系统 MUST 在用户运行 `kd-sensing-hist-beam-loso --execute` 时执行 LOSO quick validation stages，而不是仅生成 planned run。执行闭环 MUST 至少包含 source training、source-only target_test evaluation、target adaptation、adapted target_test evaluation 和 summary 写出。

#### Scenario: execute 不返回 planned 状态
- **WHEN** 用户使用合法配置运行 `kd-sensing-hist-beam-loso --execute`
- **THEN** 系统 MUST 进入真实执行路径
- **AND** 返回结果中的 `execution.status` MUST 为 `completed`、`failed` 或 `partial_failed`
- **AND** 返回结果 MUST NOT 使用 `planned` 表示 execute 模式已处理完成

#### Scenario: 按顺序执行单个 run 的 stages
- **WHEN** execute runner 处理一个 fold、variant、budget 和 seed 组合
- **THEN** 系统 MUST 按顺序执行 `source_train`、`source_only_target_test_eval`、`target_adaptation`、`adapted_target_test_eval` 和 `summary`
- **AND** 每个 stage 的状态、开始结束时间、输出路径和失败原因 MUST 写入 run metadata

#### Scenario: 保留 plan-only 行为
- **WHEN** 用户未传入 `--execute`
- **THEN** 系统 MUST 只写出 LOSO run plan
- **AND** 系统 MUST 不启动训练、adaptation 或 evaluation stage

### Requirement: LOSO execute preflight
系统 MUST 在启动训练或 adaptation 前执行 preflight。preflight MUST 检查矩阵中涉及的 DeepSense6G scene 数据根目录、CSV、启用模态资源、输出目录写权限和配置合法性。preflight 失败时 MUST 给出明确错误，不得写出表示成功的 summary。

#### Scenario: 无数据时明确失败
- **WHEN** execute 配置引用的 source scene 或 target scene 缺少数据根目录、必要 CSV 或启用模态资源
- **THEN** preflight MUST 失败
- **AND** 错误信息 MUST 包含缺失的 scene、资源类型和路径
- **AND** 系统 MUST 不启动 source training 或 target adaptation

#### Scenario: 输出目录不可写时明确失败
- **WHEN** execute 输出目录不存在且无法创建，或已存在但不可写
- **THEN** preflight MUST 失败
- **AND** 错误信息 MUST 指出不可写的输出目录

#### Scenario: preflight 成功后记录检查结果
- **WHEN** preflight 通过
- **THEN** 系统 MUST 保存 preflight metadata
- **AND** metadata MUST 记录检查过的 scenes、CSV、启用模态、输出目录和 quick validation matrix 摘要

### Requirement: Quick validation 最小执行矩阵
系统 MUST 提供 HiST-Beam quick validation 的最小可执行矩阵。默认 CLI smoke MUST 是资源探针；完整方法验证 MUST 通过显式 quick validation 配置支持先执行 target scene `34`，并能用同一入口扩展到 target scenes `33`、`32` 和 `31`。

#### Scenario: target scene 34 快速验证
- **WHEN** 用户请求默认 quick validation 或显式配置 target scene `34`
- **THEN** 系统 MUST 生成并执行 target scene `34` 的 LOSO fold
- **AND** source scenes MUST 为 `[31, 32, 33]`，除非用户显式覆盖合法 source scenes

#### Scenario: resource smoke 默认保持轻量
- **WHEN** 用户未显式指定配置运行 `kd-sensing-hist-beam-loso`
- **THEN** 默认配置 MUST 只生成轻量资源探针矩阵
- **AND** 矩阵 MUST 能在单个 target scene、单个 variant、单个 budget 和单个 seed 上运行
- **AND** 配置 MUST 使用短 epoch 和小数据比例，避免默认 CLI 启动长方法验证矩阵

#### Scenario: method quick validation variants budgets seeds 最小覆盖
- **WHEN** 用户请求完整 quick validation 方法验证矩阵
- **THEN** 系统 MUST 覆盖 variants `v0_flat`、`v3_decoupled`、`v4_adapter`、`v5_adapter_proto` 和 `v6_full_finetune`
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

### Requirement: LOSO execute summary 产物
系统 MUST 在 execute 结束后输出 LOSO summary CSV/JSON。summary MUST 汇总每个 run 的 stage 状态、metrics 路径、predictions 路径、checkpoint 来源、adaptation 效率指标和失败原因。

#### Scenario: 完成矩阵后写出 CSV 和 JSON
- **WHEN** execute runner 完成所有计划 run，或以 partial failure 结束
- **THEN** 系统 MUST 写出 `loso_summary.json`
- **AND** 系统 MUST 写出等价的 CSV summary 或记录 CSV 不可用原因

#### Scenario: summary 保留单次运行路径
- **WHEN** 某个 run 产生 metrics、predictions、checkpoint、prototype 或 metadata artifact
- **THEN** summary MUST 记录对应 artifact path
- **AND** summary MUST 能追溯到 fold、target scene、source scenes、variant、budget 和 seed

#### Scenario: summary 不伪造失败 run 指标
- **WHEN** 某个 run 或 stage 失败导致 metrics 缺失
- **THEN** summary MUST 将该 run 标记为 failed 或 missing
- **AND** 系统 MUST 不用 `0` 或其它数值伪造缺失指标

### Requirement: LOSO execute 进度与中断可诊断
系统 MUST 在 execute 过程中持续写出可诊断 metadata。用户手动中断时，系统 SHOULD 尽量写出 partial summary；无法捕获的系统级终止至少 MUST 已经写出最近开始的 run/stage metadata。

#### Scenario: stage 开始即写 running metadata
- **WHEN** execute runner 即将启动某个 stage
- **THEN** 系统 MUST 在 stage 进入训练或评估前写出 run-level `metadata.json`
- **AND** metadata MUST 标明当前 stage 为 `running`

#### Scenario: 长训练 stage 写 epoch 进度
- **WHEN** source training 或 target adaptation 完成一个 epoch
- **THEN** 系统 SHOULD 写出进度事件
- **AND** 进度事件 SHOULD 包含 stage、epoch、总 epoch、耗时和可用 loss 统计

#### Scenario: 用户中断后写 partial summary
- **WHEN** 用户通过可捕获的中断停止 execute
- **THEN** 当前 stage MUST 标记为 failed
- **AND** 未启动的计划 run MUST 标记为 missing
- **AND** 系统 MUST 写出 partial `loso_summary.json`、`loso_summary.csv` 和机器可读结论文件
