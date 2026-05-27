# cross-scene-loso-workflow Specification

## Purpose
TBD - created by archiving change add-hist-beam-cross-scene-adaptation. Update Purpose after archive.
## Requirements
### Requirement: DeepSense6G 31-34 LOSO fold 定义
系统 MUST 提供 DeepSense6G scenarios 31、32、33、34 的 leave-one-scene-out fold 定义。每个 fold MUST 包含三个 source scenes 和一个 target scene，并 MUST 可通过配置或 orchestration 入口选择单个 fold 或全部 fold。

#### Scenario: 生成四个 LOSO fold
- **WHEN** 用户请求 DeepSense6G 31-34 的默认 LOSO folds
- **THEN** 系统 MUST 生成 target scene 34、33、32、31 各一次的四个 fold
- **AND** 每个 fold 的 source scenes MUST 等于其余三个场景

#### Scenario: 选择单个 target scene
- **WHEN** 用户配置 `target_scene: 34`
- **THEN** 系统 MUST 将 source scenes 解析为 `[31, 32, 33]`，除非用户显式覆盖 source scenes
- **AND** fold metadata MUST 记录 target scene、source scenes 和 fold id

#### Scenario: 拒绝 source target 重叠
- **WHEN** 用户显式配置的 source scenes 包含 target scene
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 指出 source/target scene 不得重叠

### Requirement: Target adapt/test split 防泄漏
系统 MUST 将每个 target scene 的可用 target split 确定性拆分为 `target_adapt` 和 `target_test`。默认比例 MUST 为 20% target_adapt 和 80% target_test。`target_test` MUST 只用于最终测试，不得参与训练、adaptation、early stopping、threshold selection、prototype selection 或 normalizer/scaler fit。

#### Scenario: target split 可复现
- **WHEN** 用户使用相同 target scene、split seed 和 split 配置构建 target adapt/test split
- **THEN** 系统 MUST 产生相同的 target_adapt 和 target_test 样本集合
- **AND** split metadata MUST 记录 split seed、比例、样本数和样本选择口径

#### Scenario: target_test 不参与 adaptation
- **WHEN** 用户运行 source training 或 target adaptation
- **THEN** 系统 MUST 不把 target_test 样本放入训练或 adaptation dataloader
- **AND** 系统 MUST 不使用 target_test label 计算 supervised loss、prototype target 或 early stopping 指标

#### Scenario: adapt/test 样本无交集
- **WHEN** target adapt/test split 构建完成
- **THEN** `target_adapt` 和 `target_test` 的 sample id MUST 无交集
- **AND** 若数据包含 `seq_index`，系统 MUST 优先保证二者的 `seq_index` 无交集

#### Scenario: split metadata 写入运行产物
- **WHEN** LOSO source training、adaptation 或 evaluation 创建运行目录
- **THEN** 系统 MUST 保存 fold metadata 和 target split metadata
- **AND** metadata MUST 足以复现实验使用的 source、target_adapt 和 target_test 样本集合

### Requirement: Source multi-scene 数据加载
系统 MUST 能为 LOSO source training 构建由多个 DeepSense6G scenes 组成的训练数据。每个底层 scene dataset MUST 继续遵守现有模态按需读取、场景 metadata 和本地产物边界。

#### Scenario: 构建三 source scene 训练集
- **WHEN** fold 的 source scenes 为 `[31, 32, 33]`
- **THEN** source train dataloader MUST 覆盖三个 scene 的训练样本
- **AND** batch 或 sample metadata MUST 保留每个样本的 scene id

#### Scenario: normalizer 只从允许训练数据拟合
- **WHEN** source multi-scene dataloader 需要 GPS、LiDAR、mmWave、CSI 或其它 normalizer/scaler artifact
- **THEN** 系统 MUST 只从 source train 或配置允许的训练 split 拟合 artifact
- **AND** 系统 MUST 将 artifact 复用于 target_adapt 和 target_test 读取

#### Scenario: 未启用模态不读取对应资源
- **WHEN** LOSO 配置只启用 `image`、`radar`、`gps`
- **THEN** dataset MUST 不读取 LiDAR、mmWave 或 CSI 资源
- **AND** 缺失未启用模态文件不得阻止该 LOSO run

### Requirement: Few-shot target label budget 采样
系统 MUST 支持 target label budgets `0`、`5`、`10`、`20`、`50`。当 budget 大于 0 时，系统 MUST 从 `target_adapt` 中选择 labeled subset，并 MUST 保留其余 target_adapt 样本作为 unlabeled subset。默认采样 MUST 优先 coarse group stratified sampling。

#### Scenario: label_budget 为 0
- **WHEN** 用户配置 `label_budget: 0`
- **THEN** 系统 MUST 不产生 labeled target subset
- **AND** target_adapt 样本 MUST 只作为 unlabeled adaptation 数据使用

#### Scenario: coarse group 分层采样
- **WHEN** `label_budget` 大于 0 且 target_adapt 中存在多个 coarse group
- **THEN** 系统 MUST 优先选择覆盖不同 coarse group 的 labeled samples
- **AND** labeled sampling manifest MUST 记录每个 labeled sample 的 beam 和 coarse group

#### Scenario: 样本不足时退化
- **WHEN** target_adapt 可用样本数小于请求的 label budget
- **THEN** 系统 MUST 使用全部可用 target_adapt 样本作为 labeled subset
- **AND** sampling metadata MUST 记录 requested budget、actual labeled count 和退化原因

#### Scenario: sampling seed 可复现
- **WHEN** 用户使用相同 fold、budget 和 seed 重复采样
- **THEN** 系统 MUST 产生相同 labeled sample id 集合
- **AND** 不同 seed MAY 产生不同 labeled sample id 集合

### Requirement: LOSO 运行编排入口
系统 MUST 提供配置驱动的包内 LOSO orchestration 入口，用于依次运行 source training、source-only target_test evaluation、target adaptation、adapted target_test evaluation 和结果汇总。入口 MUST 使用 `kd_sensing.cli` 或包内模块，不得新增长期维护的根目录脚本。

#### Scenario: 运行单 fold 单 variant
- **WHEN** 用户通过 LOSO 入口指定一个 target scene、一个 variant、一个 seed 和一个 budget
- **THEN** 系统 MUST 只运行对应 fold 和配置组合
- **AND** 输出目录 MUST 包含 source、adaptation、evaluation 和 summary metadata

#### Scenario: 运行默认快速矩阵
- **WHEN** 用户请求默认 HiST-Beam quick verification matrix
- **THEN** 系统 MUST 能遍历四个 LOSO folds、配置的 seeds、配置的 variants 和配置的 label budgets
- **AND** 系统 MUST 允许用户用配置缩小 variants、budgets 或 folds 以进行 smoke test

#### Scenario: 复用已有 source checkpoint
- **WHEN** 指定 fold 和 seed 的 source checkpoint 已存在且配置允许复用
- **THEN** orchestration MUST 能跳过 source retraining 并复用该 checkpoint
- **AND** summary metadata MUST 记录 checkpoint 来源和复用行为

#### Scenario: 不覆盖既有运行产物
- **WHEN** LOSO 入口创建输出目录且目标目录已存在
- **THEN** 系统 MUST 遵守现有输出覆盖和唯一目录规则
- **AND** 未显式 overwrite 时 MUST 不覆盖已有 metrics、checkpoint、predictions 或 prototype artifact

### Requirement: LOSO 结果汇总
系统 MUST 为 HiST-Beam 快速验证输出 source-only、few-shot adaptation 和 efficiency 三类汇总表或等价 JSON/CSV。汇总 MUST 能按 fold、target scene、variant、budget 和 seed 聚合，并 MUST 记录均值与可追溯的单次运行路径。

#### Scenario: 输出 source-only 表
- **WHEN** V0、V1、V2 和 V3 source-only evaluation 完成
- **THEN** 汇总 MUST 包含每个 target scene 的 Top-1、Top-3 和 coarse accuracy
- **AND** 汇总 MUST 包含跨 fold 平均指标

#### Scenario: 输出 few-shot adaptation 表
- **WHEN** source-only V3、full fine-tuning、adapter-only 和 adapter+prototype evaluation 完成
- **THEN** 汇总 MUST 按 label budget 聚合 Top-1、Top-3、Top-5 和 coarse accuracy
- **AND** 汇总 MUST 保留每个 seed 的原始指标路径

#### Scenario: 输出 efficiency 表
- **WHEN** adaptation variants 完成
- **THEN** 汇总 MUST 包含 trainable params、trainable ratio、adapt time per epoch、total adapt time 和 target_test Top-1
- **AND** adapter variants MUST 能与 full fine-tuning baseline 横向比较

#### Scenario: 汇总不伪造缺失指标
- **WHEN** 某个 run 未产生 power metrics 或 prototype metrics
- **THEN** 汇总 MUST 将对应字段标记为不可用或缺失
- **AND** 系统 MUST 不用 0 或其它数值伪造真实指标

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
