# experiment-run-index Specification

## Purpose
定义只读实验运行索引能力，用于汇总本地 outputs/logs 中的训练与评估 run、状态、关键 artifact、资源快照和机器可读报告，同时保证索引过程不移动、删除或重写任何本地产物，便于排查长时间训练、失败 run 与复现实验记录。
## Requirements
### Requirement: 本地实验运行索引
系统 MUST 提供只读实验运行索引能力，用于扫描本地 `outputs/`、`logs/` 和可选当前进程资源状态，并生成结构化 run summary。索引过程 MUST 不修改、删除、移动或压缩任何训练、评估、日志、checkpoint 或 cache 产物。默认扫描 `outputs/` 时，系统 MUST 理解 canonical output layout，并默认跳过 `outputs/cache/`、`outputs/archive/`、`outputs/cleanup_manifests/` 等非当前 run 分区，除非用户显式把这些路径作为扫描根。

#### Scenario: 扫描输出目录
- **WHEN** 用户对 `outputs/` 运行 run index
- **THEN** 系统 MUST 发现 canonical scene、scenegroup、evaluation 和 analysis 分区中包含 `final_config.yaml`、`resolved_config.yaml`、`startup_summary.json`、`metrics.json`、`train_log.json`、checkpoint 或 TensorBoard event 的 run 目录
- **AND** 每个 run summary MUST 记录 run_dir、dataset family、experiment name、task、objective、modalities、seed、scene scope 和 artifact presence

#### Scenario: 默认跳过 cache 和 archive
- **WHEN** 用户使用默认 outputs root 构建 run index
- **THEN** 系统 MUST 不递归扫描 `outputs/cache/`、`outputs/archive/` 或 `outputs/cleanup_manifests/`
- **AND** warnings 或 roots metadata MUST 记录被默认跳过的非 run 分区

#### Scenario: 显式扫描非 run 分区
- **WHEN** 用户显式传入 `--outputs outputs/cache` 或 `--outputs outputs/archive`
- **THEN** 系统 MAY 扫描该路径
- **AND** summary MUST 标记该扫描根是显式请求的非默认分区

#### Scenario: 忽略源码边界之外的删除操作
- **WHEN** run index 读取本地 run 目录和日志目录
- **THEN** 系统 MUST 不删除、不重写、不移动任何 `outputs/`、`logs/`、checkpoint、cache 或 TensorBoard 文件
- **AND** 输出文件若启用 MUST 写入用户显式指定路径或 ignored 输出目录

### Requirement: Run state 分类
系统 MUST 为每个 run 输出稳定的 `state` 字段。状态集合 MUST 至少包含 `running`、`complete`、`started_no_metrics`、`partial`、`failed`、`killed`、`waiting`、`stale` 和 `unknown`。状态推断 MUST 同时考虑当前进程、状态 sidecar、关键 artifact 和关联日志。

#### Scenario: 完整训练 run
- **WHEN** run 目录包含 `metrics.json`、`train_log.json`、`training_outputs.npz`、`final_config.yaml` 和 `resolved_config.yaml`
- **THEN** run index MUST 将该 run 分类为 `complete`，除非当前进程或状态 sidecar 明确显示仍在运行
- **AND** summary MUST 记录主要 metrics 和 best checkpoint 路径

#### Scenario: 启动后无指标
- **WHEN** run 目录包含 `startup_summary.json`、`final_config.yaml` 和 `resolved_config.yaml` 但缺少 `metrics.json`
- **THEN** run index MUST 将该 run 分类为 `started_no_metrics`、`running` 或 `stale`
- **AND** 若存在匹配的当前训练进程，状态 MUST 为 `running`
- **AND** 若无匹配进程且 artifact 长时间未更新，状态 MUST 为 `stale`

#### Scenario: 日志显示 Killed
- **WHEN** 关联日志包含 shell 或 conda 输出的 `Killed`
- **THEN** run index MUST 将失败原因记录为 killed
- **AND** 若无匹配当前进程且 run 未完整完成，状态 MUST 为 `killed`

#### Scenario: 等待前置 checkpoint
- **WHEN** 当前 shell 或日志显示 run 正在等待某个 checkpoint 文件出现
- **THEN** run index MUST 将状态标记为 `waiting`
- **AND** summary MUST 记录等待条件或等待文件路径

### Requirement: Run summary 输出
系统 MUST 支持将 run index 输出为机器可读 JSON，并 SHOULD 提供人类可读表格和 CSV 输出。JSON schema MUST 稳定包含 `generated_at`、`roots`、`runs`、`resources` 和 `warnings` 顶层字段。

#### Scenario: 输出 JSON
- **WHEN** 用户请求 JSON 输出
- **THEN** 系统 MUST 输出包含所有匹配 run summary 的 JSON
- **AND** 每个 run summary MUST 包含 `state`、`state_reason`、`artifacts`、`metrics`、`config` 和 `timestamps`

#### Scenario: 按条件过滤
- **WHEN** 用户指定 dataset family、objective、state、run name 或时间范围过滤条件
- **THEN** run index MUST 只输出匹配 run
- **AND** summary metadata MUST 记录实际采用的过滤条件

### Requirement: 资源快照
系统 MUST 在可用时采集当前系统资源快照，用于解释正在运行的训练状态。资源快照 MUST 安全降级，不得因为缺少 `nvidia-smi`、权限不足或某个进程退出而导致 run index 失败。

#### Scenario: GPU 资源可用
- **WHEN** 当前机器可执行 `nvidia-smi`
- **THEN** run index MUST 记录 GPU index、名称、显存总量、显存使用、GPU 利用率和可识别的 Python 训练进程 PID
- **AND** 对匹配 run 的 summary SHOULD 记录关联 GPU index 和显存使用

#### Scenario: GPU 资源不可用
- **WHEN** 当前机器没有 GPU 或无法执行 `nvidia-smi`
- **THEN** run index MUST 继续输出 run summary
- **AND** resources 字段 MUST 标记 GPU snapshot unavailable 的原因

### Requirement: Run index CLI
系统 MUST 提供一个可通过 `kd_mm_beam` 环境运行的命令入口，用于生成 run index。该入口 MUST 支持 `--outputs`、`--logs`、`--format`、`--state` 和 `--output` 参数。

#### Scenario: 查看帮助信息
- **WHEN** 开发者执行 `conda run -n kd_mm_beam <run-index-entrypoint> --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 包含 outputs、logs、format、state 和 output 参数说明

#### Scenario: 写出汇总文件
- **WHEN** 用户指定 `--output outputs/analysis/run_index.json`
- **THEN** 系统 MUST 将 JSON 或 CSV 汇总写入该路径
- **AND** 命令行 MUST 输出写出路径或简短摘要

### Requirement: 运行索引提供清理摘要
实验运行索引 MUST 保持只读，并为本地产物清理提供结构化摘要。每个 run summary SHOULD 包含 run 目录大小、checkpoint 文件数量、checkpoint 总大小、日志关联、最近更新时间、状态、可复现关键 artifact 是否存在和清理候选理由。

#### Scenario: run index 输出大小与 checkpoint 摘要
- **WHEN** 用户构建 run index 并扫描 `outputs/`
- **THEN** 每个 run summary MUST 记录 run 目录总大小
- **AND** 如果存在 checkpoint，summary MUST 记录 checkpoint 数量、总大小和主要 checkpoint 路径

#### Scenario: run index 保持只读
- **WHEN** 清理 manifest 生成流程复用 run index
- **THEN** run index MUST 不删除、不移动、不压缩、不重写任何输出、日志、checkpoint 或 cache
- **AND** run index MUST 仅返回结构化摘要

### Requirement: 活跃运行保护信号
实验运行索引 MUST 为清理流程提供活跃运行保护信号。状态为 `running`、`waiting` 或最近仍在更新且无法判定完成的 run MUST 被清理 manifest 标记为 protected 或 high-risk。

#### Scenario: running run 受保护
- **WHEN** run index 通过状态文件、进程或日志判断某个 run 仍在运行
- **THEN** 清理 manifest MUST 不将该 run 列为默认可删除候选
- **AND** manifest MUST 记录活跃运行保护原因

#### Scenario: stale run 可进入人工确认候选
- **WHEN** run index 判断某个 run 超过 stale 阈值且缺少完成指标
- **THEN** 清理 manifest MAY 将该 run 列为人工确认候选
- **AND** manifest MUST 记录 stale 阈值和缺失 artifact 摘要

### Requirement: Run index legacy 扫描必须右尺寸化
运行索引 MUST 保持只读、状态分类、过滤、输出和资源快照能力，但不应继续扩张只服务历史研究线考古的默认扫描分支。默认扫描 MUST 以当前 canonical output layout 为中心；历史 archive 或非 run 分区只在显式扫描或 cleanup/organize 需要时处理。

#### Scenario: 默认扫描关注 current layout
- **WHEN** 用户对 `outputs/` 构建 run index
- **THEN** 系统 MUST 扫描 canonical scene、scenegroup、evaluation 和 analysis run 位置
- **AND** 系统 MUST 默认跳过 cache、archive、cleanup manifest 和其它非 run 分区，除非用户显式指定扫描根

#### Scenario: 删除 legacy-only 分支不影响状态分类
- **WHEN** 实现删除某个只服务历史目录命名的 discovery 分支
- **THEN** complete、partial、running、waiting、killed、stale 和 unknown 状态分类 MUST 对 current run 继续可用
- **AND** run index tests MUST 覆盖当前状态分类

### Requirement: Run index 不承担 runtime cleanup 规则库
Run index MUST 提供清理流程需要的结构化摘要，但 MUST 不成为历史删除规则库。清理候选规则 MUST 留在 runtime cleanup owner，run index 只返回 run 状态、artifact、大小和时间摘要。

#### Scenario: cleanup 复用 run summary
- **WHEN** runtime cleanup 生成 manifest 并复用 run index
- **THEN** run index MUST 只返回只读 summary
- **AND** 删除候选规则、保护判断和 action plan MUST 由 cleanup owner 决定

### Requirement: Run index claim-harvester fields
实验运行索引 MUST 提供 claim harvester 可消费的稳定字段，但 MUST 不承担 claim 判定规则库。新增字段 MUST 保持只读，并且缺失时以 warning 或空值表达。

#### Scenario: run summary 包含 identity 和 artifact paths
- **WHEN** run index 扫描到一个训练或评估 run
- **THEN** run summary MUST 包含 run_name、run_dir、config_path、config_digest、seed、scene_scope、dataset_family、metric_profile、target_source 和 artifact path 摘要
- **AND** 如果字段无法解析，summary MUST 保留 run 基本状态并记录 warning

#### Scenario: run summary 包含 eval artifacts
- **WHEN** run index 扫描到 evaluation 或 missing-pattern 输出
- **THEN** summary MUST 记录 eval artifact 类型、CSV/JSON path、mtime、size 和关联 run_name
- **AND** run index MUST 不解析 claim readiness 或统计显著性

#### Scenario: 当前进程关联 run
- **WHEN** run index 发现当前训练进程
- **THEN** summary SHOULD 记录 config path、run name、PID、GPU index 和 command line
- **AND** dashboard MAY 使用这些字段展示 running 状态

### Requirement: Run index 二级热点必须按 scanner/collector/writer 拆分
Experiment run index 重构 MUST 拆分 output/log scanning、process/resource collection、artifact summarization、table rendering 和 JSON/CSV writing，并保持 public CLI output schema。

#### Scenario: run index 输出兼容
- **WHEN** `kd-sensing-runs` is run after refactor
- **THEN** JSON output MUST 保留 `generated_at`、`roots`、`runs`、`resources` 和 `warnings`
- **AND** default skipped output partitions MUST remain compatible

