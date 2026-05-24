## ADDED Requirements

### Requirement: 本地实验运行索引
系统 MUST 提供只读实验运行索引能力，用于扫描本地 `outputs/`、`logs/` 和可选当前进程资源状态，并生成结构化 run summary。索引过程 MUST 不修改、删除、移动或压缩任何训练、评估、日志、checkpoint 或 cache 产物。

#### Scenario: 扫描输出目录
- **WHEN** 用户对 `outputs/` 运行 run index
- **THEN** 系统 MUST 发现包含 `final_config.yaml`、`resolved_config.yaml`、`startup_summary.json`、`metrics.json`、`train_log.json`、checkpoint 或 TensorBoard event 的 run 目录
- **AND** 每个 run summary MUST 记录 run_dir、dataset family、experiment name、task、objective、modalities、seed 和 artifact presence

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
