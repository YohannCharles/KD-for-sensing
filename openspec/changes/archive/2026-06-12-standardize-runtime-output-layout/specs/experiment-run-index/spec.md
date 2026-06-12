## MODIFIED Requirements

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
