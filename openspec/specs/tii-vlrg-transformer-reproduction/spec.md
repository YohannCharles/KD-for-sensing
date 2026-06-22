# tii-vlrg-transformer-reproduction Specification

## Purpose
TBD - created by archiving change reproduce-tii-vlrg-transformer-baseline. Update Purpose after archive.
## Requirements
### Requirement: TII baseline manifest
系统 MUST 提供 TII VLRG Transformer baseline reproduction manifest，用于声明外部源码、checkpoint、输入模态、数据协议、metric 口径、输出路径和复现状态。

#### Scenario: 生成 dry-run manifest
- **WHEN** 用户以 dry-run 模式运行 TII baseline reproduction 入口
- **THEN** 系统 MUST 写出 machine-readable manifest
- **AND** manifest MUST 至少包含 model_id、source_repo、source_commit、enabled_modalities、scene_set、split、metric_profile、output_root、status 和 warnings

#### Scenario: 缺失外部 artifact 不升级 claim
- **WHEN** source repo、checkpoint、预处理产物或 prediction 文件缺失
- **THEN** manifest MUST 将状态标记为 `pending`、`unavailable` 或 `blocked`
- **AND** 系统 MUST NOT 将该 run 写成真实性能 claim

### Requirement: TII 外部 workflow 执行边界
TII baseline reproduction MUST 作为 workflow/paper reproduction 处理。实现 MUST 使用包内 owner 或 CLI 包装外部预处理、推理和指标读取，不得复制当前通用训练循环或新增旧式根脚本。

#### Scenario: 入口使用项目环境
- **WHEN** TII wrapper 执行任何项目 Python 命令
- **THEN** 命令 MUST 使用 `conda run -n kd_mm_beam`
- **AND** stdout、stderr、manifest、metrics 和预测产物 MUST 写入 ignored output root

#### Scenario: 显式执行外部 workflow
- **WHEN** 用户显式传入 execute 模式并提供外部 repo / artifact 路径
- **THEN** wrapper MUST 执行 manifest 中记录的 list-form 外部命令
- **AND** 每个命令 MUST 记录 stdout、stderr、returncode 和 stage 到 ignored output root
- **AND** 任一命令失败时 manifest MUST 标记为 `blocked`，不得生成真实性能 claim

#### Scenario: 不恢复旧入口
- **WHEN** 实现新增 TII baseline 命令入口
- **THEN** 入口 MUST 是包内 CLI 或 package console script
- **AND** 系统 MUST NOT 新增 root-level legacy training script、package-level 聚合 facade 或 retired 研究线 wrapper

### Requirement: TII 指标适配
系统 MUST 能将 TII clean 指标或导入的预测结果适配为本仓库统一 DBA summary row。适配结果 MUST 保留 provenance 和 strict comparability 字段。

#### Scenario: 导入 TII metrics CSV
- **WHEN** 用户提供 TII metrics CSV、prediction CSV 或等价 summary artifact
- **THEN** 系统 MUST 输出包含 model、source、overall_clean、DBA 或 P0-P5 字段的统一 summary row
- **AND** 输出 MUST 记录 source artifact path、source commit、checkpoint provenance 和 metric profile

#### Scenario: comparability mismatch 阻止 strict ranking
- **WHEN** TII row 的 split、scene set、label space、metric profile、history window、GPS source window、prediction horizon、seed 或 difficulty digest 与当前 heatmap strict protocol 不一致
- **THEN** 系统 MUST 将该 row 标记为 not_comparable 或 external_reference
- **AND** 该 row MUST NOT 进入 strict ranking 或 claim upgrade

### Requirement: TII 产物边界
TII baseline reproduction 的外部源码副本、下载 checkpoint、预处理 cache、log、prediction、metrics 和图表 MUST 位于 ignored runtime output 目录或用户显式指定的本地路径。

#### Scenario: 运行产物不进入源码
- **WHEN** TII baseline reproduction 生成 checkpoint、cache、log、prediction 或 metrics
- **THEN** 这些产物 MUST NOT 写入 tracked source path
- **AND** manifest MUST 使用可审计路径和 fingerprint 指向这些产物

#### Scenario: 单元测试不依赖真实数据
- **WHEN** focused tests 验证 TII baseline reproduction
- **THEN** tests MUST 使用 synthetic manifest、dry-run command 或 small fixture metrics
- **AND** tests MUST NOT 读取真实 `dataset/`、外部 checkpoint 或下载 repo

